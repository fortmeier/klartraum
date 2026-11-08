// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Matrix-vector products y[t] = W x[t] with GGUF-encoded weights, for up to
// MAX_TOKENS tokens per dispatch (params[0] tokens are active). W holds
// `rows` rows of `columns` values each, row r starting at byte r * rowBytes,
// in the GPU block layout of gguf_layers::packWeights(): the GGUF blocks,
// with Q3_K padded to 112 bytes, Q6_K to 224 bytes, and the Q8_0 scale padded
// to 4 bytes, so every field read below is aligned.
//
// Each 32-wide subgroup computes ROWS_PER_SUBGROUP rows; every lane decodes 16
// consecutive weights (within one sub-block) per step, so a subgroup covers
// 512 values per step and reuses each decoded weight for TOKEN_BLOCK tokens.
// The shaders pick the decoder of their weight format.
//
// With `accumulate`, the products are added to y (a residual stream updated
// in place). With `lastTokenOnly`, only the last active token is computed and
// written to y row 0.
module matvec;

import subgroup;
__exported import gguf_decode;

static const uint MAX_TOKENS = 16;
static const uint ROWS_PER_SUBGROUP = 4;
static const uint TOKEN_BLOCK = 4;
public static const uint SUBGROUPS = 2;
static const uint WIDTH = 16u;

struct PushConstants {
    uint rows;
    uint columns;
    uint rowBytes;
    uint accumulate;
    uint lastTokenOnly;
    uint inputStride;  // floats between the inputs of consecutive tokens
    uint outputStride; // floats between the outputs of consecutive tokens
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(1, 0)]] StructuredBuffer<float4> X;
[[vk::binding(2, 0)]] StructuredBuffer<uint> params;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> Y;

// Runs in a 32 * SUBGROUPS wide workgroup.
public void matvec<Decoder : IWeightDecoder>(uint3 groupId) {
    const uint group = groupId.x + groupId.y * WorkgroupCount().x;
    const uint firstRow = (group * SUBGROUPS + subgroupId()) * ROWS_PER_SUBGROUP;
    if (firstRow >= pc.rows) return;
    const uint lane = WaveGetLaneIndex();
    const uint tokens = params[0];
    const uint firstToken = pc.lastTokenOnly != 0u ? tokens - 1u : 0u;
    const uint activeTokens = tokens - firstToken;

    // Tokens are processed in blocks of TOKEN_BLOCK with fixed-size
    // accumulators (registers); each block decodes the weights again.
    for (uint block = 0u; block < activeTokens; block += TOKEN_BLOCK) {
        const uint blockTokens = min(TOKEN_BLOCK, activeTokens - block);
        float sums[ROWS_PER_SUBGROUP][TOKEN_BLOCK];
        for (uint r = 0u; r < ROWS_PER_SUBGROUP; ++r) {
            for (uint t = 0u; t < TOKEN_BLOCK; ++t) sums[r][t] = 0.0;
        }
        for (uint element = WIDTH * lane; element < pc.columns; element += 32u * WIDTH) {
            float4 w[ROWS_PER_SUBGROUP][4];
            for (uint r = 0u; r < ROWS_PER_SUBGROUP; ++r) {
                const uint row = min(firstRow + r, pc.rows - 1u);
                Decoder.decode16(row * pc.rowBytes, element, w[r]);
            }
            for (uint t = 0u; t < TOKEN_BLOCK; ++t) {
                if (t < blockTokens) {
                    const uint base = ((firstToken + block + t) * pc.inputStride + element) / 4u;
                    for (uint k = 0u; k < 4u; ++k) {
                        const float4 x = X[base + k];
                        for (uint r = 0u; r < ROWS_PER_SUBGROUP; ++r) sums[r][t] += dot(w[r][k], x);
                    }
                }
            }
        }
        for (uint t = 0u; t < TOKEN_BLOCK; ++t) {
            if (t >= blockTokens) break;
            const uint outputToken = pc.lastTokenOnly != 0u ? 0u : block + t;
            for (uint r = 0u; r < ROWS_PER_SUBGROUP; ++r) {
                const float total = WaveActiveSum(sums[r][t]);
                const uint row = firstRow + r;
                if (lane == 0u && row < pc.rows) {
                    const uint index = outputToken * pc.outputStride + row;
                    Y[index] = pc.accumulate != 0u ? Y[index] + total : total;
                }
            }
        }
    }
}
