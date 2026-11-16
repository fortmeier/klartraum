// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Matrix products Y = X W^T (+ bias) with GGUF-encoded weights for many tokens at once, e.g. the image tokens of a
// diffusion transformer: X holds `tokens` rows of `columns` values, W `rows` rows of `columns` values in the GPU block
// layout of gguf_layers::packWeights(), row r starting at byte r * rowBytes; Y gets `tokens` rows of `rows` values.
//
// A workgroup of 256 invocations computes a tile of TILE_TOKENS tokens x TILE_ROWS rows. It walks the columns in
// steps of TILE_COLUMNS: half of the invocations decode 16 weights each into shared memory (64 rows x 32 columns), the
// other half load 16 inputs each (64 tokens x 32 columns); then every invocation accumulates 4 tokens x 4 rows from
// shared memory. The shaders pick the decoder of their weight format.
//
// With `accumulate`, the products are added to Y (a residual stream updated in place).
module matmul;

__exported import gguf_decode;

public static const uint THREADS = 256u;
static const uint TILE_TOKENS = 64u;
static const uint TILE_ROWS = 64u;
static const uint TILE_COLUMNS = 32u;

struct PushConstants {
    uint rows;
    uint columns;
    uint rowBytes;
    uint tokens;
    uint accumulate;
    uint hasBias;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(1, 0)]] StructuredBuffer<float> X;  // [tokens][columns]
[[vk::binding(2, 0)]] StructuredBuffer<float> bias; // [rows], read with hasBias
[[vk::binding(3, 0)]] RWStructuredBuffer<float> Y; // [tokens][rows]

// Padded by one column against bank conflicts.
groupshared float weights[TILE_ROWS][TILE_COLUMNS + 1];
groupshared float inputs[TILE_TOKENS][TILE_COLUMNS + 1];

// Runs in a THREADS wide workgroup; group x tiles the rows, group y the tokens.
public void matmul<Decoder : IWeightDecoder>(uint3 groupId, uint local) {
    const uint firstRow = groupId.x * TILE_ROWS, firstToken = groupId.y * TILE_TOKENS;
    // This invocation's outputs: tokens 4 (local % 16) to 3 more, rows 4 (local / 16) to 3 more.
    const uint tokenQuad = local % 16u, rowQuad = local / 16u;
    float sums[4][4];
    [ForceUnroll]
    for (uint t = 0u; t < 4u; ++t)
        [ForceUnroll]
        for (uint r = 0u; r < 4u; ++r) sums[t][r] = 0.0;

    // Invocations 0..127 decode the weights of row half / 2, columns 16 (half % 2) on; 128..255 load the inputs of
    // token half / 2, columns 16 (half % 2) on.
    const bool decodes = local < 128u;
    const uint half_ = local % 128u, loadIndex = half_ / 2u, loadColumn = 16u * (half_ % 2u);
    const uint weightRow = min(firstRow + loadIndex, pc.rows - 1u);
    const uint token = min(firstToken + loadIndex, pc.tokens - 1u);

    for (uint start = 0u; start < pc.columns; start += TILE_COLUMNS) {
        if (decodes) {
            float4 w[4];
            Decoder.decode16(weightRow * pc.rowBytes, start + loadColumn, w);
            [ForceUnroll]
            for (uint k = 0u; k < 4u; ++k)
                [ForceUnroll]
                for (uint c = 0u; c < 4u; ++c) weights[loadIndex][loadColumn + 4u * k + c] = w[k][c];
        } else {
            [ForceUnroll]
            for (uint c = 0u; c < 16u; ++c)
                inputs[loadIndex][loadColumn + c] = X[token * pc.columns + start + loadColumn + c];
        }
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll]
        for (uint c = 0u; c < TILE_COLUMNS; ++c) {
            float x[4], w[4];
            [ForceUnroll]
            for (uint t = 0u; t < 4u; ++t) x[t] = inputs[4u * tokenQuad + t][c];
            [ForceUnroll]
            for (uint r = 0u; r < 4u; ++r) w[r] = weights[4u * rowQuad + r][c];
            [ForceUnroll]
            for (uint t = 0u; t < 4u; ++t)
                [ForceUnroll]
                for (uint r = 0u; r < 4u; ++r) sums[t][r] += x[t] * w[r];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    [ForceUnroll]
    for (uint t = 0u; t < 4u; ++t) {
        const uint outputToken = firstToken + 4u * tokenQuad + t;
        if (outputToken >= pc.tokens) continue;
        [ForceUnroll]
        for (uint r = 0u; r < 4u; ++r) {
            const uint row = firstRow + 4u * rowQuad + r;
            if (row >= pc.rows) continue;
            float value = sums[t][r];
            if (pc.hasBias != 0u) value += bias[row];
            const uint index = outputToken * pc.rows + row;
            Y[index] = pc.accumulate != 0u ? Y[index] + value : value;
        }
    }
}
