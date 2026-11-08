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

// The two halves of a word as floats (GLSL unpackHalf2x16).
float2 unpackHalf(uint word) { return float2(f16tof32(word), f16tof32(word >> 16)); }

// Byte i (0..15) of four words.
uint byteOf(uint4 words, uint i) { return (words[i >> 2] >> ((i & 3u) * 8u)) & 0xffu; }

// The (scale, min) pair `index` of the 12 packed 6-bit scales of Q4_K/Q5_K,
// which are bytes 4..15 of the block's first 16 bytes `head`.
void packedScaleMin(uint4 head, uint index, out float scale, out float minimum) {
    if (index < 4u) {
        scale = float(byteOf(head, 4u + index) & 63u);
        minimum = float(byteOf(head, 8u + index) & 63u);
    } else {
        const uint packed = byteOf(head, 8u + index);
        scale = float((packed & 15u) | ((byteOf(head, index) >> 6) << 4));
        minimum = float((packed >> 4) | ((byteOf(head, 4u + index) >> 6) << 4));
    }
}

// The four bytes of a word as floats 0..255 (exact: unpackUnorm4x8 divides by 255).
float4 bytes4(uint word) { return unpackUnorm4x8ToFloat(word) * 255.0; }

// Decodes the 16 weights of a row starting at element `element` (a multiple
// of 16), four per vector. Each shader declares the weight buffer (binding 0)
// in the view its format reads, words for Q8_0 and 16-byte vectors otherwise,
// and passes it to the decode function of its format.
public interface IWeightDecoder {
    static void decode16(uint rowStart, uint element, out float4 w[4]);
}

public void decodeF32(StructuredBuffer<uint4> W4, uint rowStart, uint element, out float4 w[4]) {
    const uint index = rowStart / 16u + element / 4u;
    for (uint k = 0u; k < 4u; ++k) w[k] = asfloat(W4[index + k]);
}

public void decodeF16(StructuredBuffer<uint4> W4, uint rowStart, uint element, out float4 w[4]) {
    const uint index = rowStart / 16u + element / 8u;
    for (uint k = 0u; k < 2u; ++k) {
        const uint4 v = W4[index + k];
        w[2u * k] = float4(unpackHalf(v.x), unpackHalf(v.y));
        w[2u * k + 1u] = float4(unpackHalf(v.z), unpackHalf(v.w));
    }
}

public void decodeBF16(StructuredBuffer<uint4> W4, uint rowStart, uint element, out float4 w[4]) {
    const uint index = rowStart / 16u + element / 8u;
    for (uint k = 0u; k < 2u; ++k) {
        const uint4 v = W4[index + k];
        w[2u * k] = asfloat(uint4(v.x << 16, v.x & 0xffff0000u, v.y << 16, v.y & 0xffff0000u));
        w[2u * k + 1u] = asfloat(uint4(v.z << 16, v.z & 0xffff0000u, v.w << 16, v.w & 0xffff0000u));
    }
}

// 36-byte blocks of 32: half scale (padded to 4 bytes), 32 signed bytes.
public void decodeQ8_0(StructuredBuffer<uint> W, uint rowStart, uint element, out float4 w[4]) {
    const uint block = rowStart + (element / 32u) * 36u;
    const float d = unpackHalf(W[block >> 2]).x;
    const uint quants = (block + 4u + element % 32u) >> 2;
    // Flipping the sign bits maps -128..127 to 0..255.
    for (uint k = 0u; k < 4u; ++k) w[k] = d * (bytes4(W[quants + k] ^ 0x80808080u) - 128.0);
}

// Q4_K (144-byte blocks) and Q5_K (176-byte blocks, with a high bit per value).
void decodeQ4Q5<let Q5 : bool>(StructuredBuffer<uint4> W4, uint rowStart, uint element, out float4 w[4]) {
    const uint block = rowStart + (element / 256u) * (Q5 ? 176u : 144u);
    const uint quantOffset = Q5 ? 48u : 16u;
    // Each 64 values share 32 nibble bytes: low nibbles first, then high.
    const uint inBlock = element % 256u, group = inBlock / 64u, within = inBlock % 64u;
    const uint subBlock = 2u * group + within / 32u;
    const uint4 head = W4[block / 16u];
    const float2 scales = unpackHalf(head.x);
    float scale, minimum;
    packedScaleMin(head, subBlock, scale, minimum);
    const float factor = scales.x * scale, offset = scales.y * minimum;
    const uint4 q = W4[(block + quantOffset + 32u * group + within % 32u) / 16u];
    const uint shift = within < 32u ? 0u : 4u;
    uint4 high = uint4(0u);
    if (Q5) high = W4[(block + 16u + within % 32u) / 16u];
    for (uint k = 0u; k < 4u; ++k) {
        uint values = (q[k] >> shift) & 0x0f0f0f0fu;
        if (Q5) values |= ((high[k] >> subBlock) & 0x01010101u) << 4;
        w[k] = factor * bytes4(values) - offset;
    }
}

public void decodeQ4_K(StructuredBuffer<uint4> W4, uint rowStart, uint element, out float4 w[4]) {
    decodeQ4Q5<false>(W4, rowStart, element, w);
}

public void decodeQ5_K(StructuredBuffer<uint4> W4, uint rowStart, uint element, out float4 w[4]) {
    decodeQ4Q5<true>(W4, rowStart, element, w);
}

// 224-byte blocks. Halves of 128 values: quarters take low/high nibbles
// of two 32-byte runs of low bits and successive bit pairs of 32
// high-bit bytes; 16 signed scales at byte 192, the half scale at 208.
public void decodeQ6_K(StructuredBuffer<uint4> W4, uint rowStart, uint element, out float4 w[4]) {
    const uint block = rowStart + (element / 256u) * 224u;
    const uint inBlock = element % 256u, half_ = inBlock / 128u, within = inBlock % 128u;
    const uint quarter = within / 32u, lane = within % 32u;
    const uint4 low = W4[(block + 64u * half_ + lane + 32u * (quarter & 1u)) / 16u];
    const uint4 high = W4[(block + 128u + 32u * half_ + lane) / 16u];
    const uint4 scales = W4[(block + 192u) / 16u];
    const float d = unpackHalf(W4[(block + 208u) / 16u].x).x;
    const float factor = d * float(int(byteOf(scales, 8u * half_ + 2u * quarter + lane / 16u) << 24) >> 24);
    const uint lowShift = 4u * (quarter >> 1), highShift = 2u * quarter;
    for (uint k = 0u; k < 4u; ++k) {
        const uint values = ((low[k] >> lowShift) & 0x0f0f0f0fu) | (((high[k] >> highShift) & 0x03030303u) << 4);
        w[k] = factor * (bytes4(values) - 32.0);
    }
}

// 112-byte blocks. Halves of 128 values: four groups take successive bit
// pairs of 32 bytes; a cleared mask bit lowers the value by 4. Scales at
// byte 96, the half scale at 108.
public void decodeQ3_K(StructuredBuffer<uint4> W4, uint rowStart, uint element, out float4 w[4]) {
    const uint block = rowStart + (element / 256u) * 112u;
    const uint inBlock = element % 256u, half_ = inBlock / 128u, within = inBlock % 128u;
    const uint group = within / 32u, lane = within % 32u;
    const uint4 q = W4[(block + 32u + 32u * half_ + lane) / 16u];
    const uint4 mask = W4[(block + lane) / 16u];
    const uint4 tail = W4[(block + 96u) / 16u];
    const uint subBlock = 8u * half_ + 2u * group + lane / 16u;
    const uint scaleLow = subBlock < 8u ? byteOf(tail, subBlock) & 15u : byteOf(tail, subBlock - 8u) >> 4;
    const uint scaleHigh = (byteOf(tail, 8u + subBlock % 4u) >> (2u * (subBlock / 4u))) & 3u;
    const float factor = unpackHalf(tail.w).x * float(int(scaleLow | (scaleHigh << 4)) - 32);
    const uint shift = 2u * group, maskBit = 4u * half_ + group;
    for (uint k = 0u; k < 4u; ++k) {
        // value = low bits + 4 * mask bit - 4
        const uint values = ((q[k] >> shift) & 0x03030303u) | (((mask[k] >> maskBit) & 0x01010101u) << 2);
        w[k] = factor * (bytes4(values) - 4.0);
    }
}

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
