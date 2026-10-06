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
// The includer defines one of the WEIGHT_* macros, which selects the decoder.
//
// With `accumulate`, the products are added to y (a residual stream updated
// in place). With `lastTokenOnly`, only the last active token is computed and
// written to y row 0.

#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#define MAX_TOKENS 16
#define ROWS_PER_SUBGROUP 4
#define TOKEN_BLOCK 4
#define SUBGROUPS 2
#define WIDTH 16u

layout(local_size_x = 32 * SUBGROUPS, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint rows;
    uint columns;
    uint rowBytes;
    uint accumulate;
    uint lastTokenOnly;
    uint inputStride;   // floats between the inputs of consecutive tokens
    uint outputStride;  // floats between the outputs of consecutive tokens
} pc;

layout(set = 0, binding = 0) restrict readonly buffer WeightBuffer { uint W[]; };
layout(set = 0, binding = 0) restrict readonly buffer WeightVectors { uvec4 W4[]; };
layout(set = 0, binding = 1) restrict readonly buffer InputBuffer { vec4 X[]; };
layout(set = 0, binding = 2) restrict readonly buffer ParamsBuffer { uint params[]; };
layout(set = 0, binding = 3) buffer OutputBuffer { float Y[]; };

// Byte i (0..15) of four words.
uint byteOf(uvec4 words, uint i) { return (words[i >> 2] >> ((i & 3u) * 8u)) & 0xffu; }

// The (scale, min) pair `index` of the 12 packed 6-bit scales of Q4_K/Q5_K,
// which are bytes 4..15 of the block's first 16 bytes `head`.
void packedScaleMin(uvec4 head, uint index, out float scale, out float minimum) {
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
vec4 bytes4(uint word) { return unpackUnorm4x8(word) * 255.0; }

// Decodes the 16 weights of a row starting at element `element` (a multiple
// of 16), four per vector. The packed formats are decoded a word (four
// values) at a time.
void decode16(uint rowStart, uint element, out vec4 w[4]) {
#if defined(WEIGHT_F32)
    const uint index = rowStart / 16u + element / 4u;
    for (uint k = 0u; k < 4u; ++k) w[k] = uintBitsToFloat(W4[index + k]);
#elif defined(WEIGHT_F16) || defined(WEIGHT_BF16)
    const uint index = rowStart / 16u + element / 8u;
    for (uint k = 0u; k < 2u; ++k) {
        const uvec4 v = W4[index + k];
#if defined(WEIGHT_F16)
        w[2u * k] = vec4(unpackHalf2x16(v.x), unpackHalf2x16(v.y));
        w[2u * k + 1u] = vec4(unpackHalf2x16(v.z), unpackHalf2x16(v.w));
#else
        w[2u * k] = uintBitsToFloat(uvec4(v.x << 16, v.x & 0xffff0000u, v.y << 16, v.y & 0xffff0000u));
        w[2u * k + 1u] = uintBitsToFloat(uvec4(v.z << 16, v.z & 0xffff0000u, v.w << 16, v.w & 0xffff0000u));
#endif
    }
#elif defined(WEIGHT_Q8_0)
    // 36-byte blocks of 32: half scale (padded to 4 bytes), 32 signed bytes.
    const uint block = rowStart + (element / 32u) * 36u;
    const float d = unpackHalf2x16(W[block >> 2]).x;
    const uint quants = (block + 4u + element % 32u) >> 2;
    // Flipping the sign bits maps -128..127 to 0..255.
    for (uint k = 0u; k < 4u; ++k) w[k] = d * (bytes4(W[quants + k] ^ 0x80808080u) - 128.0);
#elif defined(WEIGHT_Q4_K) || defined(WEIGHT_Q5_K)
#if defined(WEIGHT_Q4_K)
    const uint block = rowStart + (element / 256u) * 144u;
    const uint quantOffset = 16u;
#else
    const uint block = rowStart + (element / 256u) * 176u;
    const uint quantOffset = 48u;
#endif
    // Each 64 values share 32 nibble bytes: low nibbles first, then high.
    const uint inBlock = element % 256u, group = inBlock / 64u, within = inBlock % 64u;
    const uint subBlock = 2u * group + within / 32u;
    const uvec4 head = W4[block / 16u];
    const vec2 scales = unpackHalf2x16(head.x);
    float scale, minimum;
    packedScaleMin(head, subBlock, scale, minimum);
    const float factor = scales.x * scale, offset = scales.y * minimum;
    const uvec4 q = W4[(block + quantOffset + 32u * group + within % 32u) / 16u];
    const uint shift = within < 32u ? 0u : 4u;
#if defined(WEIGHT_Q5_K)
    const uvec4 high = W4[(block + 16u + within % 32u) / 16u];
#endif
    for (uint k = 0u; k < 4u; ++k) {
        uint values = (q[k] >> shift) & 0x0f0f0f0fu;
#if defined(WEIGHT_Q5_K)
        values |= ((high[k] >> subBlock) & 0x01010101u) << 4;
#endif
        w[k] = factor * bytes4(values) - offset;
    }
#elif defined(WEIGHT_Q6_K)
    // 224-byte blocks. Halves of 128 values: quarters take low/high nibbles
    // of two 32-byte runs of low bits and successive bit pairs of 32
    // high-bit bytes; 16 signed scales at byte 192, the half scale at 208.
    const uint block = rowStart + (element / 256u) * 224u;
    const uint inBlock = element % 256u, half_ = inBlock / 128u, within = inBlock % 128u;
    const uint quarter = within / 32u, lane = within % 32u;
    const uvec4 low = W4[(block + 64u * half_ + lane + 32u * (quarter & 1u)) / 16u];
    const uvec4 high = W4[(block + 128u + 32u * half_ + lane) / 16u];
    const uvec4 scales = W4[(block + 192u) / 16u];
    const float d = unpackHalf2x16(W4[(block + 208u) / 16u].x).x;
    const float factor = d * float(int(byteOf(scales, 8u * half_ + 2u * quarter + lane / 16u) << 24) >> 24);
    const uint lowShift = 4u * (quarter >> 1), highShift = 2u * quarter;
    for (uint k = 0u; k < 4u; ++k) {
        const uint values = ((low[k] >> lowShift) & 0x0f0f0f0fu) | (((high[k] >> highShift) & 0x03030303u) << 4);
        w[k] = factor * (bytes4(values) - 32.0);
    }
#elif defined(WEIGHT_Q3_K)
    // 112-byte blocks. Halves of 128 values: four groups take successive bit
    // pairs of 32 bytes; a cleared mask bit lowers the value by 4. Scales at
    // byte 96, the half scale at 108.
    const uint block = rowStart + (element / 256u) * 112u;
    const uint inBlock = element % 256u, half_ = inBlock / 128u, within = inBlock % 128u;
    const uint group = within / 32u, lane = within % 32u;
    const uvec4 q = W4[(block + 32u + 32u * half_ + lane) / 16u];
    const uvec4 mask = W4[(block + lane) / 16u];
    const uvec4 tail = W4[(block + 96u) / 16u];
    const uint subBlock = 8u * half_ + 2u * group + lane / 16u;
    const uint scaleLow = subBlock < 8u ? byteOf(tail, subBlock) & 15u : byteOf(tail, subBlock - 8u) >> 4;
    const uint scaleHigh = (byteOf(tail, 8u + subBlock % 4u) >> (2u * (subBlock / 4u))) & 3u;
    const float factor = unpackHalf2x16(tail.w).x * float(int(scaleLow | (scaleHigh << 4)) - 32);
    const uint shift = 2u * group, maskBit = 4u * half_ + group;
    for (uint k = 0u; k < 4u; ++k) {
        // value = low bits + 4 * mask bit - 4
        const uint values = ((q[k] >> shift) & 0x03030303u) | (((mask[k] >> maskBit) & 0x01010101u) << 2);
        w[k] = factor * (bytes4(values) - 4.0);
    }
#else
#error "define one of the WEIGHT_* macros"
#endif
}

void main() {
    const uint group = gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x;
    const uint firstRow = (group * SUBGROUPS + gl_SubgroupID) * ROWS_PER_SUBGROUP;
    if (firstRow >= pc.rows) return;
    const uint lane = gl_SubgroupInvocationID;
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
            vec4 w[ROWS_PER_SUBGROUP][4];
            for (uint r = 0u; r < ROWS_PER_SUBGROUP; ++r) {
                const uint row = min(firstRow + r, pc.rows - 1u);
                decode16(row * pc.rowBytes, element, w[r]);
            }
            for (uint t = 0u; t < TOKEN_BLOCK; ++t) {
                if (t < blockTokens) {
                    const uint base = ((firstToken + block + t) * pc.inputStride + element) / 4u;
                    for (uint k = 0u; k < 4u; ++k) {
                        const vec4 x = X[base + k];
                        for (uint r = 0u; r < ROWS_PER_SUBGROUP; ++r) sums[r][t] += dot(w[r][k], x);
                    }
                }
            }
        }
        for (uint t = 0u; t < TOKEN_BLOCK; ++t) {
            if (t >= blockTokens) break;
            const uint outputToken = pc.lastTokenOnly != 0u ? 0u : block + t;
            for (uint r = 0u; r < ROWS_PER_SUBGROUP; ++r) {
                const float total = subgroupAdd(sums[r][t]);
                const uint row = firstRow + r;
                if (lane == 0u && row < pc.rows) {
                    const uint index = outputToken * pc.outputStride + row;
                    Y[index] = pc.accumulate != 0u ? Y[index] + total : total;
                }
            }
        }
    }
}
