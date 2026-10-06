// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Matrix-vector products y[t] = W x[t] with GGUF-encoded weights, for up to
// MAX_TOKENS tokens per dispatch (params[0] tokens are active). W holds
// `rows` rows of `columns` values each, row r starting at byte r * rowBytes.
//
// Each 32-wide subgroup computes ROWS_PER_SUBGROUP rows; every lane decodes 8
// consecutive weights per step, so a subgroup covers 256 values per step and
// reuses each decoded weight for all active tokens. The includer defines one
// of the WEIGHT_* macros, which selects the decoder below.
//
// With `accumulate`, the products are added to y (a residual stream updated
// in place). With `lastTokenOnly`, only the last active token is computed and
// written to y row 0.

#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require

#define MAX_TOKENS 16
#define ROWS_PER_SUBGROUP 2
#define SUBGROUPS 2

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
layout(set = 0, binding = 1) restrict readonly buffer InputBuffer { vec4 X[]; };
layout(set = 0, binding = 2) restrict readonly buffer ParamsBuffer { uint params[]; };
layout(set = 0, binding = 3) buffer OutputBuffer { float Y[]; };

uint byteAt(uint address) { return (W[address >> 2] >> ((address & 3u) * 8u)) & 0xffu; }

// Two consecutive (even-addressed) bytes as a half float.
float halfAt(uint address) {
    const uint bits = (W[address >> 2] >> ((address & 2u) * 8u)) & 0xffffu;
    return unpackHalf2x16(bits).x;
}

// Eight bytes at any byte address, little-endian in two words. The buffer is
// padded so the third word read for unaligned addresses exists.
uvec2 load8(uint address) {
    const uint index = address >> 2, shift = (address & 3u) * 8u;
    const uint a = W[index], b = W[index + 1];
    if (shift == 0u) return uvec2(a, b);
    const uint c = W[index + 2];
    return uvec2((a >> shift) | (b << (32u - shift)), (b >> shift) | (c << (32u - shift)));
}

// Byte i (0..7) of an 8-byte group.
uint byteOf(uvec2 bytes, uint i) { return (bytes[i >> 2] >> ((i & 3u) * 8u)) & 0xffu; }

// The (scale, min) pair `index` of the 12-byte packed 6-bit scales of Q4_K/Q5_K.
void packedScaleMin(uint address, uint index, out float scale, out float minimum) {
    if (index < 4u) {
        scale = float(byteAt(address + index) & 63u);
        minimum = float(byteAt(address + index + 4u) & 63u);
    } else {
        const uint packed = byteAt(address + index + 4u);
        scale = float((packed & 15u) | ((byteAt(address + index - 4u) >> 6) << 4));
        minimum = float((packed >> 4) | ((byteAt(address + index) >> 6) << 4));
    }
}

// Decodes the 8 weights of a row starting at element `element` (a multiple of 8).
void decode8(uint rowStart, uint element, out float w[8]) {
#if defined(WEIGHT_F32)
    const uint index = (rowStart >> 2) + element;
    for (uint i = 0u; i < 8u; ++i) w[i] = uintBitsToFloat(W[index + i]);
#elif defined(WEIGHT_F16)
    const uint index = (rowStart >> 2) + element / 2u;
    for (uint i = 0u; i < 4u; ++i) {
        const vec2 pair = unpackHalf2x16(W[index + i]);
        w[2u * i] = pair.x;
        w[2u * i + 1u] = pair.y;
    }
#elif defined(WEIGHT_BF16)
    const uint index = (rowStart >> 2) + element / 2u;
    for (uint i = 0u; i < 4u; ++i) {
        const uint pair = W[index + i];
        w[2u * i] = uintBitsToFloat(pair << 16);
        w[2u * i + 1u] = uintBitsToFloat(pair & 0xffff0000u);
    }
#elif defined(WEIGHT_Q8_0)
    // 34-byte blocks of 32: half scale, 32 signed bytes.
    const uint block = rowStart + (element / 32u) * 34u;
    const float d = halfAt(block);
    const uvec2 q = load8(block + 2u + element % 32u);
    for (uint i = 0u; i < 8u; ++i) w[i] = d * float(int(byteOf(q, i) << 24) >> 24);
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
    const vec2 scales = unpackHalf2x16(W[block >> 2]);
    float scale, minimum;
    packedScaleMin(block + 4u, subBlock, scale, minimum);
    const float factor = scales.x * scale, offset = scales.y * minimum;
    const uvec2 q = load8(block + quantOffset + 32u * group + within % 32u);
    const uint shift = within < 32u ? 0u : 4u;
#if defined(WEIGHT_Q5_K)
    const uvec2 high = load8(block + 16u + within % 32u);
#endif
    for (uint i = 0u; i < 8u; ++i) {
        uint value = (byteOf(q, i) >> shift) & 15u;
#if defined(WEIGHT_Q5_K)
        value |= ((byteOf(high, i) >> subBlock) & 1u) << 4;
#endif
        w[i] = factor * float(value) - offset;
    }
#elif defined(WEIGHT_Q6_K)
    // Halves of 128 values: quarters take low/high nibbles of two 32-byte
    // runs of low bits and successive bit pairs of 32 high-bit bytes.
    const uint block = rowStart + (element / 256u) * 210u;
    const uint inBlock = element % 256u, half_ = inBlock / 128u, within = inBlock % 128u;
    const uint quarter = within / 32u, lane = within % 32u;
    const uvec2 low = load8(block + 64u * half_ + lane + 32u * (quarter & 1u));
    const uvec2 high = load8(block + 128u + 32u * half_ + lane);
    const float scale = float(int(byteAt(block + 192u + 8u * half_ + 2u * quarter + lane / 16u) << 24) >> 24);
    const float factor = halfAt(block + 208u) * scale;
    const uint lowShift = 4u * (quarter >> 1), highShift = 2u * quarter;
    for (uint i = 0u; i < 8u; ++i) {
        const uint value = ((byteOf(low, i) >> lowShift) & 15u) | (((byteOf(high, i) >> highShift) & 3u) << 4);
        w[i] = factor * float(int(value) - 32);
    }
#elif defined(WEIGHT_Q3_K)
    // Halves of 128 values: four groups take successive bit pairs of 32
    // bytes; a cleared mask bit lowers the value by 4.
    const uint block = rowStart + (element / 256u) * 110u;
    const uint inBlock = element % 256u, half_ = inBlock / 128u, within = inBlock % 128u;
    const uint group = within / 32u, lane = within % 32u;
    const uvec2 q = load8(block + 32u + 32u * half_ + lane);
    const uvec2 mask = load8(block + lane);
    const uint subBlock = 8u * half_ + 2u * group + lane / 16u;
    const uint scaleLow = subBlock < 8u ? byteAt(block + 96u + subBlock) & 15u : byteAt(block + 88u + subBlock) >> 4;
    const uint scaleHigh = (byteAt(block + 104u + subBlock % 4u) >> (2u * (subBlock / 4u))) & 3u;
    const float factor = halfAt(block + 108u) * float(int(scaleLow | (scaleHigh << 4)) - 32);
    const uint shift = 2u * group, maskBit = 4u * half_ + group;
    for (uint i = 0u; i < 8u; ++i) {
        const int value = int((byteOf(q, i) >> shift) & 3u) - (((byteOf(mask, i) >> maskBit) & 1u) != 0u ? 0 : 4);
        w[i] = factor * float(value);
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

    float sums[ROWS_PER_SUBGROUP][MAX_TOKENS];
    for (uint r = 0u; r < ROWS_PER_SUBGROUP; ++r) {
        for (uint t = 0u; t < MAX_TOKENS; ++t) sums[r][t] = 0.0;
    }

    for (uint element = 8u * lane; element < pc.columns; element += 256u) {
        float w[ROWS_PER_SUBGROUP][8];
        for (uint r = 0u; r < ROWS_PER_SUBGROUP; ++r) {
            const uint row = min(firstRow + r, pc.rows - 1u);
            decode8(row * pc.rowBytes, element, w[r]);
        }
        for (uint t = 0u; t < MAX_TOKENS; ++t) {
            if (t >= activeTokens) break;
            const uint base = ((firstToken + t) * pc.inputStride + element) / 4u;
            const vec4 x0 = X[base], x1 = X[base + 1u];
            for (uint r = 0u; r < ROWS_PER_SUBGROUP; ++r) {
                sums[r][t] += dot(vec4(w[r][0], w[r][1], w[r][2], w[r][3]), x0) +
                              dot(vec4(w[r][4], w[r][5], w[r][6], w[r][7]), x1);
            }
        }
    }

    for (uint t = 0u; t < MAX_TOKENS; ++t) {
        if (t >= activeTokens) break;
        const uint outputToken = pc.lastTokenOnly != 0u ? 0u : t;
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
