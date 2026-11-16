// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The input gradient of a 2D convolution (NCHW input, OIHW weights) as a matrix product: dx[c][position] =
// sum_r W[r][c] dy[r][position] over r = (output channel, ky, kx), where dy[r][position] is the gradient of the output
// that tap (ky, kx) of the input position feeds (0 where none does, e.g. between the strides). Every 16 x 16 workgroup
// computes a 64 x 64 tile of input channels x input positions, each invocation 4 x 4 of them, staging 16 r at a time
// of both factors in shared memory.

struct PushConstants {
    uint batch, inputChannels, inputHeight, inputWidth;
    uint outputChannels, outputHeight, outputWidth;
    uint kernelHeight, kernelWidth, strideY, strideX, padTop, padLeft;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> outputGradient; // [batch][outputChannels][outputHeight][outputWidth]
[[vk::binding(1, 0)]] StructuredBuffer<float> weights;        // [outputChannels][inputChannels][kh][kw]
[[vk::binding(2, 0)]] RWStructuredBuffer<float> inputGradient; // [batch][inputChannels][inputHeight][inputWidth]

groupshared float tileW[16][64];  // [r][c]
groupshared float tileDy[16][64]; // [r][position]

// The weight of tap r for input channel c; 0 beyond the matrix.
float weightAt(uint c, uint r, uint reduction) {
    if (c >= pc.inputChannels || r >= reduction) return 0.0;
    const uint taps = pc.kernelHeight * pc.kernelWidth;
    const uint o = r / taps, tap = r - o * taps;
    return weights[(o * pc.inputChannels + c) * taps + tap];
}

// The gradient of the output that tap r of input position `position` feeds; 0 where there is none.
float gradientAt(uint r, uint position, uint reduction, uint positions) {
    if (r >= reduction || position >= positions) return 0.0;
    const uint taps = pc.kernelHeight * pc.kernelWidth;
    const uint pixels = pc.inputHeight * pc.inputWidth;
    const uint o = r / taps, tap = r - o * taps;
    const uint ky = tap / pc.kernelWidth, kx = tap - ky * pc.kernelWidth;
    const uint n = position / pixels, p = position - n * pixels;
    const uint y = p / pc.inputWidth, x = p - y * pc.inputWidth;
    // Output row oy reads input row oy * stride - pad + ky.
    const int ty = int(y + pc.padTop) - int(ky), tx = int(x + pc.padLeft) - int(kx);
    if (ty < 0 || tx < 0) return 0.0;
    const uint oy = uint(ty) / pc.strideY, ox = uint(tx) / pc.strideX;
    if (oy * pc.strideY != uint(ty) || ox * pc.strideX != uint(tx) || oy >= pc.outputHeight || ox >= pc.outputWidth)
        return 0.0;
    return outputGradient[((n * pc.outputChannels + o) * pc.outputHeight + oy) * pc.outputWidth + ox];
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 group: SV_GroupID, uint3 local: SV_GroupThreadID) {
    const uint reduction = pc.outputChannels * pc.kernelHeight * pc.kernelWidth;
    const uint pixels = pc.inputHeight * pc.inputWidth;
    const uint positions = pc.batch * pixels;
    const uint c0 = group.y * 64u, p0 = group.x * 64u;
    const uint thread = local.y * 16u + local.x;
    float sums[4][4];
    for (uint i = 0u; i < 4u; ++i)
        for (uint j = 0u; j < 4u; ++j) sums[i][j] = 0.0;
    for (uint r0 = 0u; r0 < reduction; r0 += 16u) {
        for (uint e = 0u; e < 4u; ++e) {
            const uint index = thread + e * 256u;
            // Weights: 64 channels x 16 r; gradients: 16 r x 64 positions (consecutive invocations, consecutive
            // positions).
            tileW[index % 16u][index / 16u] = weightAt(c0 + index / 16u, r0 + index % 16u, reduction);
            tileDy[index / 64u][index % 64u] = gradientAt(r0 + index / 64u, p0 + index % 64u, reduction, positions);
        }
        GroupMemoryBarrierWithGroupSync();
        for (uint r = 0u; r < 16u; ++r) {
            float a[4], b[4];
            for (uint i = 0u; i < 4u; ++i) a[i] = tileW[r][local.y * 4u + i];
            for (uint j = 0u; j < 4u; ++j) b[j] = tileDy[r][local.x * 4u + j];
            for (uint i = 0u; i < 4u; ++i)
                for (uint j = 0u; j < 4u; ++j) sums[i][j] += a[i] * b[j];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    for (uint i = 0u; i < 4u; ++i) {
        const uint c = c0 + local.y * 4u + i;
        if (c >= pc.inputChannels) continue;
        for (uint j = 0u; j < 4u; ++j) {
            const uint position = p0 + local.x * 4u + j;
            if (position >= positions) continue;
            const uint n = position / pixels, p = position - n * pixels;
            inputGradient[(n * pc.inputChannels + c) * pixels + p] = sums[i][j];
        }
    }
}
