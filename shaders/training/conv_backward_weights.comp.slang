// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The weight and bias gradients of a 2D convolution as a matrix product: dW[o][column] = sum_k dy[o][k] col[column][k]
// over k = (sample, output position), the columns the weights' (input channel, ky, kx) and one more of ones for the
// bias. Phase 0: every 16 x 16 workgroup multiplies a 16 x 16 tile of the result over one slice of k (split-K),
// staging 16 k at a time of both factors in shared memory, and writes its partial sums. Phase 1: one invocation per
// result adds the slices' partial sums in order (so the result does not depend on scheduling).

struct PushConstants {
    uint batch, inputChannels, inputHeight, inputWidth;
    uint outputChannels, outputHeight, outputWidth;
    uint kernelHeight, kernelWidth, strideY, strideX, padTop, padLeft;
    uint splits;     // slices of k
    uint sliceLength; // k per slice, a multiple of 16
    uint phase;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> input;            // [batch][inputChannels][inputHeight][inputWidth]
[[vk::binding(1, 0)]] StructuredBuffer<float> outputGradient;   // [batch][outputChannels][outputHeight][outputWidth]
[[vk::binding(2, 0)]] RWStructuredBuffer<float> weightGradient; // [outputChannels][inputChannels][kh][kw]
[[vk::binding(3, 0)]] RWStructuredBuffer<float> biasGradient;   // [outputChannels]
[[vk::binding(4, 0)]] RWStructuredBuffer<float> partial;        // [splits][outputChannels][columns]

groupshared float tileA[16][17]; // [o][k]
groupshared float tileB[16][17]; // [column][k]

uint columns() { return pc.inputChannels * pc.kernelHeight * pc.kernelWidth + 1u; }

// The output gradient of output channel o at k; 0 beyond the matrix.
float gradientAt(uint o, uint k, uint total) {
    if (o >= pc.outputChannels || k >= total) return 0.0;
    const uint positions = pc.outputHeight * pc.outputWidth;
    const uint n = k / positions, p = k % positions;
    return outputGradient[(n * pc.outputChannels + o) * positions + p];
}

// The input the weight of `column` met at k (1 for the bias column); 0 in the padding and beyond the matrix.
float columnAt(uint column, uint k, uint total) {
    const uint weights = pc.inputChannels * pc.kernelHeight * pc.kernelWidth;
    if (column > weights || k >= total) return 0.0;
    if (column == weights) return 1.0;
    const uint kx = column % pc.kernelWidth;
    const uint ky = (column / pc.kernelWidth) % pc.kernelHeight;
    const uint c = column / (pc.kernelWidth * pc.kernelHeight);
    const uint positions = pc.outputHeight * pc.outputWidth;
    const uint n = k / positions, p = k % positions;
    const int iy = int((p / pc.outputWidth) * pc.strideY + ky) - int(pc.padTop);
    const int ix = int((p % pc.outputWidth) * pc.strideX + kx) - int(pc.padLeft);
    if (iy < 0 || ix < 0 || iy >= int(pc.inputHeight) || ix >= int(pc.inputWidth)) return 0.0;
    return input[((n * pc.inputChannels + c) * pc.inputHeight + uint(iy)) * pc.inputWidth + uint(ix)];
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 group: SV_GroupID, uint3 local: SV_GroupThreadID, uint3 groups: SV_DispatchThreadID) {
    const uint total = pc.batch * pc.outputHeight * pc.outputWidth;
    const uint n = columns();
    if (pc.phase == 0u) {
        const uint o = group.y * 16u + local.y, column = group.x * 16u + local.x;
        const uint begin = group.z * pc.sliceLength, end = min(begin + pc.sliceLength, total);
        float sum = 0.0;
        for (uint k0 = begin; k0 < end; k0 += 16u) {
            tileA[local.y][local.x] = gradientAt(group.y * 16u + local.y, k0 + local.x, end);
            tileB[local.y][local.x] = columnAt(group.x * 16u + local.y, k0 + local.x, end);
            GroupMemoryBarrierWithGroupSync();
            for (uint k = 0u; k < 16u; ++k) sum += tileA[local.y][k] * tileB[local.x][k];
            GroupMemoryBarrierWithGroupSync();
        }
        if (o < pc.outputChannels && column < n) partial[(group.z * pc.outputChannels + o) * n + column] = sum;
        return;
    }
    // Phase 1: a flat index over all invocations of the dispatch.
    const uint width = 16u * ((n + 15u) / 16u);
    const uint height = 16u * ((pc.outputChannels + 15u) / 16u);
    const uint index = (groups.z * height + groups.y) * width + groups.x;
    if (index >= pc.outputChannels * n) return;
    const uint o = index / n, column = index % n;
    float sum = 0.0;
    for (uint split = 0u; split < pc.splits; ++split) sum += partial[(split * pc.outputChannels + o) * n + column];
    if (column + 1u == n)
        biasGradient[o] = sum;
    else
        weightGradient[o * (n - 1u) + column] = sum;
}
