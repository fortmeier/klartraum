// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The weight and bias gradients of a 2D convolution as a matrix product: dW[o][column] = sum_k dy[o][k] col[column][k]
// over k = (sample, output position), the columns the weights' (input channel, ky, kx) and one more of ones for the
// bias. Phase 0: every 16 x 16 workgroup multiplies a 64 x 64 tile of the result (4 x 4 per invocation) over one slice
// of k (split-K), staging 16 k at a time of both factors in shared memory, and writes its partial sums. Phase 1: the
// invocations add the slices' partial sums of the results in order (so they do not depend on scheduling).

struct PushConstants {
    uint batch, inputChannels, inputHeight, inputWidth;
    uint outputChannels, outputHeight, outputWidth;
    uint kernelHeight, kernelWidth, strideY, strideX, padTop, padLeft;
    uint splits;      // slices of k
    uint sliceLength; // k per slice, a multiple of 16
    uint phase;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> input;            // [batch][inputChannels][inputHeight][inputWidth]
[[vk::binding(1, 0)]] StructuredBuffer<float> outputGradient;   // [batch][outputChannels][outputHeight][outputWidth]
[[vk::binding(2, 0)]] RWStructuredBuffer<float> weightGradient; // [outputChannels][inputChannels][kh][kw]
[[vk::binding(3, 0)]] RWStructuredBuffer<float> biasGradient;   // [outputChannels]
[[vk::binding(4, 0)]] RWStructuredBuffer<float> partial;        // [splits][outputChannels][columns]

groupshared float tileA[16][64]; // [k][o]
groupshared float tileB[16][64]; // [k][column]

uint columns() { return pc.inputChannels * pc.kernelHeight * pc.kernelWidth + 1u; }

// The output gradient of output channel o at k; 0 beyond the matrix.
float gradientAt(uint o, uint k, uint end) {
    if (o >= pc.outputChannels || k >= end) return 0.0;
    const uint positions = pc.outputHeight * pc.outputWidth;
    const uint n = k / positions, p = k - n * positions;
    return outputGradient[(n * pc.outputChannels + o) * positions + p];
}

// The input the weight of `column` met at k (1 for the bias column); 0 in the padding and beyond the matrix.
float columnAt(uint column, uint k, uint end) {
    const uint weights = pc.inputChannels * pc.kernelHeight * pc.kernelWidth;
    if (column > weights || k >= end) return 0.0;
    if (column == weights) return 1.0;
    const uint taps = pc.kernelHeight * pc.kernelWidth;
    const uint c = column / taps, tap = column - c * taps;
    const uint ky = tap / pc.kernelWidth, kx = tap - ky * pc.kernelWidth;
    const uint positions = pc.outputHeight * pc.outputWidth;
    const uint n = k / positions, p = k - n * positions;
    const uint oy = p / pc.outputWidth, ox = p - oy * pc.outputWidth;
    const int iy = int(oy * pc.strideY + ky) - int(pc.padTop);
    const int ix = int(ox * pc.strideX + kx) - int(pc.padLeft);
    if (iy < 0 || ix < 0 || iy >= int(pc.inputHeight) || ix >= int(pc.inputWidth)) return 0.0;
    return input[((n * pc.inputChannels + c) * pc.inputHeight + uint(iy)) * pc.inputWidth + uint(ix)];
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 group: SV_GroupID, uint3 local: SV_GroupThreadID, uint3 global: SV_DispatchThreadID) {
    const uint total = pc.batch * pc.outputHeight * pc.outputWidth;
    const uint n = columns();
    if (pc.phase == 0u) {
        const uint o0 = group.y * 64u, column0 = group.x * 64u;
        const uint begin = group.z * pc.sliceLength, end = min(begin + pc.sliceLength, total);
        const uint thread = local.y * 16u + local.x;
        float sums[4][4];
        for (uint i = 0u; i < 4u; ++i)
            for (uint j = 0u; j < 4u; ++j) sums[i][j] = 0.0;
        for (uint k0 = begin; k0 < end; k0 += 16u) {
            for (uint e = 0u; e < 4u; ++e) {
                const uint index = thread + e * 256u;
                // 64 rows x 16 k each; consecutive invocations take consecutive k (consecutive positions).
                tileA[index % 16u][index / 16u] = gradientAt(o0 + index / 16u, k0 + index % 16u, end);
                tileB[index % 16u][index / 16u] = columnAt(column0 + index / 16u, k0 + index % 16u, end);
            }
            GroupMemoryBarrierWithGroupSync();
            for (uint k = 0u; k < 16u; ++k) {
                float a[4], b[4];
                for (uint i = 0u; i < 4u; ++i) a[i] = tileA[k][local.y * 4u + i];
                for (uint j = 0u; j < 4u; ++j) b[j] = tileB[k][local.x * 4u + j];
                for (uint i = 0u; i < 4u; ++i)
                    for (uint j = 0u; j < 4u; ++j) sums[i][j] += a[i] * b[j];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        for (uint i = 0u; i < 4u; ++i) {
            const uint o = o0 + local.y * 4u + i;
            if (o >= pc.outputChannels) continue;
            for (uint j = 0u; j < 4u; ++j) {
                const uint column = column0 + local.x * 4u + j;
                if (column < n) partial[(group.z * pc.outputChannels + o) * n + column] = sums[i][j];
            }
        }
        return;
    }
    // Phase 1: the results, strided over all invocations of the dispatch (fewer than results, as each invocation of
    // phase 0 makes 16).
    const uint width = 16u * ((n + 63u) / 64u);
    const uint height = 16u * ((pc.outputChannels + 63u) / 64u);
    const uint invocations = width * height * pc.splits;
    for (uint index = (global.z * height + global.y) * width + global.x; index < pc.outputChannels * n;
         index += invocations) {
        const uint o = index / n, column = index % n;
        float sum = 0.0;
        for (uint split = 0u; split < pc.splits; ++split) sum += partial[(split * pc.outputChannels + o) * n + column];
        if (column + 1u == n)
            biasGradient[o] = sum;
        else
            weightGradient[o * (n - 1u) + column] = sum;
    }
}
