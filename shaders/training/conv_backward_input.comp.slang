// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The input gradient of a 2D convolution (NCHW input, OIHW weights): input element (n, c, y, x) adds, for every
// output channel and kernel tap whose output position it feeds, the weight times that output's gradient. One
// invocation per input element; a gather, so no atomics and a fixed order.

struct PushConstants {
    uint batch, inputChannels, inputHeight, inputWidth;
    uint outputChannels, outputHeight, outputWidth;
    uint kernelHeight, kernelWidth, strideY, strideX, padTop, padLeft;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> outputGradient; // [batch][outputChannels][outputHeight][outputWidth]
[[vk::binding(1, 0)]] StructuredBuffer<float> weights;        // [outputChannels][inputChannels][kh][kw]
[[vk::binding(2, 0)]] RWStructuredBuffer<float> inputGradient; // [batch][inputChannels][inputHeight][inputWidth]

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    const uint count = pc.batch * pc.inputChannels * pc.inputHeight * pc.inputWidth;
    const uint i = id.x;
    if (i >= count) return;
    const uint x = i % pc.inputWidth;
    const uint y = (i / pc.inputWidth) % pc.inputHeight;
    const uint c = (i / (pc.inputWidth * pc.inputHeight)) % pc.inputChannels;
    const uint n = i / (pc.inputWidth * pc.inputHeight * pc.inputChannels);
    float sum = 0.0;
    for (uint ky = 0u; ky < pc.kernelHeight; ++ky) {
        // Output row oy reads input row oy * stride - pad + ky.
        const int ty = int(y) + int(pc.padTop) - int(ky);
        if (ty < 0 || ty % int(pc.strideY) != 0) continue;
        const uint oy = uint(ty) / pc.strideY;
        if (oy >= pc.outputHeight) continue;
        for (uint kx = 0u; kx < pc.kernelWidth; ++kx) {
            const int tx = int(x) + int(pc.padLeft) - int(kx);
            if (tx < 0 || tx % int(pc.strideX) != 0) continue;
            const uint ox = uint(tx) / pc.strideX;
            if (ox >= pc.outputWidth) continue;
            for (uint o = 0u; o < pc.outputChannels; ++o) {
                const float w = weights[((o * pc.inputChannels + c) * pc.kernelHeight + ky) * pc.kernelWidth + kx];
                sum += w * outputGradient[((n * pc.outputChannels + o) * pc.outputHeight + oy) * pc.outputWidth + ox];
            }
        }
    }
    inputGradient[i] = sum;
}
