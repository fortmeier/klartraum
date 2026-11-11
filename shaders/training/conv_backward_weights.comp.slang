// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The weight and bias gradients of a 2D convolution: workgroup g < weights adds, over the batch and every output
// position, the output gradient times the input the weight met there; the following workgroups add the output
// gradients of one output channel (its bias). Each of the 256 invocations takes a strided share, and a tree reduction
// in shared memory adds the shares in a fixed order, so the result does not depend on scheduling.

struct PushConstants {
    uint batch, inputChannels, inputHeight, inputWidth;
    uint outputChannels, outputHeight, outputWidth;
    uint kernelHeight, kernelWidth, strideY, strideX, padTop, padLeft;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> input;          // [batch][inputChannels][inputHeight][inputWidth]
[[vk::binding(1, 0)]] StructuredBuffer<float> outputGradient; // [batch][outputChannels][outputHeight][outputWidth]
[[vk::binding(2, 0)]] RWStructuredBuffer<float> weightGradient; // [outputChannels][inputChannels][kh][kw]
[[vk::binding(3, 0)]] RWStructuredBuffer<float> biasGradient;   // [outputChannels]

groupshared float partial[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group: SV_GroupID, uint3 local: SV_GroupThreadID) {
    const uint weightCount = pc.outputChannels * pc.inputChannels * pc.kernelHeight * pc.kernelWidth;
    const uint g = group.x;
    const bool bias = g >= weightCount;
    const uint kx = g % pc.kernelWidth;
    const uint ky = (g / pc.kernelWidth) % pc.kernelHeight;
    const uint c = (g / (pc.kernelWidth * pc.kernelHeight)) % pc.inputChannels;
    const uint o = bias ? g - weightCount : g / (pc.kernelWidth * pc.kernelHeight * pc.inputChannels);
    const uint positions = pc.outputHeight * pc.outputWidth;
    const uint total = pc.batch * positions;
    float sum = 0.0;
    for (uint k = local.x; k < total; k += 256u) {
        const uint n = k / positions;
        const uint p = k % positions;
        const float gradient = outputGradient[(n * pc.outputChannels + o) * positions + p];
        if (bias) {
            sum += gradient;
            continue;
        }
        const int iy = int((p / pc.outputWidth) * pc.strideY + ky) - int(pc.padTop);
        const int ix = int((p % pc.outputWidth) * pc.strideX + kx) - int(pc.padLeft);
        if (iy < 0 || ix < 0 || iy >= int(pc.inputHeight) || ix >= int(pc.inputWidth)) continue;
        sum += gradient * input[((n * pc.inputChannels + c) * pc.inputHeight + uint(iy)) * pc.inputWidth + uint(ix)];
    }
    partial[local.x] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (local.x < stride) partial[local.x] += partial[local.x + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (local.x == 0u) {
        if (bias)
            biasGradient[o] = partial[0];
        else
            weightGradient[g] = partial[0];
    }
}
