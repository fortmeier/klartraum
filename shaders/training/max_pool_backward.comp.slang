// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The input gradient of max_pool.comp.slang: every input element gets its window's output gradient if it was the
// window's maximum, else zero. One invocation per input element.

struct PushConstants {
    uint planes; // batch x channels
    uint inputHeight;
    uint inputWidth;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> outputGradient;
[[vk::binding(1, 0)]] StructuredBuffer<float> argmax;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> inputGradient;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    if (id.x >= pc.planes * pc.inputHeight * pc.inputWidth) return;
    const uint x = id.x % pc.inputWidth, y = (id.x / pc.inputWidth) % pc.inputHeight;
    const uint plane = id.x / (pc.inputWidth * pc.inputHeight);
    const uint outputHeight = pc.inputHeight / 2u, outputWidth = pc.inputWidth / 2u;
    if (y / 2u >= outputHeight || x / 2u >= outputWidth) {
        inputGradient[id.x] = 0.0;
        return;
    }
    const uint o = (plane * outputHeight + y / 2u) * outputWidth + x / 2u;
    const uint which = (y % 2u) * 2u + (x % 2u);
    inputGradient[id.x] = uint(argmax[o] + 0.5) == which ? outputGradient[o] : 0.0;
}
