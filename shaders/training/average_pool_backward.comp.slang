// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The input gradient of average_pool.comp.slang: a quarter of the window's output gradient, written or (accumulate)
// added. One invocation per input element.

struct PushConstants {
    uint planes;
    uint inputHeight;
    uint inputWidth;
    uint accumulate;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> outputGradient;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> inputGradient;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    if (id.x >= pc.planes * pc.inputHeight * pc.inputWidth) return;
    const uint x = id.x % pc.inputWidth, y = (id.x / pc.inputWidth) % pc.inputHeight;
    const uint plane = id.x / (pc.inputWidth * pc.inputHeight);
    const uint oh = pc.inputHeight / 2u, ow = pc.inputWidth / 2u;
    float g = 0.0;
    if (y / 2u < oh && x / 2u < ow) g = 0.25 * outputGradient[(plane * oh + y / 2u) * ow + x / 2u];
    inputGradient[id.x] = pc.accumulate != 0u ? inputGradient[id.x] + g : g;
}
