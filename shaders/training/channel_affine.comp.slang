// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// y = x * scale[c] + shift[c] per channel of an N x C x H x W tensor with up to four channels (e.g. ImageNet's
// normalization); backward (phase 1) the input gradient dy * scale[c], written or (accumulate) added. One invocation
// per element.

struct PushConstants {
    uint planes; // batch x channels
    uint channels;
    uint pixels;
    uint backward;
    uint accumulate;
    float scale[4];
    float shift[4];
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> x; // the input, or backward the output gradient
[[vk::binding(1, 0)]] RWStructuredBuffer<float> y; // the output, or backward the input gradient

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    if (id.x >= pc.planes * pc.pixels) return;
    const uint c = (id.x / pc.pixels) % pc.channels;
    const float v = pc.backward != 0u ? x[id.x] * pc.scale[c] : x[id.x] * pc.scale[c] + pc.shift[c];
    y[id.x] = pc.accumulate != 0u ? y[id.x] + v : v;
}
