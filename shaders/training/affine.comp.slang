// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// y = x * scale + shift for every element; backward the input gradient dy * scale. Written or (accumulate) added.
// One invocation per element.

struct PushConstants {
    uint count;
    float scale;
    float shift;
    uint backward;
    uint accumulate;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> x;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> y;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    if (id.x >= pc.count) return;
    const float v = pc.backward != 0u ? x[id.x] * pc.scale : x[id.x] * pc.scale + pc.shift;
    y[id.x] = pc.accumulate != 0u ? y[id.x] + v : v;
}
