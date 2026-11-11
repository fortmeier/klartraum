// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// weight x the mean absolute difference of two tensors (e.g. VGG features of an image and of its target): every
// element's share of it, and the gradient for the first tensor, written or (accumulate) added. One invocation per
// element.

struct PushConstants {
    uint count;
    float weight;
    uint accumulate;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> a;
[[vk::binding(1, 0)]] StructuredBuffer<float> b;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> gradient; // of a
[[vk::binding(3, 0)]] RWStructuredBuffer<float> shares;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    if (id.x >= pc.count) return;
    const float d = a[id.x] - b[id.x];
    const float g = pc.weight * sign(d) / float(pc.count);
    gradient[id.x] = pc.accumulate != 0u ? gradient[id.x] + g : g;
    shares[id.x] = pc.weight * abs(d) / float(pc.count);
}
