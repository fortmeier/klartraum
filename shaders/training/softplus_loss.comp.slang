// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The non-saturating GAN loss weight x mean softplus(sign x) over `count` logits from `offset` (sign -1: real, +1:
// fake): their gradients weight sign sigmoid(sign x) / count and each logit's share of the loss. One invocation per
// logit.

struct PushConstants {
    uint offset;
    uint count;
    float sign;
    float weight;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> logits;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> gradient;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> shares;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id: SV_DispatchThreadID) {
    if (id.x >= pc.count) return;
    const uint i = pc.offset + id.x;
    const float z = pc.sign * logits[i];
    // softplus(z) = max(z, 0) + log(1 + exp(-|z|)), stable for large |z|.
    shares[i] = pc.weight * (max(z, 0.0) + log(1.0 + exp(-abs(z)))) / float(pc.count);
    gradient[i] = pc.weight * pc.sign / (1.0 + exp(-z)) / float(pc.count);
}
