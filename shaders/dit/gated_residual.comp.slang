// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// X += gate * Y for every token's row of `width` values, the gate from the modulation vector (diffusion transformers'
// gated residuals).

struct PushConstants {
    uint width;
    uint count; // tokens x width
    uint gateOffset;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> Y;
[[vk::binding(1, 0)]] StructuredBuffer<float> modulation;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> X;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    const uint i = globalId.x + globalId.y * 65535u * 256u;
    if (i >= pc.count) return;
    X[i] += modulation[pc.gateOffset + i % pc.width] * Y[i];
}
