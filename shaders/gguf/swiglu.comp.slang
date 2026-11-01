// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// y = silu(gate) * up for the params[0] active tokens of `width` values each.

struct PushConstants {
    uint width;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> G;
[[vk::binding(1, 0)]] StructuredBuffer<float> U;
[[vk::binding(2, 0)]] StructuredBuffer<uint> params;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> Y;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    const uint index = globalId.x + globalId.y * WorkgroupCount().x * 256u;
    if (index >= params[0] * pc.width) return;
    const float g = G[index];
    Y[index] = g / (1.0 + exp(-g)) * U[index];
}
