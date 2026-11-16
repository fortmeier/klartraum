// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The sum of a buffer, e.g. of a loss's shares: one workgroup, each invocation adding a strided share, then a tree
// reduction in a fixed order.

struct PushConstants {
    uint count;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> data;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> result;

groupshared float partial[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 local: SV_GroupThreadID) {
    float sum = 0.0;
    for (uint i = local.x; i < pc.count; i += 256u) sum += data[i];
    partial[local.x] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (local.x < stride) partial[local.x] += partial[local.x + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (local.x == 0u) result[0] = partial[0];
}
