// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Adds the squared sum of a gradient tensor to a running total (one atomic float addition per workgroup), for the
// global norm of gradient clipping; the total must start at zero.

struct PushConstants {
    uint count;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> data;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> total;

groupshared float partial[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id: SV_DispatchThreadID, uint3 local: SV_GroupThreadID) {
    const float v = id.x < pc.count ? data[id.x] : 0.0;
    partial[local.x] = v * v;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (local.x < stride) partial[local.x] += partial[local.x + stride];
        GroupMemoryBarrierWithGroupSync();
    }
    if (local.x == 0u) {
        float previous;
        InterlockedAdd(total[0], partial[0], previous);
    }
}
