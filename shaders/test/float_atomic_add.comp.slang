// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Test kernel: invocation i adds (i % 5) + 0.25 to sums[i % 7] with an atomic float addition.

struct PushConstants {
    uint count;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] RWStructuredBuffer<float> sums;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    const uint i = globalId.x;
    if (i < pc.count) InterlockedAdd(sums[i % 7], float(i % 5) + 0.25);
}
