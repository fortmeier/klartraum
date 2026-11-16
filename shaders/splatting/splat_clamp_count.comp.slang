// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The number of instances to sort: the emitted count, at most the capacity.
struct PushConstants {
    uint capacity;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<uint> counters;    // [0]: instances emitted
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> sortCount; // one element

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
    sortCount[0] = min(counters[0], pc.capacity);
}
