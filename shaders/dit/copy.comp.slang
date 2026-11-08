// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Copies `count` values from the source at `sourceOffset` to the destination at `destinationOffset`, e.g. a stream's
// queries into its rows of a joint attention buffer.

struct PushConstants {
    uint count;
    uint sourceOffset;
    uint destinationOffset;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> source;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> destination;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    const uint i = globalId.x + globalId.y * 65535u * 256u;
    if (i >= pc.count) return;
    destination[pc.destinationOffset + i] = source[pc.sourceOffset + i];
}
