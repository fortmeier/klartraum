// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The range [start, end) of each tile's instances in the sorted keys. Tiles without instances keep the zeros the
// buffer was cleared to.
import splat_common;

[[vk::binding(0, 0)]] StructuredBuffer<uint> keys;      // sorted
[[vk::binding(1, 0)]] StructuredBuffer<uint> sortCount; // one element
[[vk::binding(2, 0)]] RWStructuredBuffer<uint2> ranges; // per tile

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    const uint i = globalId.x;
    const uint count = sortCount[0];
    if (i >= count) return;
    const uint tile = keys[i] >> DEPTH_BITS;
    if (i == 0u || (keys[i - 1u] >> DEPTH_BITS) != tile) ranges[tile].x = i;
    if (i + 1u == count || (keys[i + 1u] >> DEPTH_BITS) != tile) ranges[tile].y = i + 1u;
}
