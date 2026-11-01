// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Pass 2 of 3 of the cull + compaction: the first workgroup turns the per-workgroup
// visible counts of pass 1 into exclusive offsets, in place, and writes the total into
// the indirect draw arguments and the radix sort's count buffer (binding 6), so the
// sort processes only the visible range. The other workgroups of the
// shared dispatch size return right away.
import gsplat_cull_and_compact;

// One uint: the number of visible splats, read by the radix sort.
[[vk::binding(6, 0)]] RWScalarBuffer<uint> numberTotalGaussians;

// The current chunk of 256 counts, scanned in place.
groupshared uint scan[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 localId: SV_GroupThreadID, uint3 groupId: SV_GroupID) {
    if (groupId.x != 0u) {
        return;
    }
    const uint lid = localId.x;
    const uint numberWorkgroups = WorkgroupCount().x;
    uint carry = 0u;
    for (uint base = 0u; base < numberWorkgroups; base += 256u) {
        const uint index = base + lid;
        const uint count = index < numberWorkgroups ? workgroupOffsets[index] : 0u;
        scan[lid] = count;
        GroupMemoryBarrierWithGroupSync();
        // Inclusive Hillis-Steele scan of this chunk of 256 counts.
        for (uint stride = 1u; stride < 256u; stride <<= 1u) {
            const uint add = lid >= stride ? scan[lid - stride] : 0u;
            GroupMemoryBarrierWithGroupSync();
            scan[lid] += add;
            GroupMemoryBarrierWithGroupSync();
        }
        if (index < numberWorkgroups) {
            workgroupOffsets[index] = carry + scan[lid] - count;
        }
        carry += scan[255];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lid == 0u) {
        drawArgs[0].instanceCount = carry;
        numberTotalGaussians[0] = carry;
    }
}
