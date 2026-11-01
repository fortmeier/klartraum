// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Pass 3 of 3 of the cull + compaction: each visible splat is written to its
// workgroup's offset (pass 2) plus its rank among the visible splats of the
// workgroup, so the visible splats keep their splat-index order.
import gsplat_cull_and_compact;

// The visibility flags of the workgroup, scanned in place into ranks.
groupshared uint scan[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID, uint3 localId: SV_GroupThreadID, uint3 groupId: SV_GroupID) {
    const uint lid = localId.x;
    const uint id = globalId.x;
    uint key;
    const bool visible = visibleSplat(id, key);

    // Inclusive Hillis-Steele scan of the visibility flags gives each splat's rank.
    scan[lid] = visible ? 1u : 0u;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 1u; stride < 256u; stride <<= 1u) {
        const uint add = lid >= stride ? scan[lid - stride] : 0u;
        GroupMemoryBarrierWithGroupSync();
        scan[lid] += add;
        GroupMemoryBarrierWithGroupSync();
    }

    if (visible) {
        const uint slot = workgroupOffsets[groupId.x] + scan[lid] - 1u;
        indices[slot] = id;
        keys[slot] = key;
    }
}
