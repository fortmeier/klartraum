// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Pass 1 of 3 of the cull + compaction: counts the visible splats of each workgroup.
// Visible splats are compacted in splat-index order (pass 2 scans the counts, pass 3
// scatters), so the stable radix sort breaks depth-key ties by splat index and the
// draw order, and with it the blended image, is the same on every run.
import gsplat_cull_and_compact;

// The number of visible splats in this workgroup.
groupshared uint visibleCount;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID, uint3 localId: SV_GroupThreadID, uint3 groupId: SV_GroupID) {
    if (localId.x == 0u) {
        visibleCount = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    uint key;
    if (visibleSplat(globalId.x, key)) {
        InterlockedAdd(visibleCount, 1u);
    }
    GroupMemoryBarrierWithGroupSync();

    if (localId.x == 0u) {
        workgroupOffsets[groupId.x] = visibleCount;
    }
}
