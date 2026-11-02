// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import gsplat_radix_sort;

static const uint WORKGROUP_SIZE = 128;

// Per-bin global write cursor (seeded from histogram).
groupshared uint sharedOffsets[16];
// Per-bin base offset reserved for the current chunk.
groupshared uint localBinBase[16];
// Scratch for the per-bin rank scan.
groupshared uint scan[WORKGROUP_SIZE];

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint3 localId: SV_GroupThreadID, uint3 groupId: SV_GroupID) {
    /* Scatters elements into the output buffers based on the prefix-summed
     * histogram. Stability: elements with the same bin keep their relative
     * input order — within each 128-element chunk the rank inside a bin is a
     * parallel prefix scan over thread index (= input order), and the per-bin
     * cursor advances chunk by chunk.
     */

    uint lid = localId.x;
    uint numWorkGroups = WorkgroupCount().x;

    if (lid < 16) {
        uint offsetIdx = numWorkGroups * lid + groupId.x;
        sharedOffsets[lid] = histogram[offsetIdx];
    }
    GroupMemoryBarrierWithGroupSync();

    uint numElements = activeNumElements();
    // Rounded up, so the workgroups together cover every element.
    uint itemsPerWorkGroup0 = (numElements + numWorkGroups - 1) / numWorkGroups;
    uint iterationsPerWorkGroup = (itemsPerWorkGroup0 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    uint itemsPerWorkGroup = iterationsPerWorkGroup * WORKGROUP_SIZE;

    uint baseIdx = groupId.x * itemsPerWorkGroup;
    uint endIdx = baseIdx + itemsPerWorkGroup;

    for (uint groupStart = baseIdx; groupStart < endIdx; groupStart += WORKGROUP_SIZE) {
        uint elemIdx = groupStart + lid;
        bool isActive = (elemIdx < numElements);

        uint value = 0u, index = 0u, bin = 0xFFFFFFFFu;
        if (isActive) {
            if (pushConstants.pass % 2 == 0) {
                value = radixValuesA[elemIdx];
                index = indexesA[elemIdx];
            } else {
                value = radixValuesB[elemIdx];
                index = indexesB[elemIdx];
            }
            bin = (value >> (pushConstants.pass * 4)) & 0xFu;
        }

        // Rank within the element's bin, computed in parallel one bin at a time.
        uint myRank = 0u;
        for (uint b = 0u; b < 16u; b++) {
            uint pred = (bin == b) ? 1u : 0u;
            scan[lid] = pred;
            GroupMemoryBarrierWithGroupSync();

            // Inclusive Hillis-Steele scan over the 128 threads.
            for (uint stride = 1u; stride < WORKGROUP_SIZE; stride <<= 1u) {
                uint add = (lid >= stride) ? scan[lid - stride] : 0u;
                GroupMemoryBarrierWithGroupSync();
                scan[lid] += add;
                GroupMemoryBarrierWithGroupSync();
            }

            uint total = scan[WORKGROUP_SIZE - 1u];
            if (pred == 1u) myRank = scan[lid] - 1u;

            // First thread reserves this chunk's block of slots for bin b.
            if (lid == 0u && total > 0u) {
                InterlockedAdd(sharedOffsets[b], total, localBinBase[b]);
            }
            GroupMemoryBarrierWithGroupSync();
        }

        if (isActive) {
            uint pos = localBinBase[bin] + myRank;
            if (pushConstants.pass % 2 == 0) {
                radixValuesB[pos] = value;
                indexesB[pos] = index;
            } else {
                radixValuesA[pos] = value;
                indexesA[pos] = index;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
