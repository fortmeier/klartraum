// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import gsplat_radix_sort;

static const uint WORKGROUP_SIZE = 128;

// The workgroup's count per bin.
groupshared uint sharedHistogram[16];

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint localIndex: SV_GroupIndex, uint3 localId: SV_GroupThreadID, uint3 groupId: SV_GroupID) {
    /* This shader computes local histograms of the number of elements in each bin
     * which are then written into a consective array in global memory.
     * The shader following on this histogram creation will then perform a
     * prefix sum to compute the offsets for each bin for the final sorting pass,
     * i.e., the placement of each element in the sorted array while preserving
     * the original order of elements in the same bin.
     */

    uint binIdx = localIndex;

    if (binIdx < 16) {
        sharedHistogram[binIdx] = 0;
    }

    GroupMemoryBarrierWithGroupSync();

    uint numWorkGroups = WorkgroupCount().x;
    uint numElements = activeNumElements();
    uint itemsPerWorkGroup0 = numElements / numWorkGroups;
    uint iterationsPerWorkGroup = (itemsPerWorkGroup0 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    uint itemsPerWorkGroup = iterationsPerWorkGroup * WORKGROUP_SIZE;

    uint idx = groupId.x * itemsPerWorkGroup + localId.x;
    uint endIdx = idx + itemsPerWorkGroup;
    while ((idx < endIdx) && (idx < numElements)) {
        uint value;
        uint index;
        if (pushConstants.pass % 2 == 0) {
            value = radixValuesA[idx];
            index = indexesA[idx];
        } else {
            value = radixValuesB[idx];
            index = indexesB[idx];
        }

        // get the bin of the current value for the current pass
        uint bin = (value >> (pushConstants.pass * 4)) & 0xF;

        // count the occurrences of this bin
        InterlockedAdd(sharedHistogram[bin], 1);

        idx += WORKGROUP_SIZE;
    }

    GroupMemoryBarrierWithGroupSync();

    // now we need to write the histogram to the global memory
    if (binIdx < 16) {
        uint globalBinIdx = numWorkGroups * binIdx + groupId.x;
        histogram[globalBinIdx] = sharedHistogram[binIdx];
    }

    if (localId.x == 0) {
        if (groupId.x == 0) {
            offsets[groupId.x] = 0;
        } else {
            offsets[groupId.x] = uint(-1);
        }
    }
}
