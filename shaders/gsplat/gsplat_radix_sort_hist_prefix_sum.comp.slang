// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Computes the exclusive prefix sum over the whole linear histogram array
// (bin-major layout: histogram[numWorkGroups * bin + workGroupID]). The scatter
// pass reads histogram[numWorkGroups * bin + workGroupID] as the base output
// offset for that (bin, count-workgroup) group, so an exclusive scan over the
// linear array yields exactly those bases.
//
// A single workgroup scans the entire array: each thread serially reduces a
// contiguous segment, the segment sums are scanned in shared memory, then each
// thread writes the exclusive prefix back over its own (disjoint) segment.
import gsplat_radix_sort;

static const uint WORKGROUP_SIZE = 256;

// The sums of the threads' segments, scanned in place.
groupshared uint segmentSums[WORKGROUP_SIZE];

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint3 localId: SV_GroupThreadID, uint3 groupId: SV_GroupID) {
    if (groupId.x != 0u) return;

    uint tid = localId.x;
    uint nthreads = WORKGROUP_SIZE;
    uint L = histogram.getCount();

    uint seg = (L + nthreads - 1u) / nthreads;
    uint start = tid * seg;
    uint end = min(start + seg, L);

    uint localSum = 0u;
    for (uint i = start; i < end; ++i) localSum += histogram[i];
    segmentSums[tid] = localSum;
    GroupMemoryBarrierWithGroupSync();

    // Inclusive Hillis-Steele scan of the per-thread segment sums.
    for (uint stride = 1u; stride < nthreads; stride <<= 1u) {
        uint add = 0u;
        if (tid >= stride) add = segmentSums[tid - stride];
        GroupMemoryBarrierWithGroupSync();
        segmentSums[tid] += add;
        GroupMemoryBarrierWithGroupSync();
    }

    // Exclusive base for this thread's segment = inclusive sum minus own contribution.
    uint running = segmentSums[tid] - localSum;
    for (uint i = start; i < end; ++i) {
        uint v = histogram[i];
        histogram[i] = running;
        running += v;
    }
}
