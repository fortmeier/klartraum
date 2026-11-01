// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Pass 2 of the three-pass deterministic binning pipeline.
// Converts the per-workgroup histogram produced by gsplat_binning_count.comp
// into exclusive prefix sums over the whole linear histogram array.
//
// Buffer layout: histogram[numCountWorkGroups * binIdx + countWorkGroupID]
// After this pass each entry holds the total number of (gaussian,bin) pairs
// that precede it in linear (bin-major) order. The scatter shader reads
// histogram[numCountWorkGroups * binIdx + countWorkGroupID] as the base write
// offset for that (bin, count-workgroup) group.
//
// A single workgroup scans the entire array: each thread serially reduces a
// contiguous segment, the segment sums are scanned in shared memory, then each
// thread writes the exclusive prefix back over its own (disjoint) segment.
import scalar_buffer;

static const uint WORKGROUP_SIZE = 256;

[[vk::binding(0, 0)]] RWScalarBuffer<uint> histogram;
[[vk::binding(1, 0)]] RWScalarBuffer<uint> offsets;

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
