// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import gsplat_types;
import scalar_buffer;

// Sorted SoA — only binMask is needed for bounds computation
[[vk::binding(0, 0)]] ScalarBuffer<uint> binMask;
// One uint: the number of sorted elements.
[[vk::binding(1, 0)]] ScalarBuffer<uint> totalCount;
[[vk::binding(2, 0)]] RWScalarBuffer<StartAndEnd> startAndEnd;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint idx = globalId.x;
    if (idx >= totalCount[0]) return;

    if (idx == totalCount[0] - 1) {
        uint bm = binMask[idx];
        uint bin = uint(log2(bm));
        startAndEnd[bin].end = idx + 1;
        return;
    }

    uint bm0 = binMask[idx];
    uint bm1 = binMask[idx + 1];
    uint bin0 = uint(log2(bm0));
    uint bin1 = uint(log2(bm1));

    if (bm0 != bm1) {
        startAndEnd[bin0].end = idx + 1;
        startAndEnd[bin1].start = idx + 1;
    }
}
