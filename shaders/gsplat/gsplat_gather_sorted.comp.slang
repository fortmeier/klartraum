// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import scalar_buffer;

// Binned (unsorted) SoA inputs
[[vk::binding(0, 0)]] ScalarBuffer<float2> inPos2D;
[[vk::binding(1, 0)]] ScalarBuffer<float> inZ;
[[vk::binding(2, 0)]] ScalarBuffer<uint> inBinMask;
[[vk::binding(3, 0)]] ScalarBuffer<float4> inCovInv;
[[vk::binding(4, 0)]] ScalarBuffer<float4> inColorAlpha;

[[vk::binding(5, 0)]] ScalarBuffer<uint> sortedIndexes;
// One uint: the number of sorted elements.
[[vk::binding(6, 0)]] ScalarBuffer<uint> totalCount;

// Sorted SoA outputs
[[vk::binding(7, 0)]] RWScalarBuffer<float2> outPos2D;
[[vk::binding(8, 0)]] RWScalarBuffer<float> outZ;
[[vk::binding(9, 0)]] RWScalarBuffer<uint> outBinMask;
[[vk::binding(10, 0)]] RWScalarBuffer<float4> outCovInv;
[[vk::binding(11, 0)]] RWScalarBuffer<float4> outColorAlpha;

[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint idx = globalId.x;
    if (idx >= totalCount[0]) return;

    uint src = sortedIndexes[idx];
    outPos2D[idx] = inPos2D[src];
    outZ[idx] = inZ[src];
    outBinMask[idx] = inBinMask[src];
    outCovInv[idx] = inCovInv[src];
    outColorAlpha[idx] = inColorAlpha[src];
}
