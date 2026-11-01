// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import scalar_buffer;

static const uint WORKGROUP_SIZE = 128;

// SoA projected Gaussian2D inputs
[[vk::binding(0, 0)]] ScalarBuffer<float2> inPos2D;
[[vk::binding(1, 0)]] ScalarBuffer<float> inZ;
[[vk::binding(2, 0)]] ScalarBuffer<float4> inCovInv;     // (m00,m10,m01,m11)
[[vk::binding(3, 0)]] ScalarBuffer<float4> inColorAlpha; // (r,g,b,alpha)

// Prefix sums from the histogram pass (reused AoS prefix sum shader)
[[vk::binding(4, 0)]] ScalarBuffer<uint> prefixSums;

// One uint: the total binned element count (accumulated by this shader)
[[vk::binding(5, 0)]] RWScalarBuffer<uint> totalCount;

// SoA binned Gaussian2D outputs
[[vk::binding(6, 0)]] RWScalarBuffer<float2> outPos2D;
[[vk::binding(7, 0)]] RWScalarBuffer<float> outZ;
[[vk::binding(8, 0)]] RWScalarBuffer<uint> outBinMask;
[[vk::binding(9, 0)]] RWScalarBuffer<float4> outCovInv;
[[vk::binding(10, 0)]] RWScalarBuffer<float4> outColorAlpha;

struct PushConstants {
    uint numElements;
    uint gridSize;
    float screenWidth;
    float screenHeight;
    uint maxOutput;
    float spreadMultiplier; // tunable: Gaussian footprint radius multiplier (default 2.5)
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

// The overlap flags of the current bin, scanned in place into ranks.
groupshared uint sharedScan[WORKGROUP_SIZE];
// The workgroup's number of Gaussians per bin.
groupshared uint sharedBinCounts[16];

// Compute per-thread overlap bitmask for ALL bins in one pass.
// Avoids inverting the covariance 16× per thread — instead computes the spread once
// directly from covInv (ci stored column-major: m00=ci.x, m10=ci.y, m01=ci.z, m11=ci.w).
// For symmetric covInv:  det = m00*m11 - m01*m10
//   cov[0][0] = m11/det,  cov[1][1] = m00/det
uint computeOverlapMask(float2 position, float4 ci) {
    float det = ci.x * ci.w - ci.y * ci.z;
    float2 spread = sqrt(float2(ci.w, ci.x) / det) * pc.spreadMultiplier;
    float2 gMin = position - spread;
    float2 gMax = position + spread;

    float cellW = pc.screenWidth / float(pc.gridSize);
    float cellH = pc.screenHeight / float(pc.gridSize);

    uint numBins = pc.gridSize * pc.gridSize;
    uint mask = 0u;
    for (uint b = 0u; b < numBins; b++) {
        uint gridX = b % pc.gridSize;
        uint gridY = b / pc.gridSize;
        float2 binMin = float2(float(gridX) * cellW, float(gridY) * cellH);
        float2 binMax = float2(float(gridX + 1u) * cellW, float(gridY + 1u) * cellH);
        if (!(gMax.x < binMin.x || gMin.x > binMax.x || gMax.y < binMin.y || gMin.y > binMax.y)) {
            mask |= (1u << b);
        }
    }
    return mask;
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint lid: SV_GroupIndex, uint3 globalId: SV_DispatchThreadID, uint3 groupId: SV_GroupID) {
    uint idx = globalId.x;
    uint wgID = groupId.x;
    uint numWGs = WorkgroupCount().x;
    uint numBins = pc.gridSize * pc.gridSize;

    bool inRange = (idx < pc.numElements);
    float2 gPos;
    float gZ;
    float4 gCovInv;
    float4 gColorAlpha;
    bool validZ = false;

    if (inRange) {
        gZ = inZ[idx];
        gPos = inPos2D[idx];
        gCovInv = inCovInv[idx];
        gColorAlpha = inColorAlpha[idx];
        validZ = (gZ >= 0.0 && gZ <= 1.0);
    }

    // Compute overlap bitmask ONCE (replaces 16× overlapsWithBin calls each with an inverse)
    uint overlapMask = (validZ) ? computeOverlapMask(gPos, gCovInv) : 0u;

    for (uint b = 0u; b < numBins; b++) {
        bool overlaps = (overlapMask >> b & 1u) != 0u;

        sharedScan[lid] = overlaps ? 1u : 0u;
        GroupMemoryBarrierWithGroupSync();

        for (uint stride = 1u; stride < 128u; stride <<= 1u) {
            uint val = sharedScan[lid];
            if (lid >= stride) val += sharedScan[lid - stride];
            GroupMemoryBarrierWithGroupSync();
            sharedScan[lid] = val;
            GroupMemoryBarrierWithGroupSync();
        }

        uint inclusiveVal = sharedScan[lid];
        uint totalInWg = sharedScan[127];
        if (lid == 0u) sharedBinCounts[b] = totalInWg;

        if (overlaps) {
            uint exclusiveRank = inclusiveVal - 1u;
            uint basePos = prefixSums[numWGs * b + wgID];
            uint globalPos = basePos + exclusiveRank;
            if (globalPos < pc.maxOutput) {
                outPos2D[globalPos] = gPos;
                outZ[globalPos] = gZ;
                outBinMask[globalPos] = 1u << b;
                outCovInv[globalPos] = gCovInv;
                outColorAlpha[globalPos] = gColorAlpha;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (lid == 0u) {
        uint workgroupTotal = 0u;
        for (uint b = 0u; b < numBins; b++) {
            uint basePos = prefixSums[numWGs * b + wgID];
            uint count = sharedBinCounts[b];
            uint available = (basePos < pc.maxOutput) ? (pc.maxOutput - basePos) : 0u;
            workgroupTotal += min(count, available);
        }
        InterlockedAdd(totalCount[0], workgroupTotal);
    }
}
