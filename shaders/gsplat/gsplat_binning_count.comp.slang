// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import gsplat_types;
import scalar_buffer;

// SoA inputs from projection
[[vk::binding(0, 0)]] ScalarBuffer<float2> pos2D;
[[vk::binding(1, 0)]] ScalarBuffer<float> zBuf;
[[vk::binding(2, 0)]] ScalarBuffer<float4> covInv; // (m00,m10,m01,m11)

[[vk::binding(3, 0)]] RWScalarBuffer<uint> histogram;
[[vk::binding(4, 0)]] RWScalarBuffer<uint> offsets;

struct PushConstants {
    uint numElements;
    uint gridSize;
    float screenWidth;
    float screenHeight;
    float spreadMultiplier; // tunable: Gaussian footprint radius multiplier (default 2.5)
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

// The workgroup's count per bin.
groupshared uint sharedHisto[16];

// Compute 2D spread (stddev × multiplier) directly from the inverse covariance float4,
// avoiding a full 2x2 matrix inverse.
// covInv is stored column-major as (m00, m10, m01, m11):
//   mat = [[m00, m01], [m10, m11]]  with symmetric m10==m01
// cov[0][0] = m11 / det,  cov[1][1] = m00 / det,  det = m00*m11 - m01*m10
float2 computeSpread(float4 ci) {
    float det = ci.x * ci.w - ci.y * ci.z;
    return sqrt(float2(ci.w, ci.x) / det) * pc.spreadMultiplier;
}

bool overlapsWithBin(float2 position, float2 spread, uint binIdx) {
    float cellW = pc.screenWidth / float(pc.gridSize);
    float cellH = pc.screenHeight / float(pc.gridSize);
    uint gridX = binIdx % pc.gridSize;
    uint gridY = binIdx / pc.gridSize;
    float2 binMin = float2(float(gridX) * cellW, float(gridY) * cellH);
    float2 binMax = float2(float(gridX + 1u) * cellW, float(gridY + 1u) * cellH);
    float2 gMin = position - spread;
    float2 gMax = position + spread;
    return !(gMax.x < binMin.x || gMin.x > binMax.x || gMax.y < binMin.y || gMin.y > binMax.y);
}

[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint lid: SV_GroupIndex, uint3 globalId: SV_DispatchThreadID, uint3 groupId: SV_GroupID) {
    uint idx = globalId.x;
    uint wgID = groupId.x;
    uint numBins = pc.gridSize * pc.gridSize;

    if (lid < numBins) sharedHisto[lid] = 0u;
    GroupMemoryBarrierWithGroupSync();

    if (idx < pc.numElements) {
        float z = zBuf[idx];
        if (z >= 0.0 && z <= 1.0) {
            float2 pos = pos2D[idx];
            float4 ci = covInv[idx];
            float2 spread = computeSpread(ci); // computed once, shared across all bin tests
            for (uint b = 0u; b < numBins; b++) {
                if (overlapsWithBin(pos, spread, b)) {
                    InterlockedAdd(sharedHisto[b], 1u);
                }
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (lid < numBins) {
        histogram[WorkgroupCount().x * lid + wgID] = sharedHisto[lid];
    }

    if (lid == 0) {
        offsets[wgID] = (wgID == 0) ? 0u : 0xFFFFFFFFu;
    }
}
