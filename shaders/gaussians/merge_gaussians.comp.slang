// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Concatenates two sets of Gaussians: A's first, then B's. The SH buffers are
// coefficient-major (sh[k * count + i]), so every coefficient is moved to
// the merged count's layout.
import scalar_buffer;

struct PushConstants {
    uint countA;
    uint countB;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] ScalarBuffer<float3> posA;
[[vk::binding(1, 0)]] ScalarBuffer<float4> rotA;
[[vk::binding(2, 0)]] ScalarBuffer<float3> scaleA;
[[vk::binding(3, 0)]] ScalarBuffer<float4> colAlphaA;
[[vk::binding(4, 0)]] ScalarBuffer<float> shRA;
[[vk::binding(5, 0)]] ScalarBuffer<float> shGA;
[[vk::binding(6, 0)]] ScalarBuffer<float> shBA;
[[vk::binding(7, 0)]] ScalarBuffer<float3> posB;
[[vk::binding(8, 0)]] ScalarBuffer<float4> rotB;
[[vk::binding(9, 0)]] ScalarBuffer<float3> scaleB;
[[vk::binding(10, 0)]] ScalarBuffer<float4> colAlphaB;
[[vk::binding(11, 0)]] ScalarBuffer<float> shRB;
[[vk::binding(12, 0)]] ScalarBuffer<float> shGB;
[[vk::binding(13, 0)]] ScalarBuffer<float> shBB;
[[vk::binding(14, 0)]] RWScalarBuffer<float3> outPos;
[[vk::binding(15, 0)]] RWScalarBuffer<float4> outRot;
[[vk::binding(16, 0)]] RWScalarBuffer<float3> outScale;
[[vk::binding(17, 0)]] RWScalarBuffer<float4> outColAlpha;
[[vk::binding(18, 0)]] RWScalarBuffer<float> outShR;
[[vk::binding(19, 0)]] RWScalarBuffer<float> outShG;
[[vk::binding(20, 0)]] RWScalarBuffer<float> outShB;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint i = globalId.x;
    uint count = pc.countA + pc.countB;
    if (i >= count) {
        return;
    }
    if (i < pc.countA) {
        outPos[i] = posA[i];
        outRot[i] = rotA[i];
        outScale[i] = scaleA[i];
        outColAlpha[i] = colAlphaA[i];
        for (uint k = 0; k < 15; ++k) {
            outShR[k * count + i] = shRA[k * pc.countA + i];
            outShG[k * count + i] = shGA[k * pc.countA + i];
            outShB[k * count + i] = shBA[k * pc.countA + i];
        }
    } else {
        uint j = i - pc.countA;
        outPos[i] = posB[j];
        outRot[i] = rotB[j];
        outScale[i] = scaleB[j];
        outColAlpha[i] = colAlphaB[j];
        for (uint k = 0; k < 15; ++k) {
            outShR[k * count + i] = shRB[k * pc.countB + j];
            outShG[k * count + i] = shGB[k * pc.countB + j];
            outShB[k * count + i] = shBB[k * pc.countB + j];
        }
    }
}
