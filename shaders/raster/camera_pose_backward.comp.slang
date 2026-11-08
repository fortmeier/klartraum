// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The gradients of the drawn view's pose (camera_pose.slang) from the vertex gradients of the same step, which hold
// dL/dx with the camera fixed. The camera sees x at y = R(q) (x - e) and projects y, so dL/dy = R(q) dL/dx; bwd_diff of
// viewPoint() turns it into each vertex's share of dL/dq and dL/de. A subgroup sums the shares before one lane adds
// them into the view's eight floats (q, then e and a zero).
import camera_pose;

struct PushConstants {
    uint vertexCount;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float4> positions;       // xyz per vertex
[[vk::binding(1, 0)]] StructuredBuffer<float> vertexGradients;  // four per vertex: dL/dx
[[vk::binding(2, 0)]] StructuredBuffer<float4> poses;           // two per view: q, (e, 0)
[[vk::binding(3, 0)]] StructuredBuffer<uint> viewIndex;         // the drawn view
[[vk::binding(4, 0)]] RWStructuredBuffer<float> poseGradients;  // eight per view, added to

[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    const uint i = globalId.x;
    const uint view = viewIndex[0];
    const float4 q = poses[2 * view];
    const float3 e = poses[2 * view + 1].xyz;
    float4 qGradient = 0.0;
    float3 eGradient = 0.0;
    if (i < pc.vertexCount) {
        const float3 x = positions[i].xyz;
        const float3 g = float3(vertexGradients[4 * i], vertexGradients[4 * i + 1], vertexGradients[4 * i + 2]);
        var qPair = diffPair(q);
        var ePair = diffPair(e);
        bwd_diff(viewPoint)(qPair, ePair, x, rotate(q, g));
        qGradient = qPair.d;
        eGradient = ePair.d;
    }
    const float4 qSum = WaveActiveSum(qGradient);
    const float3 eSum = WaveActiveSum(eGradient);
    if (WaveIsFirstLane() && (any(qSum != 0.0) || any(eSum != 0.0))) {
        for (uint c = 0; c < 4; ++c) InterlockedAdd(poseGradients[8 * view + c], qSum[c]);
        for (uint c = 0; c < 3; ++c) InterlockedAdd(poseGradients[8 * view + 4 + c], eSum[c]);
    }
}
