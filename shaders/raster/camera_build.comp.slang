// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Builds every view's camera from its pose (camera_pose.slang): M = P R(q) T(-e), and its eye e.
import camera_pose;

struct PushConstants {
    uint viewCount;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float4x4> rests;    // P per view
[[vk::binding(1, 0)]] StructuredBuffer<float4> poses;      // two per view: q, (e, 0)
[[vk::binding(2, 0)]] RWStructuredBuffer<float4x4> cameras; // M per view, written
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> eyes;      // (e, 1) per view, written

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    const uint v = globalId.x;
    if (v >= pc.viewCount) return;
    const float4 q = poses[2 * v];
    const float3 e = poses[2 * v + 1].xyz;
    const float4x4 translation =
        float4x4(1.0, 0.0, 0.0, -e.x, 0.0, 1.0, 0.0, -e.y, 0.0, 0.0, 1.0, -e.z, 0.0, 0.0, 0.0, 1.0);
    cameras[v] = mul(rests[v], mul(rotationMatrix(q), translation));
    eyes[v] = float4(e, 1.0);
}
