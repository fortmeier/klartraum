// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Adds every triangle's area-weighted normal (the cross product of two of its edges) to its three vertices: after
// one run on cleared buffers each vertex holds the sum, whose direction is its smooth normal.

struct PushConstants {
    uint triangleCount;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float4> positions;  // xyz per vertex
[[vk::binding(1, 0)]] StructuredBuffer<uint> triangles;    // three vertex indices per triangle
[[vk::binding(2, 0)]] RWStructuredBuffer<float> normals;   // four floats per vertex, xyz added to

[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    const uint t = globalId.x;
    if (t >= pc.triangleCount) return;
    const uint v0 = triangles[3 * t], v1 = triangles[3 * t + 1], v2 = triangles[3 * t + 2];
    const float3 p0 = positions[v0].xyz;
    const float3 n = cross(positions[v1].xyz - p0, positions[v2].xyz - p0);
    for (uint c = 0; c < 3; ++c) {
        InterlockedAdd(normals[4 * v0 + c], n[c]);
        InterlockedAdd(normals[4 * v1 + c], n[c]);
        InterlockedAdd(normals[4 * v2 + c], n[c]);
    }
}
