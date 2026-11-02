// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Draws a triangle mesh for the visibility pass without vertex buffers: vertex
// i is corner i % 3 of triangle i / 3, whose vertex indices come from the
// triangle buffer, and its position from the position buffer, so the mesh can
// be changed on the GPU (e.g. by an optimizer) between draws. The camera is the
// view the view-index buffer selects, also set on the GPU.
import visibility_common;

[[vk::binding(0, 0)]] StructuredBuffer<float4> positions;     // xyz, w unused
[[vk::binding(1, 0)]] StructuredBuffer<uint> triangles;       // three vertex indices per triangle
[[vk::binding(2, 0)]] StructuredBuffer<float4x4> cameras;     // clip-from-world matrix per view
[[vk::binding(3, 0)]] StructuredBuffer<uint> viewIndex;       // one element: the view to draw

[shader("vertex")]
VisibilityVertex main(uint vertexIndex: SV_VulkanVertexID) {
    VisibilityVertex output;
    const float3 position = positions[triangles[vertexIndex]].xyz;
    output.position = mul(cameras[viewIndex[0]], float4(position, 1.0));
    output.triangle = vertexIndex / 3;
    return output;
}
