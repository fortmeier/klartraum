// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Trivial vertex shader for the raster backend: all per-splat math (covariance
// projection, eigendecomposition, SH colour) is precomputed once per splat by
// gsplat_raster_project.comp and stored in the Splat2D buffer. This stage just
// reads one record, places the quad corner along the precomputed eigen-axes, and
// passes conic + colour + centre through to the fragment shader. The quad is a
// triangle strip drawn from the vertex index, with no vertex buffer.
import gsplat_raster_types;
import scalar_buffer;

// Precomputed per-splat 2D attributes, keyed by splat id.
[[vk::binding(0, 1)]] ScalarBuffer<Splat2D> splat2D;
// The sorted (back-to-front) splat-id permutation.
[[vk::binding(1, 1)]] ScalarBuffer<uint> sortedIndices;

[shader("vertex")]
SplatVertex main(uint vertexIndex: SV_VulkanVertexID, uint instanceIndex: SV_VulkanInstanceID) {
    uint splatId = sortedIndices[instanceIndex];
    return splatVertex(splat2D[splatId], vertexIndex);
}
