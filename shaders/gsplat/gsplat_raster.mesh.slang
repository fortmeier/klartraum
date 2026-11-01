// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Mesh-shader front end for the raster backend (perf plan R5), an additive,
// device-gated alternative to gsplat_raster.vert. One workgroup emits a batch of
// WG_SPLATS quads, reading the same precomputed Splat2D records the vertex path
// uses (R1) dereferenced through the sorted index. Outputs match the vertex
// shader's per-vertex interface so the existing gsplat_raster.frag is reused.
import gsplat_raster_types;
import scalar_buffer;

static const uint WG_SPLATS = 32u;

[[vk::binding(0, 1)]] ScalarBuffer<Splat2D> splat2D;
[[vk::binding(1, 1)]] ScalarBuffer<uint> sortedIndices;
// One uint: the number of visible splats.
[[vk::binding(2, 1)]] ScalarBuffer<uint> visibleCount;

[shader("mesh")]
[numthreads(WG_SPLATS, 1, 1)]
[outputtopology("triangle")]
void main(uint3 groupId: SV_GroupID, uint3 localId: SV_GroupThreadID,
          out vertices SplatVertex vertices[WG_SPLATS * 4u], out indices uint3 triangles[WG_SPLATS * 2u]) {
    uint base = groupId.x * WG_SPLATS;
    uint count = (visibleCount[0] > base) ? min(WG_SPLATS, visibleCount[0] - base) : 0u;

    SetMeshOutputCounts(count * 4u, count * 2u);
    uint lid = localId.x;
    if (lid >= count) return;

    uint splatId = sortedIndices[base + lid];
    Splat2D s = splat2D[splatId];

    uint v0 = lid * 4u;
    for (uint c = 0u; c < 4u; c++) {
        vertices[v0 + c] = splatVertex(s, c);
    }

    // Two triangles covering the quad {BL, BR, TL, TR}; cull mode is NONE so
    // winding is irrelevant.
    uint p0 = lid * 2u;
    triangles[p0 + 0u] = uint3(v0 + 0u, v0 + 1u, v0 + 2u);
    triangles[p0 + 1u] = uint3(v0 + 2u, v0 + 1u, v0 + 3u);
}
