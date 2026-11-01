// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Fills the VkDrawMeshTasksIndirectCommandEXT buffer for the raster backend's
// mesh-shader draw (perf plan R5): groupCountX = ceil(visible / WG_SPLATS),
// matching gsplat_raster.mesh's WG_SPLATS splats per workgroup. Reads the
// visible count from the sort's count buffer (the value GaussianCullAndCompact wrote). Runs after
// GaussianCullAndCompact (via its dependency edge) and before the mesh draw (via the barrier).
import scalar_buffer;

static const uint WG_SPLATS = 32u;

struct MeshArgs {
    uint groupCountX;
    uint groupCountY;
    uint groupCountZ;
};

// One uint: the number of visible splats.
[[vk::binding(0, 0)]] ScalarBuffer<uint> visibleCount;
// One MeshArgs.
[[vk::binding(1, 0)]] RWScalarBuffer<MeshArgs> args;

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
    args[0].groupCountX = (visibleCount[0] + WG_SPLATS - 1u) / WG_SPLATS;
    args[0].groupCountY = 1u;
    args[0].groupCountZ = 1u;
}
