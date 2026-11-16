// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// First pass of TreeMossLod's fitting to the pool: one thread per placed
// copy. For each of TREE_LOD_FIT_STEPS factors f = treeLodFitFactor(k) on the
// level boundaries, adds the Gaussians the copy would reserve in
// tree_lod_select.comp into the sums of `fit` (zeroed before the pass).

#define TREE_CAMERA_SET 0
#define TREE_CAMERA_BINDING 3
#include "tree_render_common.slang"

[[vk::binding(0, 0)]] StructuredBuffer<TreeLodInstance> instances;
[[vk::binding(1, 0)]] StructuredBuffer<TreeLodPatch> patches;
// [0] the step in use, [1] its factor (float bits), [2] unused, [3] unused, then the sums per step.
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> fit;
[[vk::binding(4, 0)]] StructuredBuffer<TreeRenderSettings> settings;

struct Push {
    uint instanceCount;
    uint capacity;
    uint pad0;
    uint pad1;
    vec4 boundaries; // xyz the pixel boundaries of levels 1 to 3, w the crossfade band
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared uint sums[TREE_LOD_FIT_STEPS];

[shader("compute")]
[numthreads(64, 1, 1)]
void main() {
    uint local = gl_LocalInvocationID.x;
    if (local < TREE_LOD_FIT_STEPS) {
        sums[local] = 0u;
    }
    barrier();
    uint i = gl_GlobalInvocationID.x;
    if (i < pc.instanceCount) {
        TreeLodInstance inst = instances[i];
        TreeLodPatch source = patches[inst.ids.x];
        vec3 center = inst.posYaw.xyz;
        float radius = source.bounds.x * inst.plane.z;
        float pixels = treePixelDiameter(camera.cameraWorldPos.xyz, center, radius, treeFocalPixels(settings[0]));
        if (inst.plane.z > 0.0f && pixels >= 0.65f && !treeSphereOutsideFrustum(camera.proj * camera.view, center, radius)) {
            for (uint k = 0u; k < TREE_LOD_FIT_STEPS; ++k) {
                atomicAdd(sums[k], treeLodReserved(source, pixels, pc.boundaries.xyz * treeLodFitFactor(k),
                                                   pc.boundaries.w));
            }
        }
    }
    barrier();
    if (local < TREE_LOD_FIT_STEPS && sums[local] > 0u) {
        atomicAdd(fit[4u + local], sums[local]);
    }
}
