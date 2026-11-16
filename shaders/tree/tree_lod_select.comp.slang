// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// Second pass of TreeMossLod: one thread per placed patch. Culls patches
// outside the view or smaller than 0.65 pixels, picks the level of detail from
// the patch's diameter in pixels (treeLodBlend()), and reserves its Gaussians
// in the pool: those of its level and, while crossfading, of the next coarser
// one. Patches that no longer fit are left out and counted. With pc.fit, the
// boundaries are multiplied by the factor of tree_lod_fit.comp, so that the
// copies fit the pool.

#define TREE_CAMERA_SET 0
#define TREE_CAMERA_BINDING 4
#include "tree_render_common.slang"

[[vk::binding(0, 0)]] StructuredBuffer<TreeLodInstance> instances;
[[vk::binding(1, 0)]] StructuredBuffer<TreeLodPatch> patches;
// Per instance: x first pool slot, y Gaussians, z level, w fade (float bits).
[[vk::binding(2, 0)]] RWStructuredBuffer<uvec4> records;
// [0] Gaussians reserved, [1] patches left out for lack of room; zeroed before the pass.
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> counter;
[[vk::binding(5, 0)]] StructuredBuffer<TreeRenderSettings> settings;

// [1] the factor on the boundaries (float bits); see tree_lod_fit.comp. Read with pc.fit only.
[[vk::binding(6, 0)]] StructuredBuffer<uint> fit;

struct Push {
    uint instanceCount;
    uint capacity;
    uint fit; // 1: the boundaries times the factor in `fit`
    uint pad1;
    vec4 boundaries; // xyz the pixel boundaries of levels 1 to 3, w the crossfade band
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.instanceCount) {
        return;
    }
    TreeLodInstance inst = instances[i];
    TreeLodPatch source = patches[inst.ids.x];
    vec3 center = inst.posYaw.xyz;
    float radius = source.bounds.x * inst.plane.z;
    float pixels = treePixelDiameter(camera.cameraWorldPos.xyz, center, radius, treeFocalPixels(settings[0]));
    if (pixels < 0.65f || treeSphereOutsideFrustum(camera.proj * camera.view, center, radius)) {
        records[i] = uvec4(0u);
        return;
    }
    vec3 boundaries = pc.boundaries.xyz * (pc.fit != 0u ? uintBitsToFloat(fit[1]) : 1.0f);
    vec2 lod = treeLodBlend(pixels, boundaries, pc.boundaries.w);
    uint level = uint(lod.x);
    uint count = treeLodReserved(source, pixels, boundaries, pc.boundaries.w);
    uint base = atomicAdd(counter[0], count);
    if (base + count > pc.capacity) {
        atomicAdd(counter[1], 1u);
        records[i] = uvec4(0u);
        return;
    }
    records[i] = uvec4(base, count, level, floatBitsToUint(lod.y));
}
