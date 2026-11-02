// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;
import scalar_buffer;

// The visible clumps of grass as Gaussians (TreeGrassGaussians): one thread
// per clump reserves its slots in the pool, a flat Gaussian per blade
// (treeGrassBlade(), along its middle, fainter when thinner than a pixel) for clumps
// within a third of the radius, and a single tuft beyond, as dense as the
// clump's blades. Lit like the classical blades, through the haze. Clumps
// that find no room are left out.

#define TREE_CAMERA_SET 0
#define TREE_CAMERA_BINDING 8
#include "tree_render_common.slang"
#include "tree_grass_common.slang"
#define TREE_LIGHT_SET 0
#define TREE_LIGHT_BINDING 7
#include "tree_light.slang"

[[vk::binding(0, 0)]] RWScalarBuffer<vec3> outPos;
[[vk::binding(1, 0)]] RWScalarBuffer<vec4> outRot;
[[vk::binding(2, 0)]] RWScalarBuffer<vec3> outScale;
[[vk::binding(3, 0)]] RWScalarBuffer<vec4> outColAlpha;
[[vk::binding(4, 0)]] StructuredBuffer<TreeGrassVisible> visibleList;
[[vk::binding(5, 0)]] StructuredBuffer<uint> counter;
[[vk::binding(6, 0)]] StructuredBuffer<TreeRenderSettings> settings;
[[vk::binding(9, 0)]] StructuredBuffer<TreeGrassSetup> setupBuffer;
// [0] the wind (treeGrassBlade()).
[[vk::binding(10, 0)]] StructuredBuffer<vec4> frame;
// [0] slots taken this frame; zeroed before the pass.
[[vk::binding(11, 0)]] RWStructuredBuffer<uint> taken;

struct Push {
    uint offset;   // of the pool in the output buffers
    uint capacity; // slots of the pool
    uint clumps;   // capacity of the visible clumps
    uint pad;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

static const float SH_C0 = 0.28209479f;

vec4 lit(vec3 albedo, vec3 normal, vec3 pos, float sunlight, float alpha) {
    TreeRenderSettings s = settings[0];
    vec2 light = treeLightAt(s, pos);
    light.x *= sunlight;
    vec3 color = treeShadeLit(albedo, normal, s, light) + treePointLight(albedo, normal, pos, s);
    vec3 displayed = treeDisplayColor(treeAtmosphere(color, pos, camera.cameraWorldPos.xyz, s), s.sky.w);
    return vec4((displayed - vec3(0.5f)) / SH_C0, alpha);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= min(counter[0], pc.clumps)) {
        return;
    }
    TreeGrassVisible clump = visibleList[i];
    TreeGrassParams g = setupBuffer[0].grass;
    bool near = clump.ground.w < 0.33f * g.grid.y;
    uint count = near ? uint(clump.kind.x) : 1u;
    uint base = atomicAdd(taken[0], count);
    if (base + count > pc.capacity) {
        return;
    }
    base += pc.offset;
    vec3 ground = normalize(vec3(-clump.ground.x, 1.0f, -clump.ground.y));
    float far = smoothstep(0.33f * g.grid.y, g.grid.y, clump.ground.w);
    vec3 albedo = mix(clump.albedo.rgb * mix(0.85f, 1.0f, far), vec3(0.8f), 0.5f * clump.ground.z);
    if (!near) {
        // A tuft standing for the clump's blades: about as wide as its cell, as dense as its blades.
        float h = clump.posHeight.w * (1.0f - 0.85f * clump.ground.z);
        vec3 pos = clump.posHeight.xyz + vec3(0.0f, 0.4f * h, 0.0f);
        outPos[base] = pos;
        outRot[base] = vec4(0.0f, 0.0f, 0.0f, 1.0f);
        outScale[base] = vec3(0.25f * g.grid.x, 0.3f * h, 0.25f * g.grid.x);
        float alpha = 0.2f + 0.5f * clamp(clump.kind.x / g.grid.z, 0.0f, 1.0f);
        outColAlpha[base] = lit(albedo, ground, pos, clump.kind.z, alpha);
        return;
    }
    float focal = treeFocalPixels(settings[0]);
    for (uint k = 0u; k < count; ++k) {
        GrassBlade b = treeGrassBlade(clump, g, k, frame[0]);
        vec3 middle = treeGrassBladePoint(b, 0.5f);
        vec3 tip = treeGrassBladePoint(b, 1.0f);
        vec3 axis = normalize(tip - b.foot);
        // Turned about +y so that local z points where it leans, then tilted about local x (local y along it).
        vec2 lean = length(axis.xz) > 1e-4f ? normalize(axis.xz) : b.facing;
        vec4 turn = qaxis(vec3(0.0f, 1.0f, 0.0f), 0.5f * TREE_PI - atan(lean.y, lean.x));
        vec4 rot = normalize(qmul(turn, qaxis(vec3(1.0f, 0.0f, 0.0f), acos(clamp(axis.y, -1.0f, 1.0f)))));
        float pixels = focal / max(treeViewDepth(middle), 1e-4f);
        float width = g.blade.y / g.grid.w;
        // The splatting widens every Gaussian by a variance of 0.3 square pixels; a blade narrower than that keeps
        // its share of the widened footprint as opacity, so it covers as much as it would.
        float sigma = 0.25f * width * pixels;
        float coverage = sigma / sqrt(sigma * sigma + 0.3f);
        float extent = distance(tip, b.foot);
        // Facing the camera and leaning up, as the classical blades.
        vec3 face = normalize(cross(vec3(-lean.y, 0.0f, lean.x), axis));
        face *= dot(face, camera.cameraWorldPos.xyz - middle) < 0.0f ? -1.0f : 1.0f;
        vec3 normal = normalize(face + vec3(0.0f, 1.2f, 0.0f));
        outPos[base + k] = 0.5f * (b.foot + tip) + 0.5f * (middle - 0.5f * (b.foot + tip));
        outRot[base + k] = rot;
        // Two sigmas across are the blade's width.
        outScale[base + k] = vec3(0.25f * width, 0.25f * extent, 0.1f * width);
        outColAlpha[base + k] = lit(albedo, normal, middle, clump.kind.z, 0.95f * coverage);
    }
}
