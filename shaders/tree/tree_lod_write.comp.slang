// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;
import scalar_buffer;

// Third pass of TreeMossLod: one workgroup per placed patch writes the
// Gaussians it reserved: its level's, and while crossfading the next coarser
// level's, with opacities that sum to the level's. They are turned and
// scaled with the patch, laid on its ground plane and coloured like captured
// Gaussians (treeCapturedColor(): tint, the trees' shadows, snow), or for
// copies of trees (TreeLodInstance::ids.w) lit from their albedos like the
// trees' Gaussians; all through the miniature world's haze.

#define TREE_CAMERA_SET 0
#define TREE_CAMERA_BINDING 10
#include "tree_render_common.slang"
#define TREE_LIGHT_SET 0
#define TREE_LIGHT_BINDING 9
#include "tree_light.slang"

[[vk::binding(0, 0)]] RWScalarBuffer<vec3> outPos;
[[vk::binding(1, 0)]] RWScalarBuffer<vec4> outRot;
[[vk::binding(2, 0)]] RWScalarBuffer<vec3> outScale;
[[vk::binding(3, 0)]] RWScalarBuffer<vec4> outColAlpha;
[[vk::binding(4, 0)]] StructuredBuffer<TreeLodSplat> library;
[[vk::binding(5, 0)]] StructuredBuffer<TreeLodPatch> patches;
[[vk::binding(6, 0)]] StructuredBuffer<TreeLodInstance> instances;
[[vk::binding(7, 0)]] StructuredBuffer<uvec4> records;
[[vk::binding(8, 0)]] StructuredBuffer<TreeRenderSettings> settings;

struct Push {
    uint offset; // of the pool in the output buffers
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

static const float SH_C0 = 0.28209479f;

[shader("compute")]
[numthreads(64, 1, 1)]
void main() {
    uint i = gl_WorkGroupID.x;
    uvec4 record = records[i];
    if (record.y == 0u) {
        return;
    }
    TreeLodInstance inst = instances[i];
    TreeLodPatch source = patches[inst.ids.x];
    TreeRenderSettings s = settings[0];
    uint level = record.z;
    float fade = uintBitsToFloat(record.w);
    vec4 turn = qaxis(vec3(0.0f, 1.0f, 0.0f), inst.posYaw.w);
    float scale = inst.plane.z;

    for (uint k = gl_LocalInvocationID.x; k < record.y; k += 64u) {
        bool coarser = k >= source.count[level];
        uint lodLevel = coarser ? level + 1u : level;
        TreeLodSplat g = library[source.offset[lodLevel] + (coarser ? k - source.count[level] : k)];
        float weight = coarser ? fade : 1.0f - fade;

        // Turned about +y and scaled; the height above the patch's ground stays and follows the ground plane.
        vec3 local = qrot(turn, scale * vec3(g.pos.x, 0.0f, g.pos.z));
        vec3 pos = inst.posYaw.xyz + local;
        pos.y += inst.plane.x * local.x + inst.plane.y * local.z + scale * g.pos.y;
        vec4 rot = normalize(qmul(turn, g.rot));
        vec3 size = scale * g.scale.xyz;

        vec3 normal = treeGaussianNormal(rot, size);
        // The shadows of the trees and of the static occluders (pot, captured objects), and the terrain's shadow on
        // the copy (ids.z, float bits of the share of the sun it takes away).
        vec2 light = treeLightAt(s, pos);
        light.x *= 1.0f - uintBitsToFloat(inst.ids.z);
        vec3 displayed;
        if (inst.ids.w == 1u) {
            // A tree's albedo, lit like the trees' Gaussians, with snow in winter and through the haze.
            vec3 albedo = mix(g.color.rgb, vec3(0.8f), 0.7f * treeSnowOn(normal, pos, s.season.y));
            vec3 color = treeShadeLit(albedo, normal, s, light) + treePointLight(albedo, normal, pos, s);
            displayed = treeDisplayColor(treeAtmosphere(color, pos, camera.cameraWorldPos.xyz, s), s.sky.w);
        } else {
            displayed = treeCapturedColor(vec3(0.5f) + SH_C0 * g.color.rgb, normal, pos, treeCapturedShade(s, light), s);
            // Through the miniature world's haze; the colours are displayed ones, so the haze's is displayed too.
            vec4 haze = treeHaze(pos, camera.cameraWorldPos.xyz, s);
            displayed = mix(displayed, treeDisplayColor(haze.rgb, s.sky.w), haze.a);
        }

        uint slot = pc.offset + record.x + k;
        outPos[slot] = pos;
        outRot[slot] = rot;
        outScale[slot] = size;
        outColAlpha[slot] = vec4((displayed - vec3(0.5f)) / SH_C0, g.color.a * weight * (1.0f - uintBitsToFloat(inst.ids.y)));
    }
}
