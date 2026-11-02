// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;
import scalar_buffer;

// The terrain as Gaussians (TreeTerrainGaussians): one workgroup per
// selected node writes a flat Gaussian for each block of 2 x 2 quads of its
// grid, at the block's middle vertex of the node's morphed vertices
// (tree_terrain_vertices.comp), jittered in its tangent plane against moire.
// Each is stretched along the slope to the block's size on the surface, so
// the Gaussians close up on steep ground too. They are coloured with the
// classical terrain's materials (treeTerrainAlbedo(), patterns finer than a
// block averaged) and lit like the trees' Gaussians. Nodes beyond the pool
// are left out and counted; Gaussians outside the soil's outline are hidden.

#define TREE_CAMERA_SET 0
#define TREE_CAMERA_BINDING 11
#include "tree_render_common.slang"
#include "tree_shading.slang"
#define TREE_LIGHT_SET 0
#define TREE_LIGHT_BINDING 10
#include "tree_light.slang"

[[vk::binding(0, 0)]] RWScalarBuffer<vec3> outPos;
[[vk::binding(1, 0)]] RWScalarBuffer<vec4> outRot;
[[vk::binding(2, 0)]] RWScalarBuffer<vec3> outScale;
[[vk::binding(3, 0)]] RWScalarBuffer<vec4> outColAlpha;
[[vk::binding(4, 0)]] StructuredBuffer<vec4> nodes;
[[vk::binding(5, 0)]] StructuredBuffer<TreeTerrainConfig> configBuffer;
[[vk::binding(6, 0)]] StructuredBuffer<vec4> vertices;
[[vk::binding(7, 0)]] StructuredBuffer<vec4> covers;
[[vk::binding(8, 0)]] StructuredBuffer<TreeRenderSettings> settings;
// [0] nodes left out for lack of room.
[[vk::binding(9, 0)]] RWStructuredBuffer<uint> stats;
// The vertices' canopy weight in x.
[[vk::binding(12, 0)]] StructuredBuffer<vec4> extras;

struct Push {
    uint offset;   // of the pool in the output buffers
    uint capacity; // slots of the pool
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

static const float SH_C0 = 0.28209479f;

[shader("compute")]
[numthreads(256, 1, 1)]
void main() {
    uint n = gl_WorkGroupID.x;
    int quads = int(configBuffer[0].lod.ranges.z);
    int blocks = quads / 2;
    uint perNode = uint(blocks * blocks);
    if ((n + 1u) * perNode > pc.capacity) {
        if (gl_LocalInvocationID.x == 0u) {
            atomicAdd(stats[0], 1u);
        }
        return;
    }
    vec4 node = nodes[n];
    TreeRenderSettings s = settings[0];
    TreeTerrainParams t = configBuffer[0].terrain;
    int row = quads + 1;
    uint base = n * uint(row * row);
    float spacing = node.z / float(quads);
    for (int q = int(gl_LocalInvocationID.x); q < blocks * blocks; q += 256) {
        int i = 2 * (q % blocks) + 1;
        int j = 2 * (q / blocks) + 1;
        uint v = base + uint(j * row + i);
        vec4 vertexValue = vertices[v];
        vec3 pos = vec3(node.x + float(i) * spacing, vertexValue.w, node.y + float(j) * spacing);
        vec3 normal = normalize(vertexValue.xyz);
        uint slot = pc.offset + n * perNode + uint(q);
        // The block's edges on the surface: along x and z over two quads.
        vec3 ex = vec3(2.0f * spacing, vertices[v + 1u].w - vertices[v - 1u].w, 0.0f);
        vec3 ez = vec3(0.0f, vertices[v + uint(row)].w - vertices[v - uint(row)].w, 2.0f * spacing);
        vec3 tx = normalize(ex - dot(ex, normal) * normal);
        vec3 tz = cross(tx, normal);
        uint key = uint(int(round((node.x - configBuffer[0].lod.root.x) / spacing)) + i) * 7919u +
                   uint(int(round((node.y - configBuffer[0].lod.root.y) / spacing)) + j);
        pos += 0.25f * (treeRandSigned(key, 0x71u) * tx * length(ex) + treeRandSigned(key, 0x72u) * tz * length(ez));
        if (treeTerrainRadius(t, pos.x, pos.z) >= 1.0f) {
            outPos[slot] = pos;
            outRot[slot] = vec4(0.0f, 0.0f, 0.0f, 1.0f);
            outScale[slot] = vec3(1e-6f);
            outColAlpha[slot] = vec4(0.0f);
            continue;
        }

        TreeLandCover cover;
        cover.vegetation = clamp(covers[v], vec4(0.0f), vec4(1.0f));
        cover.bare = vec4(max(1.0f - dot(cover.vegetation, vec4(1.0f)), 0.0f), 0.0f, 0.0f, 0.0f);
        vec3 albedo = treeTerrainAlbedo(t, cover, pos, 2.0f * spacing, s.season.x);
        float canopy = clamp(extras[v].x, 0.0f, 1.0f);
        if (canopy > 0.0f) {
            albedo = mix(albedo, treeCanopyAlbedo(t, configBuffer[0].placement, pos, 2.0f * spacing, s.season.x), canopy);
        }
        albedo = mix(albedo, vec3(0.8f), (1.0f - 0.5f * canopy) * treeTerrainSnow(t, pos, normal, s.season.x, s.season.y));
        vec2 light = treeLightAt(s, pos + 0.5f * s.lightGridMin.w * normal);
        light.x *= clamp(extras[v].y, 0.0f, 1.0f);
        vec3 color = treeShadeLit(albedo, normal, s, light) + treePointLight(albedo, normal, pos, s);
        vec3 displayed = treeOutputColor(treeAtmosphere(color, pos, camera.cameraWorldPos.xyz, s), s).rgb;

        // Turned so that its x axis runs along tx.
        vec4 rot = qfromZ(normal);
        float angle = atan(dot(tx, qrot(rot, vec3(0.0f, 1.0f, 0.0f))), dot(tx, qrot(rot, vec3(1.0f, 0.0f, 0.0f))));
        outPos[slot] = pos;
        outRot[slot] = normalize(qmul(rot, qaxis(vec3(0.0f, 0.0f, 1.0f), angle)));
        outScale[slot] = vec3(0.6f * length(ex), 0.6f * length(ez), 0.05f * spacing);
        outColAlpha[slot] = vec4((displayed - vec3(0.5f)) / SH_C0, 0.95f);
    }
}
