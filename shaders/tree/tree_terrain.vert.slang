// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// The terrain's selected nodes (TreeTerrain), one instance each: a grid of
// quads x quads, two triangles per quad sharing the diagonal from (i, j) to
// (i + 1, j + 1), with heights, normals and land cover from
// tree_terrain_vertices.comp.

#define TREE_CAMERA_SET 0
#define TREE_CAMERA_BINDING 0
#include "tree_render_common.slang"

[[vk::binding(0, 1)]] StructuredBuffer<vec4> vertices;
[[vk::binding(4, 1)]] StructuredBuffer<vec4> nodes;
[[vk::binding(5, 1)]] StructuredBuffer<TreeTerrainConfig> configBuffer;
[[vk::binding(6, 1)]] StructuredBuffer<vec4> covers;
[[vk::binding(7, 1)]] StructuredBuffer<vec4> extras;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec4 outCover; // forest, meadow, field, scrub
layout(location = 3) out float outCanopy;
layout(location = 4) out float outSunlight;

static const ivec2 corners[6] = ivec2[6](ivec2(0, 0), ivec2(1, 0), ivec2(1, 1), ivec2(0, 0), ivec2(1, 1), ivec2(0, 1));

[shader("vertex")]
void main() {
    vec4 node = nodes[gl_InstanceIndex];
    int quads = int(configBuffer[0].lod.ranges.z);
    int q = gl_VertexIndex / 6;
    ivec2 ij = ivec2(q % quads, q / quads) + corners[gl_VertexIndex % 6];
    int nodeStep = 1 << uint(node.w);
    float unit = configBuffer[0].lod.root.z / (exp2(float(configBuffer[0].lod.levels.x - 1u)) * float(quads));
    int gx = int(round((node.x - configBuffer[0].lod.root.x) / node.z)) * quads * nodeStep;
    int gz = int(round((node.y - configBuffer[0].lod.root.y) / node.z)) * quads * nodeStep;
    int index = gl_InstanceIndex * (quads + 1) * (quads + 1) + ij.y * (quads + 1) + ij.x;
    vec4 v = vertices[index];
    vec3 p = vec3(configBuffer[0].lod.root.x + float(gx + ij.x * nodeStep) * unit, v.w,
                  configBuffer[0].lod.root.y + float(gz + ij.y * nodeStep) * unit);
    outWorldPos = p;
    outNormal = v.xyz;
    outCover = covers[index];
    outCanopy = extras[index].x;
    outSunlight = extras[index].y;
    gl_Position = camera.proj * camera.view * vec4(p, 1.0f);
}
