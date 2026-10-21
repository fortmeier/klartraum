// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#version 450

// The terrain's selected nodes (TreeTerrain), one instance each: a grid of
// quads x quads, two triangles per quad sharing the diagonal from (i, j) to
// (i + 1, j + 1), with heights, normals and land cover from
// tree_terrain_vertices.comp.

#define TREE_CAMERA_SET 0
#define TREE_CAMERA_BINDING 0
#include "tree_render_common.glsl"

layout(std430, set = 1, binding = 0) readonly buffer Vertices {
    vec4 vertices[];
};
layout(std430, set = 1, binding = 4) readonly buffer Nodes {
    vec4 nodes[];
};
layout(std430, set = 1, binding = 5) readonly buffer Config {
    TreeTerrainConfig config;
};
layout(std430, set = 1, binding = 6) readonly buffer Covers {
    vec4 covers[];
};
layout(std430, set = 1, binding = 7) readonly buffer Extras {
    vec4 extras[];
};

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec4 outCover; // forest, meadow, field, scrub
layout(location = 3) out float outCanopy;

const ivec2 corners[6] = ivec2[6](ivec2(0, 0), ivec2(1, 0), ivec2(1, 1), ivec2(0, 0), ivec2(1, 1), ivec2(0, 1));

void main() {
    vec4 node = nodes[gl_InstanceIndex];
    int quads = int(config.lod.ranges.z);
    int q = gl_VertexIndex / 6;
    ivec2 ij = ivec2(q % quads, q / quads) + corners[gl_VertexIndex % 6];
    int step = 1 << uint(node.w);
    float unit = config.lod.root.z / (exp2(float(config.lod.levels.x - 1u)) * float(quads));
    int gx = int(round((node.x - config.lod.root.x) / node.z)) * quads * step;
    int gz = int(round((node.y - config.lod.root.y) / node.z)) * quads * step;
    int index = gl_InstanceIndex * (quads + 1) * (quads + 1) + ij.y * (quads + 1) + ij.x;
    vec4 v = vertices[index];
    vec3 p = vec3(config.lod.root.x + float(gx + ij.x * step) * unit, v.w,
                  config.lod.root.y + float(gz + ij.y * step) * unit);
    outWorldPos = p;
    outNormal = v.xyz;
    outCover = covers[index];
    outCanopy = extras[index].x;
    gl_Position = camera.proj * camera.view * vec4(p, 1.0f);
}
