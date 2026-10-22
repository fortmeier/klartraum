// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#version 450

// The blades of the visible clumps of grass (TreeGrass), one instance per
// clump of grid.z x 15 vertices: a blade of three segments narrowing to its
// tip (two triangles each, one for the tip), standing on the ground's slope
// within the clump's cell, bent towards its facing and swayed by the wind.
// Blades beyond the clump's count (kind.x) collapse outside the view. Far
// blades widen to a pixel.

#define TREE_CAMERA_SET 0
#define TREE_CAMERA_BINDING 0
#include "tree_render_common.glsl"

layout(std430, set = 1, binding = 0) readonly buffer Visible {
    TreeGrassVisible visible[];
};
layout(std430, set = 1, binding = 1) readonly buffer Settings {
    TreeRenderSettings settings[];
};
layout(std430, set = 1, binding = 4) readonly buffer Setup {
    TreeGrassSetup setup;
};
// [0] the wind: x time in seconds, y its strength in [0, 1], zw the direction it blows to (x, z).
layout(std430, set = 1, binding = 5) readonly buffer Frame {
    vec4 frame[];
};

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec3 outAlbedo;
layout(location = 3) out float outSunlight;

// Per vertex of a blade: x the level (0 the foot, 3 the tip), y the side (-1, 1, 0 at the tip).
const ivec2 corners[15] = ivec2[15](ivec2(0, -1), ivec2(0, 1), ivec2(1, 1), ivec2(0, -1), ivec2(1, 1), ivec2(1, -1),
                                    ivec2(1, -1), ivec2(1, 1), ivec2(2, 1), ivec2(1, -1), ivec2(2, 1), ivec2(2, -1),
                                    ivec2(2, -1), ivec2(2, 1), ivec2(3, 0));

void main() {
    TreeGrassVisible clump = visible[gl_InstanceIndex];
    TreeGrassParams g = setup.grass;
    vec4 wind = frame[0];
    uint blade = uint(gl_VertexIndex) / 15u;
    if (float(blade) >= clump.kind.x) {
        gl_Position = vec4(2.0f, 2.0f, 2.0f, 1.0f);
        return;
    }
    ivec2 corner = corners[gl_VertexIndex % 15];
    uint key = treeHash2(floatBitsToUint(clump.kind.w), blade);
    float h = clump.posHeight.w * (0.6f + 0.7f * treeRand(key, 1u)) * (1.0f - 0.85f * clump.ground.z);

    // The foot within the clump's cell, on the ground's slope.
    float a = 2.0f * TREE_PI * treeRand(key, 2u);
    vec2 o = 0.5f * g.grid.x * sqrt(treeRand(key, 3u)) * vec2(cos(a), sin(a));
    vec3 foot = clump.posHeight.xyz + vec3(o.x, clump.ground.x * o.x + clump.ground.y * o.y, o.y);

    // Bent towards its facing, more for longer blades, and swayed by the wind.
    float facing = clump.kind.y + 2.0f * TREE_PI * treeRand(key, 4u);
    vec2 dir = vec2(cos(facing), sin(facing));
    vec2 side = vec2(-dir.y, dir.x);
    float lean = 0.15f + 0.45f * treeRand(key, 5u);
    float phase = 2.0f * wind.x + 0.35f * dot(foot.xz * g.grid.w, wind.zw) + 6.0f * treeRand(key, 6u);
    vec2 bend = lean * dir + wind.y * (0.35f + 0.25f * sin(phase)) * wind.zw;

    float t = float(corner.x) / 3.0f;
    vec3 center = foot + h * vec3(bend.x * t * t, t * (1.0f - 0.25f * dot(bend, bend) * t), bend.y * t * t);
    vec3 tangent = normalize(vec3(2.0f * bend.x * t, 1.0f, 2.0f * bend.y * t));
    // At least a pixel wide.
    float pixels = treeFocalPixels(settings[0]) / max(treeViewDepth(center), 1e-4f);
    float width = max(g.blade.y / g.grid.w, 1.2f / pixels) * (1.0f - 0.8f * t);
    vec3 world = center + 0.5f * width * float(corner.y) * vec3(side.x, 0.0f, side.y);

    outWorldPos = world;
    outNormal = normalize(cross(vec3(side.x, 0.0f, side.y), tangent));
    // Darker towards the foot, where less light reaches; snow on the tips in winter.
    outAlbedo = mix(clump.albedo.rgb * (0.55f + 0.6f * t), vec3(0.8f), clump.ground.z * t);
    outSunlight = clump.kind.z;
    gl_Position = camera.proj * camera.view * vec4(world, 1.0f);
}
