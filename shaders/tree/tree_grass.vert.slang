// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// The blades of the visible clumps of grass (TreeGrass), one instance per
// clump of grid.z x 15 vertices: a blade (treeGrassBlade()) of three
// segments narrowing to its tip (two triangles each, one for the tip).
// Blades beyond the clump's count (kind.x) collapse outside the view. Far
// blades widen to a pixel.

#define TREE_CAMERA_SET 0
#define TREE_CAMERA_BINDING 0
#include "tree_render_common.slang"
#include "tree_grass_common.slang"

[[vk::binding(0, 1)]] StructuredBuffer<TreeGrassVisible> visibleList;
[[vk::binding(1, 1)]] StructuredBuffer<TreeRenderSettings> settings;
[[vk::binding(4, 1)]] StructuredBuffer<TreeGrassSetup> setupBuffer;
// [0] the wind: x time in seconds, y its strength in [0, 1], zw the direction it blows to (x, z).
[[vk::binding(5, 1)]] StructuredBuffer<vec4> frame;

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec3 outAlbedo;
layout(location = 3) out float outSunlight;
layout(location = 4) out vec4 outGround; // xyz the ground's normal, w how far the terrain takes over (0 near)

// Per vertex of a blade: x the level (0 the foot, 3 the tip), y the side (-1, 1, 0 at the tip).
static const ivec2 corners[15] = ivec2[15](ivec2(0, -1), ivec2(0, 1), ivec2(1, 1), ivec2(0, -1), ivec2(1, 1), ivec2(1, -1),
                                    ivec2(1, -1), ivec2(1, 1), ivec2(2, 1), ivec2(1, -1), ivec2(2, 1), ivec2(2, -1),
                                    ivec2(2, -1), ivec2(2, 1), ivec2(3, 0));

[shader("vertex")]
void main() {
    TreeGrassVisible clump = visibleList[gl_InstanceIndex];
    TreeGrassParams g = setupBuffer[0].grass;
    vec4 wind = frame[0];
    uint blade = uint(gl_VertexIndex) / 15u;
    if (float(blade) >= clump.kind.x) {
        gl_Position = vec4(2.0f, 2.0f, 2.0f, 1.0f);
        return;
    }
    ivec2 corner = corners[gl_VertexIndex % 15];
    GrassBlade b = treeGrassBlade(clump, g, blade, wind);
    vec2 side = vec2(-b.facing.y, b.facing.x);
    float t = float(corner.x) / 3.0f;
    vec3 center = treeGrassBladePoint(b, t);
    vec3 tangent = treeGrassBladeTangent(b, t);
    // At least a pixel wide.
    float pixels = treeFocalPixels(settings[0]) / max(treeViewDepth(center), 1e-4f);
    float width = max(g.blade.y / g.grid.w, 1.2f / pixels) * (1.0f - 0.8f * t);
    vec3 world = center + 0.5f * width * float(corner.y) * vec3(side.x, 0.0f, side.y);

    outWorldPos = world;
    outNormal = normalize(cross(vec3(side.x, 0.0f, side.y), tangent));
    // Darker towards the foot, where less light reaches, up close; towards the radius as bright as the terrain
    // beneath, which takes over. Snow on the tips in winter.
    float far = smoothstep(0.33f * g.grid.y, g.grid.y, clump.ground.w);
    outAlbedo = mix(clump.albedo.rgb * mix(0.55f + 0.6f * t, 1.0f, far), vec3(0.8f), clump.ground.z * t);
    outSunlight = clump.kind.z;
    outGround = vec4(normalize(vec3(-clump.ground.x, 1.0f, -clump.ground.y)), far);
    gl_Position = camera.proj * camera.view * vec4(world, 1.0f);
}
