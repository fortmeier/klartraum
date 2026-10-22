// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#version 450

// An impostor per visible tree (TreeImpostorRenderer): a card through the
// tree's centre facing the camera, as large as the baked tree's bounds. The
// direction to the camera in the tree's frame (its turn about +y undone)
// picks the four nearest views of the atlas and their bilinear weights; the
// season picks the two nearest season samples.

#define TREE_CAMERA_SET 0
#define TREE_CAMERA_BINDING 0
#include "tree_render_common.glsl"

layout(std430, set = 1, binding = 0) readonly buffer Instances {
    TreeSite instances[];
};
layout(std430, set = 1, binding = 1) readonly buffer Settings {
    TreeRenderSettings settings[];
};
layout(std430, set = 1, binding = 6) readonly buffer Params {
    TreeImpostorParams params;
};

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) out vec2 outCard;
layout(location = 2) flat out uvec4 outViews;
layout(location = 3) flat out vec4 outWeights;
layout(location = 4) flat out vec3 outRight;
layout(location = 5) flat out vec3 outUp;
layout(location = 6) flat out vec3 outToward;
layout(location = 7) flat out uvec2 outLayers;
layout(location = 8) flat out vec4 outBlend; // x weight of the second season layer, y fade, z tint, w sunlight

const vec2 corners[6] = vec2[6](vec2(-1.0f, -1.0f), vec2(1.0f, -1.0f), vec2(1.0f, 1.0f), vec2(-1.0f, -1.0f),
                                vec2(1.0f, 1.0f), vec2(-1.0f, 1.0f));

void main() {
    TreeSite site = instances[gl_InstanceIndex];
    uint type = min(site.kind.x == TREE_SITE_CONIFER ? 1u : 0u, params.grid.w - 1u);
    vec4 bounds = params.bounds[type];
    float scale = site.posHeight.w / params.heights[type];
    vec4 turn = qaxis(vec3(0.0f, 1.0f, 0.0f), site.kind.y);
    vec4 back = qaxis(vec3(0.0f, 1.0f, 0.0f), -site.kind.y);
    vec3 center = site.posHeight.xyz + qrot(turn, scale * bounds.xyz);
    float radius = scale * bounds.w;
    vec3 toward = normalize(camera.cameraWorldPos.xyz - center);
    vec3 local = qrot(back, toward);

    // The four nearest views.
    int views = int(params.grid.x);
    vec2 g = (0.5f * treeHemiOctEncode(local) + vec2(0.5f)) * float(views) - vec2(0.5f);
    ivec2 g0 = clamp(ivec2(floor(g)), ivec2(0), ivec2(views - 2));
    vec2 f = clamp(g - vec2(g0), vec2(0.0f), vec2(1.0f));
    uint v0 = uint(g0.y * views + g0.x);
    outViews = uvec4(v0, v0 + 1u, v0 + uint(views), v0 + uint(views) + 1u);
    outWeights = vec4((1.0f - f.x) * (1.0f - f.y), f.x * (1.0f - f.y), (1.0f - f.x) * f.y, f.x * f.y);

    // The two nearest season samples, around the year.
    float season = treeSeasonWrap(settings[0].season.x);
    uint count = params.grid.z;
    uint hi = 0u;
    while (hi < count && params.seasons[hi] <= season) {
        ++hi;
    }
    uint lo = (hi + count - 1u) % count;
    hi = hi % count;
    float from = params.seasons[lo];
    float to = params.seasons[hi];
    float span = treeSeasonWrap(to - from);
    float t = span > 0.0f ? clamp(treeSeasonWrap(season - from) / span, 0.0f, 1.0f) : 0.0f;
    outLayers = uvec2(type * count + lo, type * count + hi);

    outRight = qrot(turn, treeImpostorRight(local));
    outUp = qrot(turn, treeImpostorUp(local));
    outToward = toward;
    vec2 c = corners[gl_VertexIndex];
    vec3 world = center + radius * (c.x * outRight + c.y * outUp);
    outWorldPos = world;
    outCard = c;
    outBlend = vec4(t, site.kind.w, 0.9f + 0.2f * treeRand(floatBitsToUint(site.kind.y), 0x55u), site.kind.z);
    gl_Position = camera.proj * camera.view * vec4(world, 1.0f);
}
