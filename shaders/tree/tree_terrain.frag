// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#version 450

// The terrain inside the soil's outline: its materials blended by the land
// cover (treeTerrainAlbedo()), snow in winter and above the snow line, lit
// like the other scene surfaces. Where the forest floor's moss is resolved,
// its cushions tilt the normal, as on the soil of the pot.

#define TREE_CAMERA_SET 0
#define TREE_CAMERA_BINDING 0
#include "tree_render_common.glsl"
#include "tree_shading.glsl"
#define TREE_OCCLUSION_BINDING 2
#include "tree_occlusion.glsl"
#define TREE_LIGHT_SET 1
#define TREE_LIGHT_BINDING 3
#include "tree_light.glsl"

layout(std430, set = 1, binding = 1) readonly buffer Settings {
    TreeRenderSettings settings[];
};
layout(std430, set = 1, binding = 5) readonly buffer Config {
    TreeTerrainConfig config;
};

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inCover;

layout(location = 0) out vec4 outColor;

void main() {
    TreeRenderSettings s = settings[0];
    TreeTerrainParams t = config.terrain;
    if (treeTerrainRadius(t, inWorldPos.x, inWorldPos.z) >= 1.0f || treeBehindScene(s)) {
        discard;
    }
    TreeLandCover cover;
    cover.vegetation = clamp(inCover, vec4(0.0f), vec4(1.0f));
    cover.bare = vec4(max(1.0f - dot(cover.vegetation, vec4(1.0f)), 0.0f), 0.0f, 0.0f, 0.0f);
    float footprint = length(fwidth(inWorldPos));
    vec3 albedo = treeTerrainAlbedo(t, cover, inWorldPos, footprint, s.season.x);
    vec3 surface = normalize(inNormal);
    vec3 normal = surface;
    vec3 pd = inWorldPos * t.soil.w;
    float detail = clamp(1.5f - 140.0f * footprint * t.soil.w, 0.0f, 1.0f) * cover.vegetation.x;
    if (detail > 0.0f) {
        // The relief of the moss cushions tilts the normal.
        float e = 0.002f;
        float h = treeMossHeight(pd);
        float dx = (treeMossHeight(pd + vec3(e, 0.0f, 0.0f)) - h) / e;
        float dz = (treeMossHeight(pd + vec3(0.0f, 0.0f, e)) - h) / e;
        normal = normalize(normal - detail * vec3(dx, 0.0f, dz));
    }
    albedo = mix(albedo, vec3(0.8f), treeTerrainSnow(t, inWorldPos, surface, s.season.x, s.season.y));
    vec2 light = treeLightAt(s, inWorldPos + 0.5f * s.lightGridMin.w * normal);
    vec3 color = treeShadeLit(albedo, normal, s, light) + treePointLight(albedo, normal, inWorldPos, s);
    outColor = treeOutputColor(color, s);
}
