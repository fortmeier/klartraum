// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#version 450

// The terrain inside the soil's outline, shaded like the other scene surfaces.

#define TREE_CAMERA_SET 0
#define TREE_CAMERA_BINDING 0
#include "tree_render_common.glsl"
#include "tree_shading.glsl"
#define TREE_OCCLUSION_BINDING 2
#include "tree_occlusion.glsl"
#define TREE_LIGHT_SET 1
#define TREE_LIGHT_BINDING 3
#include "tree_light.glsl"
#include "tree_scene_shading.glsl"

layout(std430, set = 1, binding = 1) readonly buffer Settings {
    TreeRenderSettings settings[];
};
layout(std430, set = 1, binding = 5) readonly buffer Config {
    TreeTerrainConfig config;
};

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inColor;

layout(location = 0) out vec4 outColor;

void main() {
    TreeRenderSettings s = settings[0];
    if (treeTerrainRadius(config.terrain, inWorldPos.x, inWorldPos.z) >= 1.0f || treeBehindScene(s)) {
        discard;
    }
    outColor = treeSceneColor(inWorldPos, inNormal, inColor, s);
}
