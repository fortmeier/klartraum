// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#version 450

#define TREE_CAMERA_SET 0
#define TREE_CAMERA_BINDING 0
#include "tree_render_common.glsl"
#include "tree_shading.glsl"
#define TREE_OCCLUSION_BINDING 4
#include "tree_occlusion.glsl"
#define TREE_LIGHT_SET 1
#define TREE_LIGHT_BINDING 5
#include "tree_light.glsl"

layout(std430, set = 1, binding = 3) readonly buffer Settings {
    TreeRenderSettings settings[];
};

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec2 inBark;

layout(location = 0) out vec4 outColor;

void main() {
    if (treeBehindScene(settings[0])) {
        discard;
    }
    TreeRenderSettings s = settings[0];
    vec3 normal = normalize(inNormal);
    vec2 light = treeLightAt(s, inWorldPos + 0.5f * s.lightGridMin.w * normal);
    outColor = treeOutputColor(treeShadeLit(inColor, normal, s, light));
}
