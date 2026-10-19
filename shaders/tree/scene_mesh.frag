// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#version 450

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

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inColor;

layout(location = 0) out vec4 outColor;

void main() {
    TreeRenderSettings s = settings[0];
    if (treeBehindScene(s)) {
        discard;
    }
    vec3 normal = normalize(inNormal);
    // Fine structure fades out where it would be finer than about two pixels.
    float detail = clamp(1.5f - 140.0f * length(fwidth(inWorldPos)), 0.0f, 1.0f);
    vec3 albedo = treeSceneAlbedoDetail(inColor, inWorldPos, detail);
    if (inColor.w == TREE_MATERIAL_MOSS && detail > 0.0f) {
        // The relief of the moss cushions tilts the normal.
        float e = 0.002f;
        float h = treeMossHeight(inWorldPos);
        float dx = (treeMossHeight(inWorldPos + vec3(e, 0.0f, 0.0f)) - h) / e;
        float dz = (treeMossHeight(inWorldPos + vec3(0.0f, 0.0f, e)) - h) / e;
        normal = normalize(normal - detail * vec3(dx, 0.0f, dz));
    }
    vec2 light = treeLightAt(s, inWorldPos + 0.5f * s.lightGridMin.w * normal);
    vec3 color = treeShadeLit(albedo, normal, s, light);
    if (inColor.w == TREE_MATERIAL_GLAZE) {
        // The glaze's highlight.
        vec3 toCamera = normalize(camera.cameraWorldPos.xyz - inWorldPos);
        vec3 halfway = normalize(toCamera + s.sun.xyz);
        color += vec3(0.35f) * pow(max(dot(normal, halfway), 0.0f), 60.0f) * s.sun.w * light.x;
    }
    outColor = treeOutputColor(color);
}
