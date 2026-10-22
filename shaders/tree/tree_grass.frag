// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#version 450

// Blades of grass, lit from both sides like the other surfaces: the normal
// faces the camera and leans up, as light scatters through the blades.

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
layout(location = 2) in vec3 inAlbedo;
layout(location = 3) in float inSunlight;

layout(location = 0) out vec4 outColor;

void main() {
    TreeRenderSettings s = settings[0];
    if (treeBehindScene(s)) {
        discard;
    }
    vec3 normal = normalize(inNormal);
    if (dot(normal, camera.cameraWorldPos.xyz - inWorldPos) < 0.0f) {
        normal = -normal;
    }
    normal = normalize(normal + vec3(0.0f, 1.2f, 0.0f));
    vec2 light = treeLightAt(s, inWorldPos + 0.5f * s.lightGridMin.w * normal);
    light.x *= clamp(inSunlight, 0.0f, 1.0f);
    vec3 color = treeShadeLit(inAlbedo, normal, s, light) + treePointLight(inAlbedo, normal, inWorldPos, s);
    outColor = treeOutputColor(treeAtmosphere(color, inWorldPos, camera.cameraWorldPos.xyz, s), s);
}
