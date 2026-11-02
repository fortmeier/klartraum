// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// Blades of grass, lit from both sides like the other surfaces: the normal
// faces the camera and leans up, as light scatters through the blades; far
// blades take the ground's normal, so they are lit like the terrain beneath.

#define TREE_CAMERA_SET 0
#define TREE_CAMERA_BINDING 0
#include "tree_render_common.slang"
#include "tree_shading.slang"
#define TREE_OCCLUSION_BINDING 2
#include "tree_occlusion.slang"
#define TREE_LIGHT_SET 1
#define TREE_LIGHT_BINDING 3
#include "tree_light.slang"

[[vk::binding(1, 1)]] StructuredBuffer<TreeRenderSettings> settings;

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inAlbedo;
layout(location = 3) in float inSunlight;
layout(location = 4) in vec4 inGround;

layout(location = 0) out vec4 outColor;

[shader("fragment")]
void main() {
    TreeRenderSettings s = settings[0];
    if (treeBehindScene(s)) {
        discard;
    }
    vec3 normal = normalize(inNormal);
    if (dot(normal, camera.cameraWorldPos.xyz - inWorldPos) < 0.0f) {
        normal = -normal;
    }
    normal = normalize(mix(normalize(normal + vec3(0.0f, 1.2f, 0.0f)), inGround.xyz, inGround.w));
    vec2 light = treeLightAt(s, inWorldPos + 0.5f * s.lightGridMin.w * normal);
    light.x *= clamp(inSunlight, 0.0f, 1.0f);
    vec3 color = treeShadeLit(inAlbedo, normal, s, light) + treePointLight(inAlbedo, normal, inWorldPos, s);
    outColor = treeOutputColor(treeAtmosphere(color, inWorldPos, camera.cameraWorldPos.xyz, s), s);
}
