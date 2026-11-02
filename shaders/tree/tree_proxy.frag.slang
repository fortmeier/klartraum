// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// Proxy trees, lit like the other surfaces, snow on what faces up in winter.
// Sparse crowns (bare broadleaves) keep a share of their pixels, chosen by a
// hash of the pixel.

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
layout(location = 3) in float inCoverage;
layout(location = 4) in float inSunlight;

layout(location = 0) out vec4 outColor;

[shader("fragment")]
void main() {
    TreeRenderSettings s = settings[0];
    if (treeBehindScene(s) ||
        treeRand(uint(gl_FragCoord.x) * 7919u + uint(gl_FragCoord.y), 0x17u) >= inCoverage) {
        discard;
    }
    vec3 normal = normalize(inNormal);
    vec3 albedo = mix(inAlbedo, vec3(0.8f), 0.7f * s.season.y * smoothstep(0.2f, 0.7f, normal.y));
    vec2 light = treeLightAt(s, inWorldPos + 0.5f * s.lightGridMin.w * normal);
    light.x *= inSunlight;
    vec3 color = treeShadeLit(albedo, normal, s, light) + treePointLight(albedo, normal, inWorldPos, s);
    outColor = treeOutputColor(treeAtmosphere(color, inWorldPos, camera.cameraWorldPos.xyz, s), s);
}
