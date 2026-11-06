// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

#define TREE_CAMERA_SET 0
#define TREE_CAMERA_BINDING 0
#include "tree_render_common.slang"
#include "tree_shading.slang"
#define TREE_OCCLUSION_BINDING 4
#include "tree_occlusion.slang"
#define TREE_LIGHT_SET 1
#define TREE_LIGHT_BINDING 5
#include "tree_light.slang"

[[vk::binding(3, 1)]] StructuredBuffer<TreeRenderSettings> settings;

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;
layout(location = 3) in vec3 inBark;
layout(location = 4) in vec3 inTangent;

[[vk::binding(2, 1)]] StructuredBuffer<TreeTypeParams> params;
layout(location = 5) nointerpolation in uint inType;

layout(location = 0) out vec4 outColor;

[shader("fragment")]
void main() {
    if (treeBehindScene(settings[0])) {
        discard;
    }
    TreeRenderSettings s = settings[0];
    vec3 normal = normalize(inNormal);
    // Bark: furrows along the stem darken the colour and tilt the normal; they
    // fade out where they would be finer than about two pixels.
    float scale = params[inType].trunk.z;
    vec2 uv = scale * vec2(inBark.x * inBark.z, inBark.y);
    float footprint = max(length(fwidth(uv)), 1e-6f);
    float detail = scale > 0.0f ? clamp(1.5f - footprint, 0.0f, 1.0f) : 0.0f;
    vec3 albedo = inColor;
    if (detail > 0.0f) {
        float e = 0.05f;
        float h = treeBarkHeight(uv);
        float dhdu = (treeBarkHeight(uv + vec2(e, 0.0f)) - h) / e;
        float dhdv = (treeBarkHeight(uv + vec2(0.0f, e)) - h) / e;
        vec3 tangent = normalize(inTangent);
        vec3 along = normalize(cross(normal, tangent));
        normal = normalize(normal - 0.25f * detail * (dhdu * tangent + dhdv * along));
        albedo *= mix(0.9f, 0.45f + 0.9f * h, detail);
    } else {
        albedo *= 0.9f;
    }
    // Snow on the upper side of the branches.
    albedo = mix(albedo, vec3(0.8f), treeSnowOn(normalize(inNormal), inWorldPos, s.season.y));
    // A voxel and a half out, clear of the solid wood a thick stem fills the light volume with.
    vec2 light = treeLightAt(s, inWorldPos + 1.5f * s.lightGridMin.w * normal);
    vec3 color = treeShadeLit(albedo, normal, s, light) + treePointLight(albedo, normal, inWorldPos, s);
    outColor = treeSurfaceOutput(color, albedo, normal, inWorldPos, s, TREE_CLASS_STEM);
}
