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
layout(location = 3) in vec2 inCurveUv;
layout(location = 4) in float inCurveSide;
layout(location = 5) in vec2 inLeafPos;

layout(location = 0) out vec4 outColor;

void main() {
    // Loop-Blinn: keep the part of a curve triangle inside the leaf outline.
    if (inCurveSide != 0.0f && inCurveSide * treeLoopBlinn(inCurveUv) > 0.0f) {
        discard;
    }
    TreeRenderSettings s = settings[0];
    if (treeBehindScene(s)) {
        discard;
    }
    vec3 normal = normalize(inNormal);
    vec3 toCamera = normalize(camera.cameraWorldPos.xyz - inWorldPos);
    bool underside = dot(normal, toCamera) < 0.0f;
    if (underside) {
        normal = -normal;
    }
    // The midrib, a little lighter.
    float midrib = 1.0f - smoothstep(0.004f, 0.012f, abs(inLeafPos.x));
    vec3 albedo = (underside ? treeLeafUnderside(inColor) : inColor) * (1.0f + 0.35f * midrib);
    // Snow on needles and evergreen leaves that face upwards.
    albedo = mix(albedo, vec3(0.8f), 0.7f * treeSnowOn(normal, inWorldPos, s.season.y));
    vec2 light = treeLightAt(s, inWorldPos);
    vec3 color = treeShadeLit(albedo, normal, s, light) + treePointLight(albedo, normal, inWorldPos, s);
    // Light shining through the leaf from behind.
    color += 0.4f * albedo * max(dot(-normal, s.sun.xyz), 0.0f) * s.sun.w * light.x * s.lightColor.rgb;
    // The waxy top's faint highlight.
    if (!underside) {
        vec3 halfway = normalize(toCamera + s.sun.xyz);
        color += vec3(0.05f) * pow(max(dot(normal, halfway), 0.0f), 40.0f) * s.sun.w * light.x * s.lightColor.rgb;
    }
    outColor = treeOutputColor(color, s);
}
