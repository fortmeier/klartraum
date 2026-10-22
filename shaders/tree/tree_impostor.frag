// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#version 450

// The impostor's texels: bilinear in each of the four nearest views and the two
// nearest season samples, blended by their weights and coverage; coverage and
// the tree's fade decide by screen-door dithering which pixels show. Relit
// with the baked normals (in each view's frame: right, up, towards the viewer)
// like the trees' leaves, light through the crown from behind included.

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
layout(std430, set = 1, binding = 4) readonly buffer Albedo {
    uint albedo[];
};
layout(std430, set = 1, binding = 5) readonly buffer Normals {
    uint normals[];
};
layout(std430, set = 1, binding = 6) readonly buffer Params {
    TreeImpostorParams params;
};

layout(location = 0) in vec3 inWorldPos;
layout(location = 1) in vec2 inCard;
layout(location = 2) flat in uvec4 inViews;
layout(location = 3) flat in vec4 inWeights;
layout(location = 4) flat in vec3 inRight;
layout(location = 5) flat in vec3 inUp;
layout(location = 6) flat in vec3 inToward;
layout(location = 7) flat in uvec2 inLayers;
layout(location = 8) flat in vec3 inBlend;

layout(location = 0) out vec4 outColor;

vec4 unpack(uint v) {
    return vec4(float(v & 255u), float((v >> 8u) & 255u), float((v >> 16u) & 255u), float(v >> 24u)) / 255.0f;
}

// Adds the view's texels at the card position, weighted, to the sums: rgb albedo times coverage, a coverage,
// and the normal times coverage.
void addView(uint layer, uint view, float weight, inout vec4 sum, inout vec3 normal) {
    if (weight <= 0.0f) {
        return;
    }
    uint views = params.grid.x;
    int pixels = int(params.grid.y);
    uint side = views * uint(pixels);
    uint base = layer * side * side + (view / views) * uint(pixels) * side + (view % views) * uint(pixels);
    vec2 t = vec2(0.5f * inCard.x + 0.5f, 0.5f - 0.5f * inCard.y) * float(pixels) - vec2(0.5f);
    ivec2 t0 = ivec2(floor(t));
    vec2 f = t - vec2(t0);
    for (int k = 0; k < 4; ++k) {
        ivec2 p = clamp(t0 + ivec2(k % 2, k / 2), ivec2(0), ivec2(pixels - 1));
        float w = weight * (k % 2 == 0 ? 1.0f - f.x : f.x) * (k / 2 == 0 ? 1.0f - f.y : f.y);
        uint index = base + uint(p.y) * side + uint(p.x);
        vec4 a = unpack(albedo[index]);
        vec3 n = 2.0f * unpack(normals[index]).xyz - vec3(1.0f);
        sum += w * vec4(a.rgb * a.rgb * a.a, a.a);
        normal += w * a.a * n;
    }
}

void main() {
    TreeRenderSettings s = settings[0];
    if (treeBehindScene(s)) {
        discard;
    }
    vec4 sum = vec4(0.0f);
    vec3 n = vec3(0.0f);
    for (int k = 0; k < 4; ++k) {
        addView(inLayers.x, inViews[k], (1.0f - inBlend.x) * inWeights[k], sum, n);
        addView(inLayers.y, inViews[k], inBlend.x * inWeights[k], sum, n);
    }
    float coverage = sum.a * inBlend.y;
    if (coverage <= 0.0f || treeRand(uint(gl_FragCoord.x) * 7919u + uint(gl_FragCoord.y), 0x17u) >= coverage) {
        discard;
    }
    vec3 albedoColor = inBlend.z * sum.rgb / max(sum.a, 1e-4f);
    vec3 local = length(n) > 1e-4f ? normalize(n) : vec3(0.0f, 0.0f, 1.0f);
    vec3 normal = normalize(local.x * inRight + local.y * inUp + local.z * inToward);
    vec2 light = treeLightAt(s, inWorldPos + 0.5f * s.lightGridMin.w * normal);
    vec3 color = treeShadeLit(albedoColor, normal, s, light) + treePointLight(albedoColor, normal, inWorldPos, s);
    color += 0.25f * albedoColor * max(dot(-normal, s.sun.xyz), 0.0f) * s.sun.w * light.x * s.lightColor.rgb;
    outColor = treeOutputColor(color, s);
}
