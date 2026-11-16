// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;
import scalar_buffer;

// Proxy trees as Gaussians (TreeProxyGaussians): four per visible site, the
// trunk and three ellipsoids filling the crown of tree_proxy.vert's shape,
// coloured and lit like the classical proxies.

#define TREE_CAMERA_SET 0
#define TREE_CAMERA_BINDING 8
#include "tree_render_common.slang"
#include "tree_shading.slang"
#define TREE_LIGHT_SET 0
#define TREE_LIGHT_BINDING 7
#include "tree_light.slang"

[[vk::binding(0, 0)]] RWScalarBuffer<vec3> outPos;
[[vk::binding(1, 0)]] RWScalarBuffer<vec4> outRot;
[[vk::binding(2, 0)]] RWScalarBuffer<vec3> outScale;
[[vk::binding(3, 0)]] RWScalarBuffer<vec4> outColAlpha;
[[vk::binding(4, 0)]] StructuredBuffer<TreeSite> instances;
[[vk::binding(5, 0)]] StructuredBuffer<uint> counter;
[[vk::binding(6, 0)]] StructuredBuffer<TreeRenderSettings> settings;

struct Push {
    uint offset;   // of the pool in the output buffers
    uint capacity; // trees the pool holds (four slots each)
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

static const float SH_C0 = 0.28209479f;

vec4 lit(vec3 albedo, vec3 pos, vec3 normal, float alpha) {
    TreeRenderSettings s = settings[0];
    vec2 light = treeLightAt(s, pos);
    vec3 color = treeShadeLit(albedo, normal, s, light) + treePointLight(albedo, normal, pos, s);
    return vec4((treeOutputColor(color, s).rgb - vec3(0.5f)) / SH_C0, alpha);
}

void write(uint slot, vec3 pos, vec3 scale, vec4 colAlpha) {
    outPos[slot] = pos;
    outRot[slot] = vec4(0.0f, 0.0f, 0.0f, 1.0f);
    outScale[slot] = scale;
    outColAlpha[slot] = colAlpha;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= min(counter[0], pc.capacity)) {
        return;
    }
    TreeSite site = instances[i];
    TreeRenderSettings s = settings[0];
    float h = site.posHeight.w;
    vec3 foot = site.posHeight.xyz;
    vec4 shape = treeProxyShape(site.kind.x);
    vec4 crown = treeProxyCrown(site, s.season.x);
    float radius = shape.z * h;
    vec3 up = vec3(0.0f, 1.0f, 0.0f);
    vec3 albedo = mix(vec3(crown), vec3(0.8f), 0.6f * s.season.y);
    // Fading in where the canopy fades out.
    crown.w *= site.kind.w;
    float trunkAlpha = shape.w > 0.0f ? 0.95f * site.kind.w : 0.0f;
    uint base = pc.offset + 4u * i;
    float trunkTop = (shape.x + 0.05f) * h;
    write(base, foot + vec3(0.0f, 0.5f * trunkTop, 0.0f), vec3(shape.w * h, 0.5f * trunkTop, shape.w * h) * 0.7f,
          lit(vec3(0.09f, 0.07f, 0.05f), foot, normalize(s.sun.xyz), trunkAlpha));
    float below = shape.y - shape.x;
    float above = 1.0f - shape.y;
    write(base + 1u, foot + vec3(0.0f, (shape.x + 0.65f * below) * h, 0.0f),
          vec3(0.42f * radius, 0.4f * below * h, 0.42f * radius), lit(0.75f * albedo, foot, up, 0.95f * crown.w));
    write(base + 2u, foot + vec3(0.0f, (shape.y + 0.25f * above) * h, 0.0f),
          vec3(0.36f * radius, 0.28f * above * h, 0.36f * radius), lit(0.9f * albedo, foot, up, 0.95f * crown.w));
    write(base + 3u, foot + vec3(0.0f, (shape.y + 0.62f * above) * h, 0.0f),
          vec3(0.2f * radius, 0.22f * above * h, 0.2f * radius), lit(albedo, foot, up, 0.95f * crown.w));
}
