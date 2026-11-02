// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// Second pass of the light volume: for every voxel, the transmittance towards
// the sun and the visibility of the sky (the mean transmittance towards the
// zenith and four directions 50 degrees above the horizon), through the leaf
// and stem area the first pass collected. A voxel's optical depth is
// lightGridParams.x times its area over the voxel's face area, plus the
// optical depth of static occluders (pot, lantern), computed once on the CPU
// (treeStaticOcclusion()). A ray starts in the middle of its voxel, so leaves
// are half shadowed by their own voxel; static occluders in it are skipped,
// so solid surfaces do not shadow themselves.

#define TREE_FN
#include "klartraum/tree/tree_shared.hpp"

[[vk::binding(0, 0)]] StructuredBuffer<TreeRenderSettings> settings;
[[vk::binding(1, 0)]] StructuredBuffer<uint> density;
[[vk::binding(2, 0)]] RWStructuredBuffer<vec4> light; // xy through all occluders, zw through the trees only
[[vk::binding(3, 0)]] StructuredBuffer<float> staticDepth;

static const float AREA_SCALE = 1e7f;

float treeOpticalDepth(ivec3 c, TreeRenderSettings s) {
    float area = float(density[treeLightIndex(c.x, c.y, c.z, s.lightGridSize)]) / AREA_SCALE;
    return s.lightGridParams.x * area / (s.lightGridMin.w * s.lightGridMin.w);
}

float opticalDepth(ivec3 c, TreeRenderSettings s) {
    return treeOpticalDepth(c, s) + staticDepth[treeLightIndex(c.x, c.y, c.z, s.lightGridSize)];
}

// Transmittance from the centre of voxel c along direction d to the grid's
// border: x through all occluders, y through the trees only.
vec2 march(ivec3 c, vec3 d, TreeRenderSettings s) {
    ivec3 size = ivec3(s.lightGridSize.xyz);
    float trees = 0.5f * treeOpticalDepth(c, s);
    float statics = 0.0f;
    vec3 p = vec3(c) + 0.5f;
    for (int k = 0; k < 512; ++k) {
        p += d;
        ivec3 q = ivec3(floor(p));
        if (any(lessThan(q, ivec3(0))) || any(greaterThanEqual(q, size))) {
            break;
        }
        trees += treeOpticalDepth(q, s);
        statics += staticDepth[treeLightIndex(q.x, q.y, q.z, s.lightGridSize)];
    }
    return vec2(exp(-trees - statics), exp(-trees));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main() {
    TreeRenderSettings s = settings[0];
    uvec4 n = s.lightGridSize;
    uint i = gl_GlobalInvocationID.x;
    if (i >= n.x * n.y * n.z) {
        return;
    }
    ivec3 c = ivec3(i % n.x, (i / n.x) % n.y, i / (n.x * n.y));
    vec2 sun = march(c, normalize(s.sun.xyz), s);
    float e = radians(50.0f);
    vec2 sky = march(c, vec3(0.0f, 1.0f, 0.0f), s);
    sky += march(c, vec3(cos(e), sin(e), 0.0f), s);
    sky += march(c, vec3(-cos(e), sin(e), 0.0f), s);
    sky += march(c, vec3(0.0f, sin(e), cos(e)), s);
    sky += march(c, vec3(0.0f, sin(e), -cos(e)), s);
    light[i] = vec4(sun.x, sky.x / 5.0f, sun.y, sky.y / 5.0f);
}
