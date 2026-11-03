// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// Adds the moving occluders of the light volume (TreeMovingOccluders) at their current transforms: the area of every
// surface sample, shared with a tent of SPREAD voxels to each side among the voxels around it, so that the occluder's
// shadow has a soft edge and glides as it moves, as fixed point (1e-7 units^2 per step), in a grid of its own that the march counts like the static
// occluders. One thread per sample.

#define TREE_FN
#include "klartraum/tree/tree_shared.hpp"

[[vk::binding(0, 0)]] StructuredBuffer<vec4> samples; // xyz in the object's frame, w area
[[vk::binding(1, 0)]] StructuredBuffer<uint> objectOf;
[[vk::binding(2, 0)]] StructuredBuffer<mat4> worldFromObject;
[[vk::binding(3, 0)]] StructuredBuffer<TreeRenderSettings> settings;
// Zeroed before the pass.
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> moving;

struct Push {
    uint count;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

static const float AREA_SCALE = 1e7f;
// How far a sample's area spreads, in voxels to each side.
static const float SPREAD = 1.5f;
// The voxels a sample's tent reaches along each axis, from floor(g) - 1 on.
static const int REACH = 4;

[shader("compute")]
[numthreads(256, 1, 1)]
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.count) {
        return;
    }
    mat4 m = worldFromObject[objectOf[i]];
    // The area grows with the square of the transform's scale; a scale of 0 removes the object.
    float scale = length(vec3(m[0].x, m[0].y, m[0].z));
    float area = samples[i].w * scale * scale;
    if (area <= 0.0f) {
        return;
    }
    vec3 p = (m * vec4(samples[i].xyz, 1.0f)).xyz;
    TreeRenderSettings s = settings[0];
    ivec3 size = ivec3(s.lightGridSize.xyz);
    vec3 g = (p - s.lightGridMin.xyz) / s.lightGridMin.w - 0.5f;
    ivec3 i0 = ivec3(floor(g)) - 1;
    // The tent's weights along each axis for the REACH voxels from i0 on, normalized.
    float wx[REACH], wy[REACH], wz[REACH];
    float sx = 0.0f, sy = 0.0f, sz = 0.0f;
    for (int k = 0; k < REACH; ++k) {
        vec3 d = abs(vec3(i0 + k) - g);
        wx[k] = max(0.0f, 1.0f - d.x / SPREAD);
        wy[k] = max(0.0f, 1.0f - d.y / SPREAD);
        wz[k] = max(0.0f, 1.0f - d.z / SPREAD);
        sx += wx[k];
        sy += wy[k];
        sz += wz[k];
    }
    for (int k = 0; k < REACH * REACH * REACH; ++k) {
        ivec3 o = ivec3(k % REACH, (k / REACH) % REACH, k / (REACH * REACH));
        ivec3 c = i0 + o;
        float w = wx[o.x] * wy[o.y] * wz[o.z] / (sx * sy * sz);
        if (w <= 0.0f || any(lessThan(c, ivec3(0))) || any(greaterThanEqual(c, size))) {
            continue;
        }
        atomicAdd(moving[treeLightIndex(c.x, c.y, c.z, s.lightGridSize)], uint(w * area * AREA_SCALE + 0.5f));
    }
}
