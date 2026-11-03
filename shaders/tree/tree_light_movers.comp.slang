// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// Adds the moving occluders of the light volume (TreeMovingOccluders) at their current transforms: the area of every
// surface sample to the voxel it lies in, as fixed point (1e-7 units^2 per step), in a grid of its own that the march
// counts like the static occluders. One thread per sample.

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
    ivec3 c = ivec3(floor((p - s.lightGridMin.xyz) / s.lightGridMin.w));
    if (any(lessThan(c, ivec3(0))) || any(greaterThanEqual(c, ivec3(s.lightGridSize.xyz)))) {
        return;
    }
    atomicAdd(moving[treeLightIndex(c.x, c.y, c.z, s.lightGridSize)], uint(area * AREA_SCALE + 0.5f));
}
