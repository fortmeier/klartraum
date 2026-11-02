// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// First pass of the light volume: adds the area of every leaf and stem
// segment to the voxel(s) it lies in, as fixed point (1e-7 units^2 per step).
// One thread per segment slot, then one per leaf slot.

#define TREE_FN
#include "klartraum/tree/tree_shared.hpp"

[[vk::binding(0, 0)]] StructuredBuffer<SegmentRecord> segments;
[[vk::binding(1, 0)]] StructuredBuffer<LeafRecord> leaves;
[[vk::binding(2, 0)]] StructuredBuffer<uint> counters;
[[vk::binding(3, 0)]] StructuredBuffer<TreeTypeParams> params;
[[vk::binding(4, 0)]] StructuredBuffer<TreeRenderSettings> settings;
// Zeroed before the pass.
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> density;

struct Push {
    uint segmentCapacity;
    uint leafCapacity;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

static const float AREA_SCALE = 1e7f;

void addArea(vec3 p, float area) {
    TreeRenderSettings s = settings[0];
    ivec3 c = ivec3(floor((p - s.lightGridMin.xyz) / s.lightGridMin.w));
    if (any(lessThan(c, ivec3(0))) || any(greaterThanEqual(c, ivec3(s.lightGridSize.xyz)))) {
        return;
    }
    atomicAdd(density[treeLightIndex(c.x, c.y, c.z, s.lightGridSize)], uint(area * AREA_SCALE + 0.5f));
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main() {
    uint i = gl_GlobalInvocationID.x;
    float voxel = settings[0].lightGridMin.w;
    if (i < pc.segmentCapacity) {
        if (i >= min(counters[4], pc.segmentCapacity)) {
            return;
        }
        SegmentRecord d = segments[i];
        vec3 a = d.fromPosRadius.xyz;
        vec3 b = d.toPosRadius.xyz;
        float segmentLength = length(b - a);
        // Samples half a voxel apart, each with its share of the projected area 2 r l.
        uint samples = clamp(uint(ceil(2.0f * segmentLength / voxel)), 1u, 64u);
        for (uint k = 0u; k < samples; ++k) {
            float t = (float(k) + 0.5f) / float(samples);
            vec3 p = treeHermite(a, b, qforward(d.fromRot) * segmentLength, qforward(d.toRot) * segmentLength, t);
            float r = mix(d.fromPosRadius.w, d.toPosRadius.w, t);
            addArea(p, 2.0f * r * segmentLength / float(samples));
        }
        return;
    }
    uint j = i - pc.segmentCapacity;
    if (j >= min(counters[5], pc.leafCapacity)) {
        return;
    }
    LeafRecord leaf = leaves[j];
    TreeTypeParams p = params[treeType(leaf.ids)];
    TreeLeafOutline o = treeLeafOutline(p.leafShape, 0.5f * p.leafCount.z);
    float lobes = clamp(p.leafLobes.x, 1.0f, 7.0f);
    float area = 2.0f * treeLeafHalfArea(o, 0.0f) * lobes * leaf.posScale.w * leaf.posScale.w;
    addArea(leaf.posScale.xyz + 0.5f * leaf.posScale.w * qforward(leaf.rot), area);
}
