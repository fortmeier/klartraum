// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// One thread per stem segment: culls segments outside the view or thinner
// than settings.viewport.w pixels and sorts the others into the draw bucket
// that fits their tessellation factors, writing the bucket's indirect draw
// arguments.

#define TREE_CAMERA_SET 0
#define TREE_CAMERA_BINDING 2
#include "tree_render_common.slang"

[[vk::binding(0, 0)]] StructuredBuffer<SegmentRecord> segments;
[[vk::binding(1, 0)]] StructuredBuffer<uint> counters;
[[vk::binding(3, 0)]] StructuredBuffer<TreeRenderSettings> settings;
// Bucket b's segment indices start at b * pc.capacity.
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> lists;
// One VkDrawIndirectCommand per bucket; zeroed before the pass.
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> drawArgs;
// [0] drawn segments, [1] triangles of the tessellation, [2] culled segments.
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> stats;

struct Push {
    uint capacity;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

bool outsideFrustum(vec3 center, float radius) {
    mat4 m = camera.proj * camera.view;
    vec4 rows[4] = vec4[4](vec4(m[0][0], m[1][0], m[2][0], m[3][0]), vec4(m[0][1], m[1][1], m[2][1], m[3][1]),
                           vec4(m[0][2], m[1][2], m[2][2], m[3][2]), vec4(m[0][3], m[1][3], m[2][3], m[3][3]));
    // Vulkan clip space: -w <= x, y <= w and 0 <= z <= w.
    vec4 planes[5] = vec4[5](rows[3] + rows[0], rows[3] - rows[0], rows[3] + rows[1], rows[3] - rows[1], rows[2]);
    for (int i = 0; i < 5; ++i) {
        vec4 p = planes[i] / length(planes[i].xyz);
        if (dot(p.xyz, center) + p.w < -radius) {
            return true;
        }
    }
    return false;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main() {
    uint index = gl_GlobalInvocationID.x;
    if (index == 0u) {
        for (uint b = 0u; b < TREE_STEM_BUCKETS; ++b) {
            drawArgs[4u * b] = 6u * treeStemBucketAround(b) * treeStemBucketAlong(b);
        }
    }
    uint count = min(counters[4], pc.capacity);
    if (index >= count) {
        return;
    }
    SegmentRecord d = segments[index];
    TreeRenderSettings s = settings[0];
    StemRing a = stemFrom(d);
    StemRing b = stemTo(d);

    float segmentLength = length(b.pos - a.pos);
    vec3 center = 0.5f * (a.pos + b.pos);
    // The spline bulges at most a quarter of the length beyond the chord.
    float bound = 0.75f * segmentLength + max(a.radius, b.radius);
    bool thin = max(treeRingPixelRadius(a, s), treeRingPixelRadius(b, s)) < s.viewport.w;
    if (thin || outsideFrustum(center, bound)) {
        atomicAdd(stats[2], 1u);
        return;
    }

    float f = treeRingFactor(a, s);
    float t = treeRingFactor(b, s);
    float v = treeSegmentAlongFactor(a, b, s);
    uint bucket = treeStemBucket(max(f, t), v);
    uint slot = atomicAdd(drawArgs[4u * bucket + 1u], 1u);
    lists[bucket * pc.capacity + slot] = index;

    float around = float(treeStemBucketAround(bucket));
    float along = float(treeStemBucketAlong(bucket));
    float u = 0.5f * (clamp(f, 1.0f, around) + clamp(t, 1.0f, around));
    atomicAdd(stats[0], 1u);
    atomicAdd(stats[1], 2u * uint(ceil(u)) * uint(ceil(clamp(v, 1.0f, along))));
}
