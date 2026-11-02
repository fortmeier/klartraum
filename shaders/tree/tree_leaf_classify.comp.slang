// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import glsl;

// One thread per leaf: culls leaves outside the view or thinned out by the
// level of detail and sorts the others into the full or the simplified draw
// bucket, writing the buckets' indirect draw arguments.

#define TREE_CAMERA_SET 0
#define TREE_CAMERA_BINDING 2
#include "tree_render_common.slang"
#include "tree_leaf_common.slang"

[[vk::binding(0, 0)]] StructuredBuffer<LeafRecord> leaves;
[[vk::binding(1, 0)]] StructuredBuffer<uint> counters;
[[vk::binding(3, 0)]] StructuredBuffer<TreeRenderSettings> settings;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> lists;
[[vk::binding(5, 0)]] RWStructuredBuffer<uint> drawArgs;
// [0] drawn leaves, [1] triangles, [2] culled leaves.
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> stats;
[[vk::binding(7, 0)]] StructuredBuffer<TreeTypeParams> params;

struct Push {
    uint capacity;
    uint lobes; // lobes drawn per leaf: the most of any tree type
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

bool outsideFrustum(vec3 center, float radius) {
    mat4 m = camera.proj * camera.view;
    vec4 rows[4] = vec4[4](vec4(m[0][0], m[1][0], m[2][0], m[3][0]), vec4(m[0][1], m[1][1], m[2][1], m[3][1]),
                           vec4(m[0][2], m[1][2], m[2][2], m[3][2]), vec4(m[0][3], m[1][3], m[2][3], m[3][3]));
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
        for (uint b = 0u; b < TREE_LEAF_BUCKETS; ++b) {
            drawArgs[4u * b] = pc.lobes * 2u * treeLeafHalfVertices(b);
        }
    }
    uint count = min(counters[5], pc.capacity);
    if (index >= count) {
        return;
    }
    LeafRecord leaf = leaves[index];
    TreeRenderSettings s = settings[0];
    vec2 lod = treeLeafLodOf(leaf, s);
    float leafLength = leaf.posScale.w * max(lod.y, 1.0f);
    vec3 center = leaf.posScale.xyz + 0.5f * leafLength * qforward(leaf.rot);
    if (lod.y <= 0.0f || outsideFrustum(center, leafLength)) {
        atomicAdd(stats[2], 1u);
        return;
    }
    uint bucket = lod.x >= 1.0f ? 1u : 0u;
    uint slot = atomicAdd(drawArgs[4u * bucket + 1u], 1u);
    lists[bucket * pc.capacity + slot] = index;

    uint lobes = uint(clamp(params[treeType(leaf.ids)].leafLobes.x, 1.0f, float(pc.lobes)));
    atomicAdd(stats[0], 1u);
    atomicAdd(stats[1], lobes * 2u * treeLeafHalfVertices(bucket) / 3u);
}
