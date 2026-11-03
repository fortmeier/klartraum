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
// The optical depth per voxel inside a thick stem.
static const float SOLID_DEPTH = 2.0f;

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
        // Samples half a voxel apart, each with its share of the projected area 2 r l on the stem's axis. A stem
        // thicker than a voxel is solid wood across its width instead: its cross-section, sampled half a voxel apart,
        // gets nearly opaque depth (SOLID_DEPTH per voxel), so that a trunk shadows as wide as it is; the two blend
        // from half a voxel to one and a half in radius.
        uint samples = clamp(uint(ceil(2.0f * segmentLength / voxel)), 1u, 64u);
        TreeRenderSettings s = settings[0];
        // The area that gives a voxel SOLID_DEPTH from the 2 x 4 cross-section samples it holds per axial voxel.
        float solidArea = SOLID_DEPTH * voxel * voxel / max(s.lightGridParams.x, 1e-6f) / 8.0f;
        for (uint k = 0u; k < samples; ++k) {
            float t = (float(k) + 0.5f) / float(samples);
            vec3 tangent = treeHermiteTangent(a, b, qforward(d.fromRot) * segmentLength,
                                              qforward(d.toRot) * segmentLength, t);
            vec3 p = treeHermite(a, b, qforward(d.fromRot) * segmentLength, qforward(d.toRot) * segmentLength, t);
            float r = mix(d.fromPosRadius.w, d.toPosRadius.w, t);
            float solid = smoothstep(0.5f, 1.5f, r / voxel);
            if (solid < 1.0f) {
                addArea(p, (1.0f - solid) * 2.0f * r * segmentLength / float(samples));
            }
            if (solid > 0.0f) {
                vec3 axis = normalize(tangent);
                vec3 u = normalize(cross(axis, abs(axis.y) < 0.9f ? vec3(0.0f, 1.0f, 0.0f) : vec3(1.0f, 0.0f, 0.0f)));
                vec3 v = cross(axis, u);
                // Half a voxel inside the surface, as the voxels the samples land in reach half a voxel beyond them.
                float inner = max(r - 0.5f * voxel, 0.25f * voxel);
                int reach = min(int(ceil(2.0f * inner / voxel)), 24);
                for (int i = -reach; i <= reach; ++i) {
                    for (int j = -reach; j <= reach; ++j) {
                        vec2 q = 0.5f * voxel * vec2(float(i), float(j));
                        if (dot(q, q) <= inner * inner) {
                            addArea(p + q.x * u + q.y * v, solid * solidArea);
                        }
                    }
                }
            }
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
