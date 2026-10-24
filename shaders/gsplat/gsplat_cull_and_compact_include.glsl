// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Bindings, push constants and the visibility test shared by the three passes of the
// raster backend's cull + compaction (gsplat_cull_and_compact_count.comp, gsplat_cull_and_compact_scan.comp,
// gsplat_cull_and_compact_scatter.comp). The passes share one descriptor set and dispatch size.
layout(scalar, set = 0, binding = 0) readonly buffer Positions { vec3 positions[]; };

layout(set = 0, binding = 1) uniform UBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 cameraWorldPos;
} ubo;

layout(scalar, set = 0, binding = 2) buffer Keys    { uint keys[]; };
layout(scalar, set = 0, binding = 3) buffer Indices { uint indices[]; };

layout(scalar, set = 0, binding = 4) buffer DrawArgs {
    uint vertexCount;
    uint instanceCount;
    uint firstVertex;
    uint firstInstance;
} drawArgs;

// (r, g, b, post-activation alpha). Used only for the opacity cull below.
layout(scalar, set = 0, binding = 5) readonly buffer ColorsAlpha { vec4 colorsAlpha[]; };

layout(push_constant) uniform PC {
    uint  numSplats;
    float frustumDilation;
    float alphaThreshold;  // drop splats with post-activation alpha below this (0 = off)
} pc;

// Order-preserving float -> uint mapping: non-negative floats keep their
// IEEE-754 bit pattern (already monotonic), negative floats get all bits
// flipped so larger-magnitude negatives map to smaller uints. This lets an
// ascending uint radix sort reproduce ascending float order.
uint encodeDepthKey(float z) {
    uint b = floatBitsToUint(z);
    uint mask = (b & 0x80000000u) != 0u ? 0xFFFFFFFFu : 0x80000000u;
    return b ^ mask;
}

// Per-workgroup visible counts (pass 1), turned into exclusive offsets in place (pass 2).
layout(scalar, set = 0, binding = 7) buffer WorkgroupOffsets { uint workgroupOffsets[]; };

// Whether splat `id` survives the frustum and opacity cull; `key` is its depth key.
// View space here is right-handed (camera looks down -Z), so farther splats have more
// negative viewPos.z; encoding it directly maps them to smaller keys, and the ascending
// sort in stage B then yields back-to-front draw order, as required for premultiplied
// "over" blending.
bool visibleSplat(uint id, out uint key) {
    key = 0u;
    if (id >= pc.numSplats) {
        return false;
    }

    vec4 viewPos = ubo.view * ubo.model * vec4(positions[id], 1.0);
    vec4 ndc = ubo.proj * viewPos;
    ndc /= ndc.w;

    // Frustum-cull on the (dilated) splat center; the dilation keeps splats
    // whose footprint still overlaps the screen near the edges.
    float clip = 1.0 + pc.frustumDilation;
    if (abs(ndc.x) > clip || abs(ndc.y) > clip || ndc.z < -pc.frustumDilation || ndc.z > 1.0) {
        return false;
    }

    // Opacity cull (perf plan R4): a splat with post-activation alpha below the
    // threshold contributes less than that alpha even at its centre, so at
    // ~1/255 it is invisible. Dropping it here removes its vertex, fragment and
    // sort work and raises the cull rate that the visible-count sort benefits
    // from. threshold 0 disables the cull.
    if (colorsAlpha[id].a < pc.alphaThreshold) {
        return false;
    }

    key = encodeDepthKey(viewPos.z);
    return true;
}
