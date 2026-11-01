// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Bindings, push constants and the visibility test shared by the three passes of the
// raster backend's cull + compaction (gsplat_cull_and_compact_count, _scan and
// _scatter). The passes share one descriptor set and dispatch size.
module gsplat_cull_and_compact;

__exported import scalar_buffer;

public struct CameraUBO {
    public float4x4 model;
    public float4x4 view;
    public float4x4 proj;
    public float4 cameraWorldPos;
};

public struct DrawArgs {
    public uint vertexCount;
    public uint instanceCount;
    public uint firstVertex;
    public uint firstInstance;
};

public struct CullPushConstants {
    public uint numSplats;
    public float frustumDilation;
    public float alphaThreshold; // drop splats with post-activation alpha below this (0 = off)
};
[[vk::push_constant]] public ConstantBuffer<CullPushConstants> pc;

[[vk::binding(0, 0)]] public ScalarBuffer<float3> positions;
[[vk::binding(1, 0)]] public ConstantBuffer<CameraUBO> ubo;
[[vk::binding(2, 0)]] public RWScalarBuffer<uint> keys;
[[vk::binding(3, 0)]] public RWScalarBuffer<uint> indices;
// One DrawArgs: the indirect draw of the visible splats.
[[vk::binding(4, 0)]] public RWScalarBuffer<DrawArgs> drawArgs;
// (r, g, b, post-activation alpha). Used only for the opacity cull below.
[[vk::binding(5, 0)]] public ScalarBuffer<float4> colorsAlpha;
// Per-workgroup visible counts (pass 1), turned into exclusive offsets in place (pass 2).
[[vk::binding(7, 0)]] public RWScalarBuffer<uint> workgroupOffsets;

// Order-preserving float -> uint mapping: non-negative floats keep their
// IEEE-754 bit pattern (already monotonic), negative floats get all bits
// flipped so larger-magnitude negatives map to smaller uints. This lets an
// ascending uint radix sort reproduce ascending float order.
uint encodeDepthKey(float z) {
    uint b = asuint(z);
    uint mask = (b & 0x80000000u) != 0u ? 0xFFFFFFFFu : 0x80000000u;
    return b ^ mask;
}

// Whether splat `id` survives the frustum and opacity cull; `key` is its depth key.
// View space here is right-handed (camera looks down -Z), so farther splats have more
// negative viewPos.z; encoding it directly maps them to smaller keys, and the ascending
// sort in stage B then yields back-to-front draw order, as required for premultiplied
// "over" blending.
public bool visibleSplat(uint id, out uint key) {
    key = 0u;
    if (id >= pc.numSplats) {
        return false;
    }

    float4 viewPos = mul(mul(ubo.view, ubo.model), float4(positions[id], 1.0));
    float4 ndc = mul(ubo.proj, viewPos);
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
