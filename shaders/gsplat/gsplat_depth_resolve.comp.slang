// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Decodes the raster backend's encoded depths (gsplat_raster_depth.frag) into
// NDC depths; pixels no splat covered opaquely enough become +inf.
import gsplat_depth_encoding;

struct PushConstants {
    uint pixelCount;
    uint flags;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] StructuredBuffer<uint> depthBits;
[[vk::binding(1, 0)]] RWStructuredBuffer<float> depth;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> sceneDepth;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint pixel = globalId.x;
    if (pixel < pc.pixelCount) {
        float selected = decodeDepth(depthBits[pixel]);
        if ((pc.flags & 2u) != 0u) depth[pixel] = selected;
        if ((pc.flags & 1u) != 0u && !isinf(selected)) sceneDepth[pixel] = clamp(selected, 0.0, 1.0);
    }
}
