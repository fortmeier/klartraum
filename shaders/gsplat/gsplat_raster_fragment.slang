// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Body of gsplat_raster.frag and gsplat_raster_depth.frag. With outputDepth,
// a fragment meeting the configured opacity threshold can lower the
// pixel's depth to the splat's: an InterlockedMin on an
// order-preserving uint encoding of the NDC depth, decoded by
// gsplat_depth_resolve.comp. The splats arrive back to front, so where the
// transmittance crosses 0.5 is not known here; the nearest opaque-enough splat
// is. The depth resources are only part of the depth variant's module.
module gsplat_raster_fragment;

__exported import gsplat_raster_types;
import gsplat_depth_encoding;

// With outputDepth: the encoded nearest depth per pixel.
[[vk::binding(2, 1)]] RWStructuredBuffer<uint> depthBits;

struct PushConstants {
    float2 resolution;
    float2 focal;
    float splatScale;
    int shDegree;
    uint numSplats;
    float depthThreshold;
};
// With outputDepth.
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

// Returns the premultiplied colour of the fragment of `input`, or discards it.
public float4 rasterFragment<let outputDepth : bool>(SplatVertex input) {
    // Exact per-pixel Gaussian weight from the conic (inverse projected
    // covariance): exp(-0.5 * diff^T * conic * diff), the same evaluation
    // gsplat_binned_splatting performs — conic is symmetric
    // (conic = (a, b, c) for [[a,b],[b,c]]), so the cross term is 2*b*dx*dy.
    float2 d = input.position.xy - input.centerPx;
    float q = input.conic.x * d.x * d.x + 2.0 * input.conic.y * d.x * d.y + input.conic.z * d.y * d.y;
    float falloff = exp(-0.5 * q);
    if (falloff < 1.0 / 255.0) {
        discard;
    }

    if (outputDepth) {
        uint pixel = uint(input.position.y) * uint(pc.resolution.x) + uint(input.position.x);
        if (input.color.a * falloff >= pc.depthThreshold) {
            InterlockedMin(depthBits[pixel], encodeDepth(input.depth));
        }
    }

    // The colour is premultiplied (rgb already scaled by alpha); multiplying by
    // the falloff keeps it premultiplied for the hardware blender.
    return input.color * falloff;
}
