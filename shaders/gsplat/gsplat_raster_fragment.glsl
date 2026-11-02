// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Body of gsplat_raster.frag and gsplat_raster_depth.frag. With OUTPUT_DEPTH
// defined, a fragment whose splat covers the pixel with an opacity of at least
// 0.5 also lowers the pixel's depth to the splat's: an atomicMin on an
// order-preserving uint encoding of the NDC depth, decoded by
// gsplat_depth_resolve.comp. The splats arrive back to front, so where the
// transmittance crosses 0.5 is not known here; the nearest opaque-enough splat
// is.

layout(location = 0) in vec4 inColor;
layout(location = 1) in vec3 inConic;
layout(location = 2) in vec2 inCenterPx;
layout(location = 3) flat in float inDepth;

layout(location = 0) out vec4 outFragColor;

#ifdef OUTPUT_DEPTH
#include "gsplat_depth_encoding.glsl"

layout(set = 1, binding = 2) buffer DepthBits { uint depthBits[]; };
layout(push_constant) uniform PushConstants {
    vec2 resolution;
} pc;

const float CORE_OPACITY = 0.5;
#endif

void main() {
    // Exact per-pixel Gaussian weight from the conic (inverse projected
    // covariance): exp(-0.5 * diff^T * conic * diff), the same evaluation
    // gsplat_binned_splatting.comp performs — conic is symmetric
    // (inConic = (a, b, c) for [[a,b],[b,c]]), so the cross term is 2*b*dx*dy.
    vec2  d = gl_FragCoord.xy - inCenterPx;
    float q = inConic.x * d.x * d.x + 2.0 * inConic.y * d.x * d.y + inConic.z * d.y * d.y;
    float falloff = exp(-0.5 * q);
    if (falloff < 1.0 / 255.0) {
        discard;
    }

#ifdef OUTPUT_DEPTH
    if (inColor.a * falloff >= CORE_OPACITY) {
        uint pixel = uint(gl_FragCoord.y) * uint(pc.resolution.x) + uint(gl_FragCoord.x);
        atomicMin(depthBits[pixel], encodeDepth(inDepth));
    }
#endif

    // inColor is premultiplied (rgb already scaled by alpha); multiplying by
    // the falloff keeps it premultiplied for the hardware blender.
    outFragColor = inColor * falloff;
}
