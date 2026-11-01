// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Per-splat 2D attribute precompute for the raster backend (perf plan R1+R2).
// Runs once per splat and writes a compact Splat2D record the vertex shader
// reads verbatim — moving the covariance projection, 2x2 eigendecomposition and
// SH evaluation out of the vertex stage, where the same math previously ran four
// times per splat (once per quad corner). The math is identical to the prior
// gsplat_raster.vert / gsplat_projection.comp implementation.

import gsplat_covariance;
import gsplat_raster_types;
import gsplat_types;
import scalar_buffer;

// set 0: per-splat SoA inputs (band-major SH: shX[band * numSplats + idx]),
// the camera UBO, and the Splat2D output buffer.
[[vk::binding(0, 0)]] ScalarBuffer<float3> positions;
[[vk::binding(1, 0)]] ScalarBuffer<float4> rotations;
[[vk::binding(2, 0)]] ScalarBuffer<float3> scales;
[[vk::binding(3, 0)]] ScalarBuffer<float4> colorsAlpha;
[[vk::binding(4, 0)]] ScalarBuffer<float> shR;
[[vk::binding(5, 0)]] ScalarBuffer<float> shG;
[[vk::binding(6, 0)]] ScalarBuffer<float> shB;
[[vk::binding(7, 0)]] ConstantBuffer<CameraUBO> ubo;
[[vk::binding(8, 0)]] RWScalarBuffer<Splat2D> splat2D;

struct PushConstants {
    uint numSplats;
    float screenWidth;
    float screenHeight;
    float splatScale;
    int shDegree;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

// World-space SH evaluation, gated on `deg`: bands beyond the requested degree
// are neither read from memory nor evaluated (perf plan R2). For deg == 3 the
// accumulation order matches the prior unconditional degree-3 loop, so a
// full-degree scene reproduces the previous result.
float computeSH(float base, float c[15], float3 dir, int deg) {
    float value = 0.5 + base * 0.282095;
    if (deg < 1) return value;

    float x = dir.x, y = dir.y, z = dir.z;
    value += c[0] * (-0.488603 * y);
    value += c[1] * (-0.488603 * z);
    value += c[2] * (-0.488603 * x);
    if (deg < 2) return value;

    float x2 = x*x, y2 = y*y, z2 = z*z;
    value += c[3] * (1.092548 * x * y);
    value += c[4] * (-1.092548 * y * z);
    value += c[5] * (0.315392 * (2.0*z2 - x2 - y2));
    value += c[6] * (-1.092548 * x * z);
    value += c[7] * (0.546274 * (x2 - y2));
    if (deg < 3) return value;

    value += c[8]  * (-0.590044 * y * (3.0*x2 - y2));
    value += c[9]  * (2.890611 * x * y * z);
    value += c[10] * (-0.457046 * y * (4.0*z2 - x2 - y2));
    value += c[11] * (0.373176 * z * (2.0*z2 - 3.0*x2 - 3.0*y2));
    value += c[12] * (-0.457046 * x * (4.0*z2 - x2 - y2));
    value += c[13] * (1.445306 * z * (x2 - y2));
    value += c[14] * (-0.590044 * x * (x2 - 3.0*y2));
    return value;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint id = globalId.x;
    if (id >= pc.numSplats) return;

    float3 position   = positions[id];
    float4 rotation   = rotations[id];
    float3 scale      = scales[id];
    float4 colorAlpha = colorsAlpha[id];

    float4 clip = mul(mul(mul(ubo.proj, ubo.view), ubo.model), float4(position, 1.0));
    float2 ndcCenter = clip.xy / clip.w;

    float swh = pc.screenWidth  * 0.5;
    float shh = pc.screenHeight * 0.5;

    // Projected covariance in pixels, with the same dilation both backends use.
    float2x2 cov = computeCovarianceMatrix2D(position, rotation, scale, ubo);
    float covA = cov[0][0] * swh * swh + 0.3;
    float covB = cov[1][0] * swh * shh;
    float covD = cov[1][1] * shh * shh + 0.3;

    // Closed-form eigendecomposition of the symmetric 2x2 covariance: the
    // eigenvectors are the ellipse's major/minor axes, sqrt(eigenvalue) the axis
    // sigma — orienting and sizing the quad to the splat's actual footprint.
    float trace = covA + covD;
    float det   = covA * covD - covB * covB;
    float disc  = sqrt(max(trace * trace * 0.25 - det, 0.0));
    float lambda1 = trace * 0.5 + disc;
    float lambda2 = max(trace * 0.5 - disc, 0.0);

    float2 majorDir = (abs(covB) > 1e-6)
        ? normalize(float2(covB, lambda1 - covA))
        : ((covA >= covD) ? float2(1.0, 0.0) : float2(0.0, 1.0));
    float2 minorDir = float2(-majorDir.y, majorDir.x);

    // Opacity-aware quad extent (perf plan R4): shrink the quad to the radius
    // where the splat's actual contribution alpha*exp(-0.5*q) drops to 1/255,
    // instead of always using splatScale sigma. At an s-sigma edge the
    // Mahalanobis distance q = s^2, so the cutoff is s = sqrt(2*ln(255*alpha)).
    // Clamp to splatScale so opaque splats keep the original footprint and only
    // low-opacity splats get smaller quads (cutting fragment overdraw). Splats
    // with alpha <= 1/255 yield a zero-size quad (they contribute < 1/255
    // everywhere, so the fragment shader would discard them anyway).
    float aCutoff  = 2.0 * log(255.0 * max(colorAlpha.a, 1e-8));
    float effScale = min(pc.splatScale, sqrt(max(aCutoff, 0.0)));
    float majorRadiusPx = effScale * sqrt(lambda1);
    float minorRadiusPx = effScale * sqrt(lambda2);

    // SH colour — coefficients live in world space, so the view direction is the
    // world-space camera->Gaussian vector. Only the requested degree's bands are
    // loaded and evaluated.
    int nc = pc.shDegree * (pc.shDegree + 2); // 0, 3, 8 or 15
    float shRc[15], shGc[15], shBc[15];
    for (int i = 0; i < nc; i++) {
        shRc[i] = shR[i * pc.numSplats + id];
        shGc[i] = shG[i * pc.numSplats + id];
        shBc[i] = shB[i * pc.numSplats + id];
    }
    float3 dir = normalize(position - ubo.cameraWorldPos.xyz);
    float3 color = float3(
        computeSH(colorAlpha.r, shRc, dir, pc.shDegree),
        computeSH(colorAlpha.g, shGc, dir, pc.shDegree),
        computeSH(colorAlpha.b, shBc, dir, pc.shDegree));

    float invDet = 1.0 / max(det, 1e-12);
    float alpha  = colorAlpha.a;

    Splat2D rec;
    rec.color     = float4(color * alpha, alpha);
    rec.conic     = float3(covD, -covB, covA) * invDet;
    rec.centerPx  = float2((ndcCenter.x + 1.0) * swh, (ndcCenter.y + 1.0) * shh);
    rec.ndcCenter = ndcCenter;
    rec.ndcAxis1  = majorDir * majorRadiusPx / float2(swh, shh);
    rec.ndcAxis2  = minorDir * minorRadiusPx / float2(swh, shh);
    rec.depth     = clip.z / clip.w;
    splat2D[id]   = rec;
}
