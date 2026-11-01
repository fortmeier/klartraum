// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

import gsplat_covariance;
import gsplat_types;
import matrix_math;
import scalar_buffer;

// --- Gaussian3D SoA inputs (band-major SH: shR[band * numElements + idx]) ---
[[vk::binding(0, 0)]] ScalarBuffer<float3> positions;
[[vk::binding(1, 0)]] ScalarBuffer<float4> rotations;
[[vk::binding(2, 0)]] ScalarBuffer<float3> scales;
[[vk::binding(3, 0)]] ScalarBuffer<float4> colorsAlpha; // (r,g,b,alpha)
[[vk::binding(4, 0)]] ScalarBuffer<float> shR;          // shR[band*N+idx]
[[vk::binding(5, 0)]] ScalarBuffer<float> shG;
[[vk::binding(6, 0)]] ScalarBuffer<float> shB;

[[vk::binding(7, 0)]] ConstantBuffer<CameraUBO> ubo;

// --- Gaussian2D SoA outputs ---
[[vk::binding(8, 0)]] RWScalarBuffer<float2> pos2D;
[[vk::binding(9, 0)]] RWScalarBuffer<float> zBuf;
[[vk::binding(10, 0)]] RWScalarBuffer<uint> binMaskBuf;
// covInv packed as float4: (m00, m10, m01, m11), column by column
[[vk::binding(11, 0)]] RWScalarBuffer<float4> covInvBuf;
[[vk::binding(12, 0)]] RWScalarBuffer<float4> colorAlpha2D; // (r,g,b,alpha)

struct PushConstants {
    uint numElements;
    uint gridSize;
    float screenWidth;
    float screenHeight;
};
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

float computeSH(float base, float coeffs[15], float3 dir) {
    float x = dir.x, y = dir.y, z = dir.z;
    float x2 = x * x, y2 = y * y, z2 = z * z;
    float sh[16];
    sh[0] = 0.282095;
    sh[1] = -0.488603 * y;
    sh[2] = -0.488603 * z;
    sh[3] = -0.488603 * x;
    sh[4] = 1.092548 * x * y;
    sh[5] = -1.092548 * y * z;
    sh[6] = 0.315392 * (2.0 * z2 - x2 - y2);
    sh[7] = -1.092548 * x * z;
    sh[8] = 0.546274 * (x2 - y2);
    sh[9] = -0.590044 * y * (3.0 * x2 - y2);
    sh[10] = 2.890611 * x * y * z;
    sh[11] = -0.457046 * y * (4.0 * z2 - x2 - y2);
    sh[12] = 0.373176 * z * (2.0 * z2 - 3.0 * x2 - 3.0 * y2);
    sh[13] = -0.457046 * x * (4.0 * z2 - x2 - y2);
    sh[14] = 1.445306 * z * (x2 - y2);
    sh[15] = -0.590044 * x * (x2 - 3.0 * y2);
    float value = 0.5 + base * sh[0];
    for (int i = 0; i < 15; i++) value += coeffs[i] * sh[i + 1];
    return value;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint3 globalId: SV_DispatchThreadID) {
    uint idx = globalId.x;
    if (idx >= pc.numElements) return;

    float3 p = positions[idx];
    float4 r = rotations[idx];
    float3 s = scales[idx];
    float4 ca = colorsAlpha[idx];

    // Load SH coefficients: band-major layout shX[band * numElements + idx]
    float shRc[15], shGc[15], shBc[15];
    for (int i = 0; i < 15; i++) {
        shRc[i] = shR[i * pc.numElements + idx];
        shGc[i] = shG[i * pc.numElements + idx];
        shBc[i] = shB[i * pc.numElements + idx];
    }

    // Project position to clip space
    float4 clipPos = mul(mul(mul(ubo.proj, ubo.view), ubo.model), float4(p, 1.0));
    float2 normPos = clipPos.xy / clipPos.w;
    float depth = (clipPos.xyz / clipPos.w).z;

    float swh = pc.screenWidth * 0.5;
    float shh = pc.screenHeight * 0.5;

    pos2D[idx] = float2((normPos.x + 1.0) * swh, (normPos.y + 1.0) * shh);
    zBuf[idx] = depth;
    binMaskBuf[idx] = uint(-1); // downstream binning pass fills real value

    // Compute projected covariance and its inverse.
    // The Jacobian maps NDC (which spans 2 units) to pixels anisotropically:
    // x by (screenWidth/2)² and y by (screenHeight/2)²
    float2x2 cov = computeCovarianceMatrix2D(p, r, s, ubo);
    cov[0][0] *= swh * swh;
    cov[1][0] *= swh * shh;
    cov[0][1] *= swh * shh;
    cov[1][1] *= shh * shh;
    cov[0][0] += 0.3;
    cov[1][1] += 0.3;
    float2x2 covI = inverse(cov);
    // Store column by column: (m00, m10, m01, m11)
    covInvBuf[idx] = float4(covI[0][0], covI[1][0], covI[0][1], covI[1][1]);

    // Spherical harmonics colour — SH coefficients are defined in world space,
    // so the view direction must be the world-space camera→Gaussian vector.
    float3 dir = normalize(p - ubo.cameraWorldPos.xyz);
    float3 color = float3(computeSH(ca.r, shRc, dir), computeSH(ca.g, shGc, dir), computeSH(ca.b, shBc, dir));
    colorAlpha2D[idx] = float4(color, ca.a);
}
