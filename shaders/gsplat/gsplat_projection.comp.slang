// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

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

struct UBO {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
    float4 cameraWorldPos;
};
[[vk::binding(7, 0)]] ConstantBuffer<UBO> ubo;

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

// Builds the matrix whose columns are the rows of the rotation of q, i.e. R^T.
float3x3 quatToMat3(float4 q) {
    float x = q.x, y = q.y, z = q.z, w = q.w;
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, xz = x * z, yz = y * z;
    float wx = w * x, wy = w * y, wz = w * z;
    // float3x3(...) takes rows; the transpose makes the triples columns.
    return transpose(float3x3(
        1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy),
        2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx),
        2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)));
}

// Projected 2D covariance of a 3D Gaussian, in NDC units (Kerbl et al. 2023, Sec. 4;
// Zwicker et al., EWA Splatting 2002). The caller scales the entries to pixels.
float2x2 computeCovarianceMatrix2D(float3 p, float4 rotation, float3 gscale) {
    // 3D covariance Sigma = R S S^T R^T (Kerbl Eq. 6). quatToMat3 returns R^T;
    // transposing it yields R.
    float3x3 R = transpose(quatToMat3(rotation));
    float3x3 RS = mul(R, float3x3(gscale.x, 0.0, 0.0,
                                  0.0, gscale.y, 0.0,
                                  0.0, 0.0, gscale.z));
    float3x3 Sigma = mul(RS, transpose(RS));

    // Viewing transformation W and the camera-space mean t.
    float4x4 modelView = mul(ubo.model, ubo.view);
    float3x3 W = (float3x3)modelView;
    float3 t = mul(modelView, float4(p, 1.0)).xyz;

    // Signed focal terms; proj[1][1] is negative due to Vulkan's Y-flip, and the
    // sign carries through to the Jacobian's off-diagonal term.
    float fx = ubo.proj[0][0];
    float fy = ubo.proj[1][1];

    // The camera looks down -z (clip w = -t.z), so ndc = (-fx * t.x / t.z, -fy * t.y / t.z).
    // The affine approximation is evaluated at the mean, with the projected x/y
    // limited to a [-1.3, 1.3] NDC guard band (keeping the depth) so Gaussians far
    // outside the view do not get an exploding Jacobian.
    // The band is centred on the view: an off-centre projection (e.g. cropProjection())
    // adds a constant to the NDC, ndc.x = -fx * t.x / t.z - P[0][2] (row 0, column 2),
    // so the view's centre lies at t.x / t.z = -P[0][2] / fx. The constant does not
    // change the Jacobian.
    float limX = 1.3 / abs(fx);
    float limY = 1.3 / abs(fy);
    float centreX = -ubo.proj[0][2] / fx;
    float centreY = -ubo.proj[1][2] / fy;
    float tx = clamp(t.x / t.z, centreX - limX, centreX + limX);
    float ty = clamp(t.y / t.z, centreY - limY, centreY + limY);

    // Jacobian of the NDC x/y w.r.t. camera-space position (local affine
    // approximation of Zwicker et al.); the third row is zero, so only the upper-left
    // 2x2 block of the result is meaningful.
    float3x3 J = float3x3(
        -fx / t.z, 0.0, fx * tx / t.z,
        0.0, -fy / t.z, fy * ty / t.z,
        0.0, 0.0, 0.0);

    // Sigma' = J W Sigma W^T J^T (Kerbl Eq. 5), keeping its 2x2 image-plane part.
    float3x3 T = mul(J, W);
    float3x3 cov = mul(mul(T, Sigma), transpose(T));
    return (float2x2)cov;
}

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
    float2x2 cov = computeCovarianceMatrix2D(p, r, s);
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
