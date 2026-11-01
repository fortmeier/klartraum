// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// The projected covariance shared by the compute and raster projections.
module gsplat_covariance;

import gsplat_types;

// Builds the matrix whose columns are the rows of the rotation of q, i.e. R^T.
public float3x3 quatToMat3(float4 q) {
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
public float2x2 computeCovarianceMatrix2D(float3 p, float4 rotation, float3 gscale, CameraUBO ubo) {
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
