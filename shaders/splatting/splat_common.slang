// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Differentiable Gaussian splatting for training (Kerbl et al. 2023), shared by the forward and backward kernels.
// A Gaussian has a position, log scales, a rotation quaternion (x, y, z, w; normalized here), a logit opacity and an
// rgb colour. Under a camera (clip from world, Vulkan conventions; pixel (x, y) has its centre at (x + 0.5, y + 0.5))
// it projects to a screen mean, a conic (the inverse of its screen covariance, EWA splatting with the exact Jacobian
// of the projection at the mean, plus a 0.3 pixel^2 low-pass) and an opacity. Its alpha at a pixel is
// opacity x exp(-d^T conic d / 2), d the offset from the mean. Slang's bwd_diff of these functions gives the
// gradients of the backward kernels.
module splat_common;

// Pixels per tile side; a tile is one workgroup of the render kernels.
public static const uint TILE = 16;
// Sort keys: tile index in the high 12 bits, quantized depth in the low 20.
public static const uint DEPTH_BITS = 20;
// Splats fainter than this at a pixel are skipped.
public static const float MIN_ALPHA = 1.0 / 255.0;
// Alpha is clamped to this, so transmittance never reaches zero.
public static const float MAX_ALPHA = 0.99;
// Blending stops once the transmittance would fall below this.
public static const float MIN_TRANSMITTANCE = 1e-4;

// The screen position of the mean and its clip w (the depth).
[Differentiable]
public float3 projectMean(float3 position, no_diff float4x4 camera, no_diff float2 size) {
    const float4 clip = mul(camera, float4(position, 1.0));
    return float3((clip.xy / clip.w * 0.5 + 0.5) * size, clip.w);
}

// The screen covariance (xx, xy, yy) in pixels^2, including the low-pass.
[Differentiable]
public float3 projectCovariance(float3 position, float3 logScale, float4 rotation, no_diff float4x4 camera,
                                no_diff float2 size) {
    const float4 clip = mul(camera, float4(position, 1.0));
    const float3 row0 = float3(camera[0][0], camera[0][1], camera[0][2]);
    const float3 row1 = float3(camera[1][0], camera[1][1], camera[1][2]);
    const float3 row3 = float3(camera[3][0], camera[3][1], camera[3][2]);
    // Rows of the Jacobian of the screen position with respect to the world position.
    const float3 j0 = 0.5 * size.x * (row0 * clip.w - clip.x * row3) / (clip.w * clip.w);
    const float3 j1 = 0.5 * size.y * (row1 * clip.w - clip.y * row3) / (clip.w * clip.w);
    const float4 q = rotation / length(rotation);
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    // Columns of the rotation matrix, scaled: the covariance is M M^T with M = R diag(scale).
    const float3 scale = exp(logScale);
    const float3 m0 = scale.x * float3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y));
    const float3 m1 = scale.y * float3(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x));
    const float3 m2 = scale.z * float3(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y));
    const float3 a = float3(dot(j0, m0), dot(j0, m1), dot(j0, m2));
    const float3 b = float3(dot(j1, m0), dot(j1, m1), dot(j1, m2));
    return float3(dot(a, a) + 0.3, dot(a, b), dot(b, b) + 0.3);
}

// The conic (inverse covariance: xx, xy, yy) of a screen covariance.
[Differentiable]
public float3 conicOf(float3 covariance) {
    const float determinant = covariance.x * covariance.z - covariance.y * covariance.y;
    return float3(covariance.z, -covariance.y, covariance.x) / determinant;
}

[Differentiable]
public float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

// The alpha of a splat at a pixel centre, before the clamp to MAX_ALPHA.
[Differentiable]
public float splatAlpha(float2 mean, float3 conic, float opacity, no_diff float2 pixel) {
    const float2 d = pixel - mean;
    const float power = -0.5 * (conic.x * d.x * d.x + conic.z * d.y * d.y) - conic.y * d.x * d.y;
    return opacity * exp(min(power, 0.0));
}

// The radius in pixels holding three standard deviations of a screen covariance.
public float splatRadius(float3 covariance) {
    const float middle = 0.5 * (covariance.x + covariance.z);
    const float determinant = covariance.x * covariance.z - covariance.y * covariance.y;
    return ceil(3.0 * sqrt(middle + sqrt(max(0.1, middle * middle - determinant))));
}
