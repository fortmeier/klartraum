// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Shared by the Gaussian compute elements.
module gaussian_common;

__exported import scalar_buffer;

// Quaternions are stored x, y, z, w.
public float4 quatMul(float4 a, float4 b) {
    return float4(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}

// The rotation matrix of a unit quaternion; mul(quatToMat3(q), v) rotates v.
public float3x3 quatToMat3(float4 q) {
    float x = q.x, y = q.y, z = q.z, w = q.w;
    // The vectors are the matrix columns.
    return transpose(float3x3(
        float3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)),
        float3(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)),
        float3(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y))));
}

// Transform buffer layout written by make_transform.comp.slang.
public static const int kTranslation = 0; // 3 floats
public static const int kScale = 3;
public static const int kRotation = 4; // quaternion, 4 floats
public static const int kShBand1 = 8;  // 3 x 3, row-major
public static const int kShBand2 = 17; // 5 x 5
public static const int kShBand3 = 42; // 7 x 7
