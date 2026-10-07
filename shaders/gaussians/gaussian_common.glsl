// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Shared by the Gaussian compute elements.

// Quaternions are stored x, y, z, w.
vec4 quatMul(vec4 a, vec4 b) {
    return vec4(a.w * b.xyz + b.w * a.xyz + cross(a.xyz, b.xyz), a.w * b.w - dot(a.xyz, b.xyz));
}

// The rotation matrix of a unit quaternion (columns, as GLSL stores them).
mat3 quatToMat3(vec4 q) {
    float x = q.x, y = q.y, z = q.z, w = q.w;
    return mat3(
        vec3(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)),
        vec3(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)),
        vec3(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)));
}

// Transform buffer layout written by make_transform.comp.
const int kTranslation = 0;  // 3 floats
const int kScale = 3;
const int kRotation = 4;     // quaternion, 4 floats
const int kShBand1 = 8;      // 3 x 3, row-major
const int kShBand2 = 17;     // 5 x 5
const int kShBand3 = 42;     // 7 x 7
