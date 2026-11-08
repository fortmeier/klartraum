// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Cameras as poses (camera_build.comp.slang, camera_pose_backward.comp.slang): each view's clip-from-world matrix is
//   M = P R(q) T(-e),
// with q a quaternion (x, y, z, w; normalized where used), e the eye's position and P the fixed rest (the projection
// and the calibrated camera's orientation): a vertex x is seen at y = R(q) (x - e) and projected by P. Training q and e
// corrects the calibrated poses.
module camera_pose;

// The rotation of quaternion q applied to v.
[Differentiable]
public float3 rotate(float4 q, float3 v) {
    const float4 n = normalize(q);
    const float3 u = n.xyz;
    return v + 2.0 * cross(u, cross(u, v) + n.w * v);
}

// Where the camera of pose (q, e) sees the point x, before P.
[Differentiable]
public float3 viewPoint(float4 q, float3 e, no_diff float3 x) {
    return rotate(q, x - e);
}

// The rotation matrix of quaternion q, as a 4 x 4 matrix (for mul(M, v)).
public float4x4 rotationMatrix(float4 q) {
    const float3 c0 = rotate(q, float3(1.0, 0.0, 0.0)), c1 = rotate(q, float3(0.0, 1.0, 0.0)),
                 c2 = rotate(q, float3(0.0, 0.0, 1.0));
    return float4x4(c0.x, c1.x, c2.x, 0.0, c0.y, c1.y, c2.y, 0.0, c0.z, c1.z, c2.z, 0.0, 0.0, 0.0, 0.0, 1.0);
}
