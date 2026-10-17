// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Interface shared by skeleton_mesh.vert and skeleton_mesh.frag. Must match
// SkeletonMeshPushConstants in include/klartraum/skeleton_mesh_renderer.hpp.
layout(push_constant) uniform PushConstants {
    vec2 resolution;
    uint boneCount;
    uint drawSpheres;
    float occlusionBias;
    uint useOcclusion;
} pc;
