// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Interface shared by openpose.vert and openpose.frag. Must match
// OpenPosePushConstants in include/klartraum/openpose_skeleton_renderer.hpp.
layout(push_constant) uniform PushConstants {
    vec2 resolution;
    float stickWidth;
    float jointRadius;
    uint firstJoint;
    uint drawJoints;
    float occlusionBias;
    uint useOcclusion;
    uint hideOccludedFace;
} pc;
