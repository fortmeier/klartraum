// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Interface shared by openpose.vert and openpose.frag.
module openpose_common;

// Must match OpenPosePushConstants in include/klartraum/openpose_skeleton_renderer.hpp.
public struct OpenPosePushConstants {
    public float2 resolution;
    public float stickWidth;
    public float jointRadius;
    public uint firstJoint;
    public uint drawJoints;
    public float occlusionBias;
    public uint useOcclusion;
    public uint hideOccludedFace;
};
[[vk::push_constant]] public ConstantBuffer<OpenPosePushConstants> pc;

// The vertex outputs of openpose.vert.
public struct OpenPoseVertex {
    public float4 position : SV_Position;
    // framebuffer coordinates (pixel centres at .5)
    [[vk::location(0)]] public nointerpolation float2 center;
    // ellipse semi-axes; x is the disc radius for joints
    [[vk::location(1)]] public nointerpolation float2 axes;
    // major axis (cos, sin) of a limb
    [[vk::location(2)]] public nointerpolation float2 direction;
    [[vk::location(3)]] public nointerpolation float3 color;
    // For the per-pixel occlusion test in openpose.frag: the shape's bone in framebuffer
    // coordinates (both ends at the joint for a joint disc), a depth at each end that is
    // linear in screen space (1 / distance with a perspective projection, NDC depth
    // otherwise), and the projection constants that turn the scene's NDC depth into it.
    [[vk::location(4)]] public nointerpolation float4 bone;
    [[vk::location(5)]] public nointerpolation float2 boneDepth;
    // P[2][3], P[2][2] (row, column), 1 if perspective
    [[vk::location(6)]] public nointerpolation float3 projection;
};
