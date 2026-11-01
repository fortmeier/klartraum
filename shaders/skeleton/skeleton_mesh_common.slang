// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Interface shared by skeleton_mesh.vert and skeleton_mesh.frag.
module skeleton_mesh_common;

// Must match SkeletonMeshPushConstants in include/klartraum/skeleton_mesh_renderer.hpp.
public struct SkeletonMeshPushConstants {
    public float2 resolution;
    public uint boneCount;
    public uint drawSpheres;
    public float occlusionBias;
    public uint useOcclusion;
};
[[vk::push_constant]] public ConstantBuffer<SkeletonMeshPushConstants> pc;

// The vertex outputs of skeleton_mesh.vert.
public struct SkeletonMeshVertex {
    public float4 position : SV_Position;
    [[vk::location(0)]] public float3 viewPosition;
    [[vk::location(1)]] public float3 viewNormal;
    // clip z; the NDC depth with an orthographic projection
    [[vk::location(2)]] public float clipDepth;
    // rgb, opacity
    [[vk::location(3)]] public nointerpolation float4 colour;
    // P[2][3], P[2][2] (row, column), 1 if perspective
    [[vk::location(4)]] public nointerpolation float3 projection;
};
