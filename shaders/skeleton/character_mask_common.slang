// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Interface shared by character_mask.vert and character_mask.frag.
module character_mask_common;

// Must match CharacterMaskPushConstants in include/klartraum/character_mask.hpp.
public struct CharacterMaskPushConstants {
    public float2 resolution;
    public float margin;
    public float occlusionBias;
    public uint useOcclusion;
};
[[vk::push_constant]] public ConstantBuffer<CharacterMaskPushConstants> pc;

// The vertex outputs of character_mask.vert.
public struct CharacterMaskVertex {
    public float4 position : SV_Position;
    // the segment's ends in framebuffer coordinates
    [[vk::location(0)]] public nointerpolation float4 bone;
    // capsule radius in pixels at each end
    [[vk::location(1)]] public nointerpolation float2 radii;
    // see occlusion.slang
    [[vk::location(2)]] public nointerpolation float2 boneDepth;
    [[vk::location(3)]] public nointerpolation float3 projection;
};
