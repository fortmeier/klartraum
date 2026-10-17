// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Interface shared by character_mask.vert and character_mask.frag. Must match
// CharacterMaskPushConstants in include/klartraum/character_mask.hpp.
layout(push_constant) uniform PushConstants {
    vec2 resolution;
    float margin;
    float occlusionBias;
    uint useOcclusion;
} pc;
