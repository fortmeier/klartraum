// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#version 450

// The pixels within a tapered capsule around the segment: closer to the nearest
// point of the bone than the radius there, interpolated between the ends.

#include "character_mask_common.glsl"

layout(location = 0) flat in vec4 bone;
layout(location = 1) flat in vec2 radii;
layout(location = 2) flat in vec2 boneDepth;
layout(location = 3) flat in vec3 projection;

layout(location = 0) out vec4 outColor;

#include "occlusion.glsl"

void main() {
    vec2 first = bone.xy, along = bone.zw - bone.xy;
    float lengthSquared = dot(along, along);
    float t = lengthSquared > 0.0 ? clamp(dot(gl_FragCoord.xy - first, along) / lengthSquared, 0.0, 1.0) : 0.0;
    if (length(gl_FragCoord.xy - (first + t * along)) > mix(radii.x, radii.y, t)) {
        discard;
    }
    if (pc.useOcclusion != 0u && occluded(bone, boneDepth, projection, pc.occlusionBias, pc.resolution)) {
        discard;
    }
    outColor = vec4(1.0);
}
