// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#version 450

#include "openpose_common.glsl"

layout(location = 0) flat in vec2 center;
layout(location = 1) flat in vec2 axes;
layout(location = 2) flat in vec2 direction;
layout(location = 3) flat in vec3 color;
layout(location = 4) flat in vec4 bone;
layout(location = 5) flat in vec2 boneDepth;
layout(location = 6) flat in vec3 projection;

layout(location = 0) out vec4 outColor;

#include "occlusion.glsl"

void main() {
    vec2 offset = gl_FragCoord.xy - center;
    if (pc.drawJoints != 0u) {
        // OpenCV's filled circle: the pixels whose offset lies within the radius.
        if (dot(offset, offset) > axes.x * axes.x + 1e-3) {
            discard;
        }
    } else {
        vec2 local = vec2(dot(offset, direction), dot(offset, vec2(-direction.y, direction.x))) / axes;
        if (dot(local, local) > 1.0) {
            discard;
        }
    }
    if (pc.useOcclusion != 0u && occluded(bone, boneDepth, projection, pc.occlusionBias, pc.resolution)) {
        discard;
    }
    outColor = vec4(color, 1.0);
}
