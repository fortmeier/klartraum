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

// NDC depth of the scene per pixel (+inf: no surface); read only with pc.useOcclusion.
layout(std430, set = 1, binding = 1) readonly buffer OcclusionDepth {
    float occlusionDepth[];
};

// Whether the scene surface at this pixel lies in front of the skeleton. The
// skeleton's depth here is that of the nearest point of its bone, interpolated
// along the bone in screen space: 1 / distance (perspective) or NDC depth is
// linear there. The fragment is behind if it is farther than the surface by more
// than pc.occlusionBias of the surface's distance.
bool occluded() {
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    float surface = occlusionDepth[uint(pixel.y) * uint(pc.resolution.x) + uint(pixel.x)];
    if (isinf(surface)) {
        return false;
    }
    vec2 first = bone.xy, along = bone.zw - bone.xy;
    float lengthSquared = dot(along, along);
    float t = lengthSquared > 0.0 ? clamp(dot(gl_FragCoord.xy - first, along) / lengthSquared, 0.0, 1.0) : 0.0;
    float depth = mix(boneDepth.x, boneDepth.y, t);
    if (projection.z != 0.0) {
        float distance = 1.0 / depth;
        float surfaceDistance = projection.x / (surface + projection.y);
        return distance > surfaceDistance * (1.0 + pc.occlusionBias);
    }
    return depth > surface + pc.occlusionBias * abs(surface);
}

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
    if (pc.useOcclusion != 0u && occluded()) {
        discard;
    }
    outColor = vec4(color, 1.0);
}
