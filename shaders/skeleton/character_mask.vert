// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#version 450

// One quad per segment of the character mask (instance = segment), around a
// capsule whose radius at each end is the segment's radius plus the margin,
// projected at that end's distance.

#include "character_mask_common.glsl"

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 cameraWorldPos;
} ubo;

layout(std430, set = 1, binding = 0) readonly buffer Joints {
    vec4 joints[];
};

layout(location = 0) flat out vec4 bone;      // the segment's ends in framebuffer coordinates
layout(location = 1) flat out vec2 radii;     // capsule radius in pixels at each end
layout(location = 2) flat out vec2 boneDepth; // see occlusion.glsl
layout(location = 3) flat out vec3 projection;

// Joint pairs and radii of characterMaskSegments() (keep in sync).
const uint SEGMENT_COUNT = 21u;
const uint SEGMENT_JOINTS[2 * SEGMENT_COUNT] = uint[](1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 1, 8, 8, 9, 9, 10, 1, 11, 11,
                                                      12, 12, 13, 1, 0, 0, 14, 14, 16, 0, 15, 15, 17, 2, 8, 5, 11, 8,
                                                      11, 16, 17);
const float SEGMENT_RADII[SEGMENT_COUNT] = float[](0.07, 0.07, 0.07, 0.055, 0.07, 0.055, 0.14, 0.085, 0.065, 0.14, 0.085,
                                                    0.065, 0.06, 0.05, 0.05, 0.05, 0.05, 0.11, 0.11, 0.11, 0.1);
// The last segment, the head, is a disc around the middle of its two joints (the ears).
const uint HEAD_DISC = SEGMENT_COUNT - 1u;

bool perspective() { return ubo.proj[2][3] != 0.0; }

// Position in pixels, screen-linear depth and pixels per length unit at a joint (w: visibility); false if it is
// invisible or behind the camera.
bool projectPoint(vec4 value, out vec2 position, out float depth, out float pixelsPerUnit) {
    vec4 clip = ubo.proj * ubo.view * ubo.model * vec4(value.xyz, 1.0);
    position = (clip.xy / clip.w * 0.5 + 0.5) * pc.resolution;
    depth = perspective() ? 1.0 / clip.w : clip.z / clip.w;
    // The model matrix scales uniformly; the vertical focal length in pixels is |P[1][1]| * height / 2.
    float modelScale = length(ubo.model[0].xyz);
    pixelsPerUnit = modelScale * abs(ubo.proj[1][1]) * 0.5 * pc.resolution.y / clip.w;
    return value.w > 0.0 && clip.w > 0.0;
}

void main() {
    // Triangle-strip corners (-1, -1), (1, -1), (-1, 1), (1, 1).
    vec2 corner = vec2(gl_VertexIndex & 1, gl_VertexIndex >> 1) * 2.0 - 1.0;
    uint segment = uint(gl_InstanceIndex);
    projection = vec3(ubo.proj[3][2], ubo.proj[2][2], perspective() ? 1.0 : 0.0);

    vec2 first, second;
    float firstDepth, secondDepth, firstScale, secondScale;
    vec4 firstJoint = joints[SEGMENT_JOINTS[2u * segment]], secondJoint = joints[SEGMENT_JOINTS[2u * segment + 1u]];
    if (segment == HEAD_DISC) {
        // Both ends at the middle of the joints, visible only if both are.
        firstJoint = vec4(0.5 * (firstJoint.xyz + secondJoint.xyz), min(firstJoint.w, secondJoint.w));
        secondJoint = firstJoint;
    }
    bool visible = projectPoint(firstJoint, first, firstDepth, firstScale);
    visible = projectPoint(secondJoint, second, secondDepth, secondScale) && visible;
    if (!visible) {
        gl_Position = vec4(-2.0, -2.0, 0.0, 1.0);
        return;
    }
    float radius = SEGMENT_RADII[segment] + pc.margin;
    bone = vec4(first, second);
    radii = vec2(radius * firstScale, radius * secondScale);
    boneDepth = vec2(firstDepth, secondDepth);

    // A quad around the capsule, one pixel larger on every side.
    vec2 along = second - first;
    float boneLength = length(along);
    vec2 direction = boneLength > 1e-4 ? along / boneLength : vec2(1.0, 0.0);
    vec2 perpendicular = vec2(-direction.y, direction.x);
    float extent = max(radii.x, radii.y) + 1.0;
    vec2 centre = 0.5 * (first + second);
    vec2 position = centre + corner.x * (0.5 * boneLength + extent) * direction + corner.y * extent * perpendicular;
    gl_Position = vec4(position / pc.resolution * 2.0 - 1.0, 0.0, 1.0);
}
