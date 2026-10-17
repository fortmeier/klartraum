// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#version 450

// One quad per limb (drawJoints == 0, instance = limb) or per joint
// (drawJoints == 1, instance = joint) of one skeleton. Shape parameters follow
// OpenCV as called by ControlNet's draw_bodypose: positions in pixels where
// pixel i spans [i, i + 1), truncated to whole pixels; limb half length and
// angle truncated to whole pixels and degrees.

#include "openpose_common.glsl"

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 cameraWorldPos;
} ubo;

layout(std430, set = 1, binding = 0) readonly buffer Joints {
    vec4 joints[];
};

layout(location = 0) flat out vec2 center;    // framebuffer coordinates (pixel centres at .5)
layout(location = 1) flat out vec2 axes;      // ellipse semi-axes; x is the disc radius for joints
layout(location = 2) flat out vec2 direction; // major axis (cos, sin) of a limb
layout(location = 3) flat out vec3 color;
// For the per-pixel occlusion test in openpose.frag: the shape's bone in framebuffer
// coordinates (both ends at the joint for a joint disc), a depth at each end that is
// linear in screen space (1 / distance with a perspective projection, NDC depth
// otherwise), and the projection constants that turn the scene's NDC depth into it.
layout(location = 4) flat out vec4 bone;
layout(location = 5) flat out vec2 boneDepth;
layout(location = 6) flat out vec3 projection; // P[3][2], P[2][2], 1 if perspective

// Pairs of COCO-18 joint indices, see openpose::kLimbs.
const uint LIMBS[34] = uint[](1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 1, 8, 8, 9, 9, 10, 1, 11, 11, 12, 12, 13, 1, 0, 0,
                              14, 14, 16, 0, 15, 15, 17);

// See openpose::kJointColors.
const uvec3 COLORS[18] = uvec3[](uvec3(255, 0, 0), uvec3(255, 85, 0), uvec3(255, 170, 0), uvec3(255, 255, 0),
                                 uvec3(170, 255, 0), uvec3(85, 255, 0), uvec3(0, 255, 0), uvec3(0, 255, 85),
                                 uvec3(0, 255, 170), uvec3(0, 255, 255), uvec3(0, 170, 255), uvec3(0, 85, 255),
                                 uvec3(0, 0, 255), uvec3(85, 0, 255), uvec3(170, 0, 255), uvec3(255, 0, 255),
                                 uvec3(255, 0, 170), uvec3(255, 0, 85));

// OpenCV fills the outline of its ellipse polygons, too: an analytic ellipse
// enlarged by this many pixels covers nearly the same pixels.
const float ELLIPSE_COVERAGE_MARGIN = 0.6;

// Guards the truncation of a position against float error at whole pixels,
// e.g. a normalized 0.25 on a 256 pixel image.
const float TRUNCATION_EPSILON = 1e-3;

bool perspective() { return ubo.proj[2][3] != 0.0; }

const uint NOSE = 0u, RIGHT_EYE = 14u, LEFT_EYE = 15u, RIGHT_EAR = 16u, LEFT_EAR = 17u;
// How far (cosine) a face keypoint may turn away from the camera before it counts as hidden: slightly past the
// profile for nose and eyes, clearly behind the head for an ear or eye on the far side.
const float FACING_AWAY = -0.1;
const float FAR_SIDE = -0.3;

// Whether the head hides the face keypoint `joint` from the camera (see setHideOccludedFace()).
bool hiddenByHead(uint joint) {
    if (joint != NOSE && joint != RIGHT_EYE && joint != LEFT_EYE && joint != RIGHT_EAR && joint != LEFT_EAR) {
        return false;
    }
    vec4 rightEar = joints[pc.firstJoint + RIGHT_EAR], leftEar = joints[pc.firstJoint + LEFT_EAR];
    vec4 nose = joints[pc.firstJoint + NOSE];
    if (rightEar.w <= 0.0 || leftEar.w <= 0.0 || nose.w <= 0.0) {
        return false;
    }
    vec3 centre = 0.5 * (rightEar.xyz + leftEar.xyz);
    // The direction from the head to the camera, in the joints' space.
    mat4 cameraToJoints = inverse(ubo.view * ubo.model);
    vec3 towardsCamera = perspective() ? cameraToJoints[3].xyz - centre : (cameraToJoints * vec4(0.0, 0.0, -1.0, 0.0)).xyz;
    towardsCamera = normalize(towardsCamera);
    vec3 facing = nose.xyz - centre;
    if (length(facing) < 1e-6) {
        return false;
    }
    facing = normalize(facing);
    bool facingAway = dot(facing, towardsCamera) < FACING_AWAY;
    if (joint == NOSE) {
        return facingAway;
    }
    // The part of the keypoint's offset across the face: left or right of the head.
    vec3 offset = joints[pc.firstJoint + joint].xyz - centre;
    vec3 across = offset - dot(offset, facing) * facing;
    bool farSide = length(across) > 1e-6 && dot(normalize(across), towardsCamera) < FAR_SIDE;
    bool eye = joint == RIGHT_EYE || joint == LEFT_EYE;
    return farSide || (eye && facingAway);
}

// The joint's position in pixels and its screen-linear depth (see `boneDepth`);
// false if it is invisible or behind the camera.
bool projectJoint(uint joint, out vec2 position, out float depth) {
    vec4 value = joints[pc.firstJoint + joint];
    vec4 clip = ubo.proj * ubo.view * ubo.model * vec4(value.xyz, 1.0);
    position = (clip.xy / clip.w * 0.5 + 0.5) * pc.resolution;
    // With a perspective projection clip.w is the distance along the view axis.
    depth = perspective() ? 1.0 / clip.w : clip.z / clip.w;
    return value.w > 0.0 && clip.w > 0.0 && !(pc.hideOccludedFace != 0u && hiddenByHead(joint));
}

vec2 truncatePixel(vec2 position) { return trunc(position + sign(position) * TRUNCATION_EPSILON); }

void hide() {
    gl_Position = vec4(-2.0, -2.0, 0.0, 1.0);
}

void main() {
    // Triangle-strip corners (-1, -1), (1, -1), (-1, 1), (1, 1).
    vec2 corner = vec2(gl_VertexIndex & 1, gl_VertexIndex >> 1) * 2.0 - 1.0;
    uint index = uint(gl_InstanceIndex);
    vec2 halfExtent;

    projection = vec3(ubo.proj[3][2], ubo.proj[2][2], perspective() ? 1.0 : 0.0);
    if (pc.drawJoints != 0u) {
        vec2 position;
        float depth;
        if (!projectJoint(index, position, depth)) {
            hide();
            return;
        }
        bone = vec4(position, position);
        boneDepth = vec2(depth);
        center = truncatePixel(position) + 0.5;
        axes = vec2(pc.jointRadius);
        direction = vec2(1.0, 0.0);
        color = vec3(COLORS[index]) / 255.0;
        halfExtent = axes + 1.0;
    } else {
        vec2 first, second;
        float firstDepth, secondDepth;
        bool visible = projectJoint(LIMBS[2 * index], first, firstDepth);
        visible = projectJoint(LIMBS[2 * index + 1], second, secondDepth) && visible;
        if (!visible) {
            hide();
            return;
        }
        bone = vec4(first, second);
        boneDepth = vec2(firstDepth, secondDepth);
        vec2 delta = first - second;
        float angle = radians(trunc(degrees(atan(delta.y, delta.x))));
        center = truncatePixel(0.5 * (first + second)) + 0.5;
        axes = vec2(trunc(length(delta) * 0.5), pc.stickWidth) + ELLIPSE_COVERAGE_MARGIN;
        direction = vec2(cos(angle), sin(angle));
        color = vec3((COLORS[index] * 6u) / 10u) / 255.0;
        halfExtent = axes + 1.0;
    }

    vec2 perpendicular = vec2(-direction.y, direction.x);
    vec2 position = center + corner.x * halfExtent.x * direction + corner.y * halfExtent.y * perpendicular;
    gl_Position = vec4(position / pc.resolution * 2.0 - 1.0, 0.0, 1.0);
}
