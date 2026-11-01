// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// One quad per limb (drawJoints == 0, instance = limb) or per joint
// (drawJoints == 1, instance = joint) of one skeleton. Shape parameters follow
// OpenCV as called by ControlNet's draw_bodypose: positions in pixels where
// pixel i spans [i, i + 1), truncated to whole pixels; limb half length and
// angle truncated to whole pixels and degrees.
import matrix_math;
import openpose_common;

struct UniformBufferObject {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
    float4 cameraWorldPos;
};
[[vk::binding(0, 0)]] ConstantBuffer<UniformBufferObject> ubo;

[[vk::binding(0, 1)]] StructuredBuffer<float4> joints;

// Pairs of COCO-18 joint indices, see openpose::kLimbs.
static const uint LIMBS[34] = { 1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 1, 8, 8, 9, 9, 10, 1, 11, 11, 12, 12, 13, 1, 0, 0,
                                14, 14, 16, 0, 15, 15, 17 };

// See openpose::kJointColors.
static const uint3 COLORS[18] = { uint3(255, 0, 0), uint3(255, 85, 0), uint3(255, 170, 0), uint3(255, 255, 0),
                                  uint3(170, 255, 0), uint3(85, 255, 0), uint3(0, 255, 0), uint3(0, 255, 85),
                                  uint3(0, 255, 170), uint3(0, 255, 255), uint3(0, 170, 255), uint3(0, 85, 255),
                                  uint3(0, 0, 255), uint3(85, 0, 255), uint3(170, 0, 255), uint3(255, 0, 255),
                                  uint3(255, 0, 170), uint3(255, 0, 85) };

// OpenCV fills the outline of its ellipse polygons, too: an analytic ellipse
// enlarged by this many pixels covers nearly the same pixels.
static const float ELLIPSE_COVERAGE_MARGIN = 0.6;

// Guards the truncation of a position against float error at whole pixels,
// e.g. a normalized 0.25 on a 256 pixel image.
static const float TRUNCATION_EPSILON = 1e-3;

// P[3][2] (row 3, column 2) is -1 for a perspective projection and 0 otherwise.
bool perspective() { return ubo.proj[3][2] != 0.0; }

static const uint NOSE = 0u, RIGHT_EYE = 14u, LEFT_EYE = 15u, RIGHT_EAR = 16u, LEFT_EAR = 17u;
// How far (cosine) a face keypoint may turn away from the camera before it counts as hidden: slightly past the
// profile for nose and eyes, clearly behind the head for an ear or eye on the far side.
static const float FACING_AWAY = -0.1;
static const float FAR_SIDE = -0.3;

// Whether the head hides the face keypoint `joint` from the camera (see setHideOccludedFace()).
bool hiddenByHead(uint joint) {
    if (joint != NOSE && joint != RIGHT_EYE && joint != LEFT_EYE && joint != RIGHT_EAR && joint != LEFT_EAR) {
        return false;
    }
    float4 rightEar = joints[pc.firstJoint + RIGHT_EAR], leftEar = joints[pc.firstJoint + LEFT_EAR];
    float4 nose = joints[pc.firstJoint + NOSE];
    if (rightEar.w <= 0.0 || leftEar.w <= 0.0 || nose.w <= 0.0) {
        return false;
    }
    float3 centre = 0.5 * (rightEar.xyz + leftEar.xyz);
    // The direction from the head to the camera, in the joints' space.
    float4x4 cameraToJoints = inverse(mul(ubo.view, ubo.model));
    // The camera position is the translation column of cameraToJoints.
    float3 cameraPosition = float3(cameraToJoints[0][3], cameraToJoints[1][3], cameraToJoints[2][3]);
    float3 towardsCamera =
        perspective() ? cameraPosition - centre : mul(cameraToJoints, float4(0.0, 0.0, -1.0, 0.0)).xyz;
    towardsCamera = normalize(towardsCamera);
    float3 facing = nose.xyz - centre;
    if (length(facing) < 1e-6) {
        return false;
    }
    facing = normalize(facing);
    bool facingAway = dot(facing, towardsCamera) < FACING_AWAY;
    if (joint == NOSE) {
        return facingAway;
    }
    // The part of the keypoint's offset across the face: left or right of the head.
    float3 offset = joints[pc.firstJoint + joint].xyz - centre;
    float3 across = offset - dot(offset, facing) * facing;
    bool farSide = length(across) > 1e-6 && dot(normalize(across), towardsCamera) < FAR_SIDE;
    bool eye = joint == RIGHT_EYE || joint == LEFT_EYE;
    return farSide || (eye && facingAway);
}

// The joint's position in pixels and its screen-linear depth (see `boneDepth`);
// false if it is invisible or behind the camera.
bool projectJoint(uint joint, out float2 position, out float depth) {
    float4 value = joints[pc.firstJoint + joint];
    float4 clip = mul(mul(mul(ubo.proj, ubo.view), ubo.model), float4(value.xyz, 1.0));
    position = (clip.xy / clip.w * 0.5 + 0.5) * pc.resolution;
    // With a perspective projection clip.w is the distance along the view axis.
    depth = perspective() ? 1.0 / clip.w : clip.z / clip.w;
    return value.w > 0.0 && clip.w > 0.0 && !(pc.hideOccludedFace != 0u && hiddenByHead(joint));
}

float2 truncatePixel(float2 position) { return trunc(position + sign(position) * TRUNCATION_EPSILON); }

[shader("vertex")]
OpenPoseVertex main(uint vertexIndex: SV_VulkanVertexID, uint instanceIndex: SV_VulkanInstanceID) {
    OpenPoseVertex output = {};
    // Triangle-strip corners (-1, -1), (1, -1), (-1, 1), (1, 1).
    float2 corner = float2(vertexIndex & 1, vertexIndex >> 1) * 2.0 - 1.0;
    uint index = instanceIndex;
    float2 halfExtent;

    output.projection = float3(ubo.proj[2][3], ubo.proj[2][2], perspective() ? 1.0 : 0.0);
    // A shape that is not drawn moves off screen.
    const float4 hidden = float4(-2.0, -2.0, 0.0, 1.0);
    if (pc.drawJoints != 0u) {
        float2 position;
        float depth;
        if (!projectJoint(index, position, depth)) {
            output.position = hidden;
            return output;
        }
        output.bone = float4(position, position);
        output.boneDepth = float2(depth);
        output.center = truncatePixel(position) + 0.5;
        output.axes = float2(pc.jointRadius);
        output.direction = float2(1.0, 0.0);
        output.color = float3(COLORS[index]) / 255.0;
        halfExtent = output.axes + 1.0;
    } else {
        float2 first, second;
        float firstDepth, secondDepth;
        bool visible = projectJoint(LIMBS[2 * index], first, firstDepth);
        visible = projectJoint(LIMBS[2 * index + 1], second, secondDepth) && visible;
        if (!visible) {
            output.position = hidden;
            return output;
        }
        output.bone = float4(first, second);
        output.boneDepth = float2(firstDepth, secondDepth);
        float2 delta = first - second;
        float angle = radians(trunc(degrees(atan2(delta.y, delta.x))));
        output.center = truncatePixel(0.5 * (first + second)) + 0.5;
        output.axes = float2(trunc(length(delta) * 0.5), pc.stickWidth) + ELLIPSE_COVERAGE_MARGIN;
        output.direction = float2(cos(angle), sin(angle));
        output.color = float3((COLORS[index] * 6u) / 10u) / 255.0;
        halfExtent = output.axes + 1.0;
    }

    float2 perpendicular = float2(-output.direction.y, output.direction.x);
    float2 position =
        output.center + corner.x * halfExtent.x * output.direction + corner.y * halfExtent.y * perpendicular;
    output.position = float4(position / pc.resolution * 2.0 - 1.0, 0.0, 1.0);
    return output;
}
