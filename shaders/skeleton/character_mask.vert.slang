// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// One quad per segment of the character mask (instance = segment), around a
// capsule whose radius at each end is the segment's radius plus the margin,
// projected at that end's distance.
import character_mask_common;

struct UniformBufferObject {
    float4x4 model;
    float4x4 view;
    float4x4 proj;
    float4 cameraWorldPos;
};
[[vk::binding(0, 0)]] ConstantBuffer<UniformBufferObject> ubo;

[[vk::binding(0, 1)]] StructuredBuffer<float4> joints;

// Joint pairs and radii of characterMaskSegments() (keep in sync).
static const uint SEGMENT_COUNT = 21u;
static const uint SEGMENT_JOINTS[2 * SEGMENT_COUNT] = { 1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 1, 8, 8, 9, 9, 10, 1, 11,
                                                        11, 12, 12, 13, 1, 0, 0, 14, 14, 16, 0, 15, 15, 17, 2, 8, 5,
                                                        11, 8, 11, 16, 17 };
static const float SEGMENT_RADII[SEGMENT_COUNT] = { 0.07, 0.07, 0.07, 0.055, 0.07, 0.055, 0.14, 0.085, 0.065, 0.14,
                                                    0.085, 0.065, 0.06, 0.05, 0.05, 0.05, 0.05, 0.11, 0.11, 0.11, 0.1 };
// The last segment, the head, is a disc around the middle of its two joints (the ears).
static const uint HEAD_DISC = SEGMENT_COUNT - 1u;

// P[3][2] (row 3, column 2) is -1 for a perspective projection and 0 otherwise.
bool perspective() { return ubo.proj[3][2] != 0.0; }

// Position in pixels, screen-linear depth and pixels per length unit at a joint (w: visibility); false if it is
// invisible or behind the camera.
bool projectPoint(float4 value, out float2 position, out float depth, out float pixelsPerUnit) {
    float4 clip = mul(mul(mul(ubo.proj, ubo.view), ubo.model), float4(value.xyz, 1.0));
    position = (clip.xy / clip.w * 0.5 + 0.5) * pc.resolution;
    depth = perspective() ? 1.0 / clip.w : clip.z / clip.w;
    // The model matrix scales uniformly (the length of its first column); the vertical focal length in pixels is
    // |P[1][1]| * height / 2.
    float modelScale = length(float3(ubo.model[0][0], ubo.model[1][0], ubo.model[2][0]));
    pixelsPerUnit = modelScale * abs(ubo.proj[1][1]) * 0.5 * pc.resolution.y / clip.w;
    return value.w > 0.0 && clip.w > 0.0;
}

[shader("vertex")]
CharacterMaskVertex main(uint vertexIndex: SV_VulkanVertexID, uint instanceIndex: SV_VulkanInstanceID) {
    CharacterMaskVertex output = {};
    // Triangle-strip corners (-1, -1), (1, -1), (-1, 1), (1, 1).
    float2 corner = float2(vertexIndex & 1, vertexIndex >> 1) * 2.0 - 1.0;
    uint segment = instanceIndex;
    output.projection = float3(ubo.proj[2][3], ubo.proj[2][2], perspective() ? 1.0 : 0.0);

    float2 first, second;
    float firstDepth, secondDepth, firstScale, secondScale;
    float4 firstJoint = joints[SEGMENT_JOINTS[2u * segment]], secondJoint = joints[SEGMENT_JOINTS[2u * segment + 1u]];
    if (segment == HEAD_DISC) {
        // Both ends at the middle of the joints, visible only if both are.
        firstJoint = float4(0.5 * (firstJoint.xyz + secondJoint.xyz), min(firstJoint.w, secondJoint.w));
        secondJoint = firstJoint;
    }
    bool visible = projectPoint(firstJoint, first, firstDepth, firstScale);
    visible = projectPoint(secondJoint, second, secondDepth, secondScale) && visible;
    if (!visible) {
        output.position = float4(-2.0, -2.0, 0.0, 1.0);
        return output;
    }
    float radius = SEGMENT_RADII[segment] + pc.margin;
    output.bone = float4(first, second);
    output.radii = float2(radius * firstScale, radius * secondScale);
    output.boneDepth = float2(firstDepth, secondDepth);

    // A quad around the capsule, one pixel larger on every side.
    float2 along = second - first;
    float boneLength = length(along);
    float2 direction = boneLength > 1e-4 ? along / boneLength : float2(1.0, 0.0);
    float2 perpendicular = float2(-direction.y, direction.x);
    float extent = max(output.radii.x, output.radii.y) + 1.0;
    float2 centre = 0.5 * (first + second);
    float2 position = centre + corner.x * (0.5 * boneLength + extent) * direction + corner.y * extent * perpendicular;
    output.position = float4(position / pc.resolution * 2.0 - 1.0, 0.0, 1.0);
    return output;
}
