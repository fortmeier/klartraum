// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Applies the network output to the state (controller.py: integrate): moves
// and turns the root, appends it to the history, blends the predicted pose with
// the previous one advanced by the predicted velocity, keeps the predicted
// future trajectory, and writes the joints in world space and as COCO-18
// OpenPose keypoints. One invocation updates the state in place.

import character_common;

[[vk::binding(0, 0)]] RWStructuredBuffer<float> state;
[[vk::binding(1, 0)]] StructuredBuffer<float> outputData;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> joints;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> openpose;

// Controller joint of each COCO-18 keypoint; the neck (keypoint 1) is the middle of the shoulders.
static const uint OPENPOSE_JOINTS[18] = { 5u, 0u, 10u, 11u, 12u, 13u, 14u, 15u, 16u, 17u, 18u, 20u, 21u, 22u,
                                          6u, 7u, 8u, 9u };
static const uint RIGHT_SHOULDER = 10u;
static const uint LEFT_SHOULDER = 13u;

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
    float2 root = float2(state[0], state[1]);
    float angle = state[2];
    float2 delta = toWorld(float2(outputData[0], outputData[1]), angle);
    float2 newRoot = root + delta;
    float newAngle = angle + outputData[2];

    for (uint i = 0u; i + 1u < HISTORY; ++i) {
        for (uint k = 0u; k < 5u; ++k) {
            state[HISTORY_OFFSET + i * 5u + k] = state[HISTORY_OFFSET + (i + 1u) * 5u + k];
        }
    }
    uint newest = HISTORY_OFFSET + (HISTORY - 1u) * 5u;
    state[newest + 0u] = newRoot.x;
    state[newest + 1u] = newRoot.y;
    state[newest + 2u] = newAngle;
    state[newest + 3u] = delta.x * float(FPS);
    state[newest + 4u] = delta.y * float(FPS);

    const float frameTime = 1.0 / float(FPS);
    for (uint joint = 0u; joint < JOINT_COUNT; ++joint) {
        uint base = POSE_OFFSET + joint * 6u;
        uint predicted = 3u + joint * 6u;
        float3 previous = float3(state[base], state[base + 1u], state[base + 2u]);
        float3 predictedPosition = float3(outputData[predicted], outputData[predicted + 1u], outputData[predicted + 2u]);
        float3 predictedVelocity =
            float3(outputData[predicted + 3u], outputData[predicted + 4u], outputData[predicted + 5u]);
        float2 previousWorld = root + toWorld(previous.xz, angle);
        float2 carriedXZ = toLocal(previousWorld - newRoot, newAngle);
        float3 carried = float3(carriedXZ.x, previous.y, carriedXZ.y) + predictedVelocity * frameTime;
        float3 position = predictedPosition + (1.0 - POSE_BLEND) * (carried - predictedPosition);
        int parent = CONSTRAINT_PARENTS[joint];
        if (parent >= 0) {
            uint parentBase = POSE_OFFSET + uint(parent) * 6u;
            float3 parentPosition = float3(state[parentBase], state[parentBase + 1u], state[parentBase + 2u]);
            float3 bone = position - parentPosition;
            float boneLength = length(bone);
            if (boneLength > 1e-6) {
                position = parentPosition + bone * (state[REST_OFFSET + joint] / boneLength);
            }
        }
        state[base + 0u] = position.x;
        state[base + 1u] = position.y;
        state[base + 2u] = position.z;
        state[base + 3u] = predictedVelocity.x;
        state[base + 4u] = predictedVelocity.y;
        state[base + 5u] = predictedVelocity.z;
        float2 world = newRoot + toWorld(position.xz, newAngle);
        joints[joint] = float4(world.x, position.y, world.y, 1.0);
    }

    uint futureStart = 3u + POSE_SIZE;
    for (uint i = 0u; i < FUTURE_SAMPLES * 6u; ++i) {
        state[FUTURE_OFFSET + i] = outputData[futureStart + i];
    }
    state[0] = newRoot.x;
    state[1] = newRoot.y;
    state[2] = newAngle;

    for (uint keypoint = 0u; keypoint < 18u; ++keypoint) {
        openpose[keypoint] = keypoint == 1u ? float4(0.5 * (joints[RIGHT_SHOULDER].xyz + joints[LEFT_SHOULDER].xyz), 1.0)
                                            : joints[OPENPOSE_JOINTS[keypoint]];
    }
}
