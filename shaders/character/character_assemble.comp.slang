// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Builds the motion and gating inputs of the controller network from the
// state and the control (controller.py: assemble). The past trajectory comes
// from the root history, the future trajectory blends the network's own
// prediction towards the control, increasingly with the time ahead.

import character_common;

[[vk::binding(0, 0)]] StructuredBuffer<float> state;
[[vk::binding(1, 0)]] StructuredBuffer<float> control;
[[vk::binding(2, 0)]] RWStructuredBuffer<float> motionInput;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> gatingInput;

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
    float2 root = float2(state[0], state[1]);
    float angle = state[2];
    uint style = uint(control[4]);
    float responsiveness = control[5];
    float2 velocity = toLocal(float2(control[0], control[1]), angle);
    float2 facing = normalizeOrForward(toLocal(float2(control[2], control[3]), angle));

    for (uint pastSample = 0u; pastSample < PAST_SAMPLES; ++pastSample) {
        uint entry = HISTORY_OFFSET + (HISTORY - 1u - (PAST_SAMPLES - 1u - pastSample) * TRAJECTORY_STEP) * 5u;
        uint base = pastSample * 6u;
        float2 position = toLocal(float2(state[entry], state[entry + 1u]) - root, angle);
        float2 direction = toLocal(float2(sin(state[entry + 2u]), cos(state[entry + 2u])), angle);
        float2 rootVelocity = toLocal(float2(state[entry + 3u], state[entry + 4u]), angle);
        motionInput[base + 0u] = position.x;
        motionInput[base + 1u] = position.y;
        motionInput[base + 2u] = direction.x;
        motionInput[base + 3u] = direction.y;
        motionInput[base + 4u] = rootVelocity.x;
        motionInput[base + 5u] = rootVelocity.y;
    }

    for (uint future = 0u; future < FUTURE_SAMPLES; ++future) {
        uint predicted = FUTURE_OFFSET + future * 6u;
        float time = float((future + 1u) * TRAJECTORY_STEP) / float(FPS);
        float weight = responsiveness * (float(future + 1u) / float(FUTURE_SAMPLES));
        float2 position = float2(state[predicted], state[predicted + 1u]);
        position += weight * (velocity * time - position);
        float2 direction = float2(state[predicted + 2u], state[predicted + 3u]);
        direction = normalizeOrForward(direction + weight * (facing - direction));
        float2 rootVelocity = float2(state[predicted + 4u], state[predicted + 5u]);
        rootVelocity += weight * (velocity - rootVelocity);
        uint base = (PAST_SAMPLES + future) * 6u;
        motionInput[base + 0u] = position.x;
        motionInput[base + 1u] = position.y;
        motionInput[base + 2u] = direction.x;
        motionInput[base + 3u] = direction.y;
        motionInput[base + 4u] = rootVelocity.x;
        motionInput[base + 5u] = rootVelocity.y;
        uint gatingBase = GATING_JOINT_COUNT * 3u + future * 4u;
        gatingInput[gatingBase + 0u] = direction.x;
        gatingInput[gatingBase + 1u] = direction.y;
        gatingInput[gatingBase + 2u] = rootVelocity.x;
        gatingInput[gatingBase + 3u] = rootVelocity.y;
    }

    for (uint i = 0u; i < POSE_SIZE; ++i) {
        motionInput[TRAJECTORY_SIZE + i] = state[POSE_OFFSET + i];
    }
    for (uint i = 0u; i < STYLE_COUNT; ++i) {
        float oneHot = i == style ? 1.0 : 0.0;
        motionInput[TRAJECTORY_SIZE + POSE_SIZE + i] = oneHot;
        gatingInput[GATING_JOINT_COUNT * 3u + FUTURE_SAMPLES * 4u + i] = oneHot;
    }
    for (uint i = 0u; i < GATING_JOINT_COUNT; ++i) {
        uint joint = POSE_OFFSET + GATING_JOINTS[i] * 6u;
        gatingInput[i * 3u + 0u] = state[joint + 3u];
        gatingInput[i * 3u + 1u] = state[joint + 4u];
        gatingInput[i * 3u + 2u] = state[joint + 5u];
    }
}
