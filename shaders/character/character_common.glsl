// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Layout of the character controller's features and state. Mirrors
// scripts/character_animation/features.py and controller.py (keep in sync).

const uint FPS = 30u;
const uint JOINT_COUNT = 24u;
const uint STYLE_COUNT = 12u;
const uint TRAJECTORY_SAMPLES = 7u; // frames -30..30 in steps of 10; the last FUTURE_SAMPLES are in the future
const uint PAST_SAMPLES = 4u;
const uint FUTURE_SAMPLES = 3u;
const uint TRAJECTORY_STEP = 10u;
const uint TRAJECTORY_SIZE = TRAJECTORY_SAMPLES * 6u;
const uint POSE_SIZE = JOINT_COUNT * 6u;
const uint GATING_JOINT_COUNT = 6u;

const uint HISTORY = 31u;
const uint HISTORY_OFFSET = 4u;
const uint POSE_OFFSET = HISTORY_OFFSET + HISTORY * 5u;
const uint FUTURE_OFFSET = POSE_OFFSET + POSE_SIZE;
const uint REST_OFFSET = FUTURE_OFFSET + FUTURE_SAMPLES * 6u;
const uint STATE_SIZE = REST_OFFSET + JOINT_COUNT;

// Right ankle, left ankle, right toe, left toe, right wrist, left wrist.
const uint GATING_JOINTS[GATING_JOINT_COUNT] = uint[](18u, 22u, 19u, 23u, 12u, 15u);

// Joint whose bone to each joint keeps its rest length (stored at REST_OFFSET);
// -1: unconstrained. Parents come before their children.
const int CONSTRAINT_PARENTS[JOINT_COUNT] = int[](-1, -1, -1, -1, 3, 3, 3, 3, 3, 3, -1, 10, 11, -1, 13, 14, -1, 16, 17,
                                                   18, -1, 20, 21, 22);

// Fraction of the predicted position that is fed back; the rest is the
// previous position advanced by the predicted velocity.
const float POSE_BLEND = 0.5;

// Rotates a world (x, z) vector into the root frame of facing angle `angle`.
vec2 toLocal(vec2 v, float angle) {
    float c = cos(angle), s = sin(angle);
    return vec2(v.x * c - v.y * s, v.x * s + v.y * c);
}

vec2 toWorld(vec2 v, float angle) {
    float c = cos(angle), s = sin(angle);
    return vec2(v.x * c + v.y * s, -v.x * s + v.y * c);
}

vec2 normalizeOrForward(vec2 v) {
    float len = length(v);
    return len < 1e-6 ? vec2(0.0, 1.0) : v / len;
}
