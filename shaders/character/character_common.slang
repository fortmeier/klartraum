// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Layout of the character controller's features and state. Mirrors
// scripts/character_animation/features.py and controller.py (keep in sync).
module character_common;

public static const uint FPS = 30u;
public static const uint JOINT_COUNT = 24u;
public static const uint STYLE_COUNT = 12u;
public static const uint TRAJECTORY_SAMPLES = 7u; // frames -30..30 in steps of 10; the last FUTURE_SAMPLES are in the future
public static const uint PAST_SAMPLES = 4u;
public static const uint FUTURE_SAMPLES = 3u;
public static const uint TRAJECTORY_STEP = 10u;
public static const uint TRAJECTORY_SIZE = TRAJECTORY_SAMPLES * 6u;
public static const uint POSE_SIZE = JOINT_COUNT * 6u;
public static const uint GATING_JOINT_COUNT = 6u;

public static const uint HISTORY = 31u;
public static const uint HISTORY_OFFSET = 4u;
public static const uint POSE_OFFSET = HISTORY_OFFSET + HISTORY * 5u;
public static const uint FUTURE_OFFSET = POSE_OFFSET + POSE_SIZE;
public static const uint REST_OFFSET = FUTURE_OFFSET + FUTURE_SAMPLES * 6u;
public static const uint STATE_SIZE = REST_OFFSET + JOINT_COUNT;

// Right ankle, left ankle, right toe, left toe, right wrist, left wrist.
public static const uint GATING_JOINTS[GATING_JOINT_COUNT] = { 18u, 22u, 19u, 23u, 12u, 15u };

// Joint whose bone to each joint keeps its rest length (stored at REST_OFFSET);
// -1: unconstrained. Parents come before their children.
public static const int CONSTRAINT_PARENTS[JOINT_COUNT] = { -1, -1, -1, -1, 3, 3, 3, 3, 3, 3, -1, 10, 11, -1, 13, 14,
                                                            -1, 16, 17, 18, -1, 20, 21, 22 };

// Fraction of the predicted position that is fed back; the rest is the
// previous position advanced by the predicted velocity.
public static const float POSE_BLEND = 0.5;

// Rotates a world (x, z) vector into the root frame of facing angle `angle`.
public float2 toLocal(float2 v, float angle) {
    float c = cos(angle), s = sin(angle);
    return float2(v.x * c - v.y * s, v.x * s + v.y * c);
}

public float2 toWorld(float2 v, float angle) {
    float c = cos(angle), s = sin(angle);
    return float2(v.x * c + v.y * s, -v.x * s + v.y * c);
}

public float2 normalizeOrForward(float2 v) {
    float len = length(v);
    return len < 1e-6 ? float2(0.0, 1.0) : v / len;
}
