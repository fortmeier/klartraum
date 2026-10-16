# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Reference implementation of the controller loop that Klartraum runs on the GPU.

shaders/character/character_assemble.comp and character_integrate.comp
implement exactly these steps (keep in sync); the test fixtures written by
export_fixtures.py come from this file. All arithmetic is float32.

State buffer (floats):
    [0, 4)      root x, root z, root angle, unused
    [4, 159)    history of the last 31 frames, oldest first: x, z, angle, velocity x, velocity z (world)
    [159, 303)  pose of the current frame: per joint position xyz and velocity xyz in the root frame
    [303, 321)  predicted future trajectory (3 samples of position, direction, velocity xz; root frame)
    [321, 345)  rest length of the bone from each joint's constraint parent (constant)

Control buffer (floats): desired velocity x, z (world, m/s), desired facing x, z (world), style index,
responsiveness (0: follow the network's own prediction, 1: follow the control), unused, unused.
"""

from __future__ import annotations

import numpy as np

import features

HISTORY = 31
ROOT_OFFSET = 0
HISTORY_OFFSET = 4
POSE_OFFSET = HISTORY_OFFSET + HISTORY * 5
FUTURE_OFFSET = POSE_OFFSET + features.POSE_SIZE
REST_OFFSET = FUTURE_OFFSET + len(features.FUTURE_OFFSETS) * 6
STATE_SIZE = REST_OFFSET + features.JOINT_COUNT
CONTROL_SIZE = 8

# Fraction of the network's predicted position fed back, the rest comes from
# the previous position advanced by the predicted velocity (as in PFNN).
POSE_BLEND = np.float32(0.5)

# The network predicts free joint positions, which lets limbs stretch or shrink
# a little. After each step, every joint with a constraint parent is moved
# along its bone to the bone's rest length, parents before children. The
# trunk joints have none: the reduced skeleton leaves out the collar and spine
# joints, so their distances are not rigid.
CONSTRAINT_PARENTS = {
    "HeadTop": "Head", "Nose": "Head", "RightEye": "Head", "LeftEye": "Head", "RightEar": "Head", "LeftEar": "Head",
    "RightElbow": "RightShoulder", "RightWrist": "RightElbow", "LeftElbow": "LeftShoulder", "LeftWrist": "LeftElbow",
    "RightKnee": "RightHip", "RightAnkle": "RightKnee", "RightToe": "RightAnkle",
    "LeftKnee": "LeftHip", "LeftAnkle": "LeftKnee", "LeftToe": "LeftAnkle",
}
PARENTS = [features.JOINTS.index(CONSTRAINT_PARENTS[name]) if name in CONSTRAINT_PARENTS else -1
           for name in features.JOINTS]

# COCO-18 keypoints from controller joints; the neck is the middle of the shoulders.
OPENPOSE_JOINTS = [
    "Nose", None, "RightShoulder", "RightElbow", "RightWrist", "LeftShoulder", "LeftElbow", "LeftWrist",
    "RightHip", "RightKnee", "RightAnkle", "LeftHip", "LeftKnee", "LeftAnkle", "RightEye", "LeftEye",
    "RightEar", "LeftEar",
]

f32 = np.float32


def to_local(x, z, angle):
    c, s = np.cos(angle, dtype=f32), np.sin(angle, dtype=f32)
    return f32(x * c - z * s), f32(x * s + z * c)


def to_world(x, z, angle):
    c, s = np.cos(angle, dtype=f32), np.sin(angle, dtype=f32)
    return f32(x * c + z * s), f32(-x * s + z * c)


def normalize(x, z):
    length = np.sqrt(f32(x * x + z * z), dtype=f32)
    if length < f32(1e-6):
        return f32(0.0), f32(1.0)
    return f32(x / length), f32(z / length)


def assemble(state: np.ndarray, control: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    root_x, root_z, angle = state[0], state[1], state[2]
    motion = np.zeros(features.MOTION_INPUT_SIZE, dtype=f32)
    gating = np.zeros(features.GATING_INPUT_SIZE, dtype=f32)
    style = int(control[4])
    responsiveness = control[5]
    velocity_x, velocity_z = to_local(control[0], control[1], angle)
    facing_x, facing_z = normalize(*to_local(control[2], control[3], angle))

    past = [offset for offset in features.TRAJECTORY_OFFSETS if offset <= 0]
    for sample, offset in enumerate(past):
        entry = state[HISTORY_OFFSET + (HISTORY - 1 + offset) * 5 :][:5]
        base = sample * 6
        motion[base + 0], motion[base + 1] = to_local(entry[0] - root_x, entry[1] - root_z, angle)
        motion[base + 2], motion[base + 3] = to_local(np.sin(entry[2], dtype=f32), np.cos(entry[2], dtype=f32), angle)
        motion[base + 4], motion[base + 5] = to_local(entry[3], entry[4], angle)

    for future, offset in enumerate(features.FUTURE_OFFSETS):
        predicted = state[FUTURE_OFFSET + future * 6 :][:6]
        time = f32(offset / features.FPS)
        weight = f32(responsiveness * f32((future + 1) / len(features.FUTURE_OFFSETS)))
        base = (len(past) + future) * 6
        motion[base + 0] = predicted[0] + weight * (velocity_x * time - predicted[0])
        motion[base + 1] = predicted[1] + weight * (velocity_z * time - predicted[1])
        motion[base + 2], motion[base + 3] = normalize(
            predicted[2] + weight * (facing_x - predicted[2]), predicted[3] + weight * (facing_z - predicted[3])
        )
        motion[base + 4] = predicted[4] + weight * (velocity_x - predicted[4])
        motion[base + 5] = predicted[5] + weight * (velocity_z - predicted[5])
        gating_base = len(features.GATING_JOINTS) * 3 + future * 4
        gating[gating_base : gating_base + 4] = motion[base + 2 : base + 6]

    pose = state[POSE_OFFSET : POSE_OFFSET + features.POSE_SIZE]
    motion[features.TRAJECTORY_SIZE : features.TRAJECTORY_SIZE + features.POSE_SIZE] = pose
    motion[features.TRAJECTORY_SIZE + features.POSE_SIZE + style] = 1.0
    for index, name in enumerate(features.GATING_JOINTS):
        joint = features.JOINTS.index(name)
        gating[index * 3 : index * 3 + 3] = pose[joint * 6 + 3 : joint * 6 + 6]
    gating[len(features.GATING_JOINTS) * 3 + len(features.FUTURE_OFFSETS) * 4 + style] = 1.0
    return motion, gating


def integrate(state: np.ndarray, output: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Next state, world joints [JOINT_COUNT, 3] and OpenPose joints [18, 4]."""
    next_state = state.copy()
    root_x, root_z, angle = state[0], state[1], state[2]
    delta_x, delta_z = to_world(output[0], output[1], angle)
    new_x, new_z = f32(root_x + delta_x), f32(root_z + delta_z)
    new_angle = f32(angle + output[2])
    next_state[0], next_state[1], next_state[2] = new_x, new_z, new_angle

    history = state[HISTORY_OFFSET:POSE_OFFSET].reshape(HISTORY, 5)
    shifted = np.concatenate([history[1:], [[new_x, new_z, new_angle, delta_x * f32(features.FPS),
                                             delta_z * f32(features.FPS)]]]).astype(f32)
    next_state[HISTORY_OFFSET:POSE_OFFSET] = shifted.ravel()

    world = np.zeros((features.JOINT_COUNT, 3), dtype=f32)
    for joint in range(features.JOINT_COUNT):
        previous = state[POSE_OFFSET + joint * 6 :][:3]
        predicted = output[3 + joint * 6 :][:6]
        # The previous position in the new root frame, advanced by the predicted velocity.
        world_x, world_z = to_world(previous[0], previous[2], angle)
        carried_x, carried_z = to_local(root_x + world_x - new_x, root_z + world_z - new_z, new_angle)
        frame_time = f32(1.0 / features.FPS)
        carried = np.array(
            [carried_x + predicted[3] * frame_time, previous[1] + predicted[4] * frame_time,
             carried_z + predicted[5] * frame_time],
            dtype=f32,
        )
        position = (predicted[:3] + (f32(1.0) - POSE_BLEND) * (carried - predicted[:3])).astype(f32)
        parent = PARENTS[joint]
        if parent >= 0:
            parent_position = next_state[POSE_OFFSET + parent * 6 :][:3]
            bone = (position - parent_position).astype(f32)
            length = np.sqrt(np.sum(bone * bone, dtype=f32), dtype=f32)
            if length > f32(1e-6):
                position = (parent_position + bone * (state[REST_OFFSET + joint] / length)).astype(f32)
        base = POSE_OFFSET + joint * 6
        next_state[base : base + 3] = position
        next_state[base + 3 : base + 6] = predicted[3:6]
        world_x, world_z = to_world(position[0], position[2], new_angle)
        world[joint] = (new_x + world_x, position[1], new_z + world_z)

    future_start = 3 + features.POSE_SIZE
    next_state[FUTURE_OFFSET:REST_OFFSET] = output[future_start : future_start + len(features.FUTURE_OFFSETS) * 6]
    return next_state, world, openpose(world)


def openpose(world: np.ndarray) -> np.ndarray:
    result = np.ones((18, 4), dtype=f32)
    for keypoint, name in enumerate(OPENPOSE_JOINTS):
        if name is None:
            right = world[features.JOINTS.index("RightShoulder")]
            left = world[features.JOINTS.index("LeftShoulder")]
            result[keypoint, :3] = f32(0.5) * (right + left)
        else:
            result[keypoint, :3] = world[features.JOINTS.index(name)]
    return result


def initial_state(positions: np.ndarray, frame: int) -> np.ndarray:
    """State of a dataset frame, with the root moved to the origin and facing +z."""
    root, angle = features.root_trajectory(positions)
    origin, origin_angle = root[frame].copy(), angle[frame]
    state = np.zeros(STATE_SIZE, dtype=f32)
    root_velocity = np.zeros_like(root)
    root_velocity[1:] = (root[1:] - root[:-1]) * features.FPS
    for index in range(HISTORY):
        sample = frame - (HISTORY - 1) + index
        x, z = features.to_local(root[sample] - origin, origin_angle)
        vx, vz = features.to_local(root_velocity[sample], origin_angle)
        state[HISTORY_OFFSET + index * 5 : HISTORY_OFFSET + index * 5 + 5] = (
            x, z, features.wrap_angle(angle[sample] - origin_angle), vx, vz)
    x, z, a = state[HISTORY_OFFSET + (HISTORY - 1) * 5 :][:3]
    state[0], state[1], state[2] = x, z, a

    _, _, outputs = features.frame_features(positions, 0)
    # frame_features starts at the first frame with a full past trajectory; its output for frame - 1
    # holds the pose and future trajectory of `frame`.
    output = outputs[frame - 1 + features.TRAJECTORY_OFFSETS[0]]
    state[POSE_OFFSET:REST_OFFSET] = output[3:]
    for joint, parent in enumerate(PARENTS):
        if parent >= 0:
            state[REST_OFFSET + joint] = np.mean(np.linalg.norm(positions[:, joint] - positions[:, parent], axis=1))
    return state


def rollout(session, state: np.ndarray, controls: np.ndarray):
    """Runs the controller for len(controls) frames; returns per-frame inputs, outputs and joints."""
    records = {"motion_inputs": [], "gating_inputs": [], "outputs": [], "joints": [], "openpose": []}
    for control in controls:
        motion, gating = assemble(state, control)
        output = session.run(None, {"motion_input": motion[None], "gating_input": gating[None]})[0][0].astype(f32)
        state, world, keypoints = integrate(state, output)
        for name, value in zip(records, (motion, gating, output, world, keypoints)):
            records[name].append(value)
    return state, {name: np.array(value, dtype=f32) for name, value in records.items()}
