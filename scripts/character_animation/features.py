# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Feature layout of the character controller and extraction from 100STYLE.

The controller predicts the next pose of a character from its current pose
and a trajectory of its root, like the phase-functioned and mode-adaptive
networks of Holden, Zhang, Starke et al. Everything is expressed in the root
frame of the character: the hips projected onto the ground, facing the
smoothed direction perpendicular to hips and shoulders. Units are metres and
seconds at 30 frames per second.

The same layout is implemented by shaders/character/*.comp (keep in sync).

Motion input (frame i):   trajectory 7 x (pos.xz, dir.xz, vel.xz) at frames i-30..i+30 in steps of 10,
                          joint positions and velocities (24 x 6), style one-hot (12)
Gating input (frame i):   velocities of ankles, toes and wrists (6 x 3), trajectory dir.xz and vel.xz of
                          the three future samples (3 x 4), style one-hot (12)
Output (frame i+1):       root delta (dx, dz, dangle) in the root frame of frame i, joint positions and
                          velocities (24 x 6) and the future trajectory (3 x 6) in the root frame of i+1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import bvh

FPS = 30
SOURCE_FPS = 60
TRAJECTORY_OFFSETS = [-30, -20, -10, 0, 10, 20, 30]
FUTURE_OFFSETS = [10, 20, 30]
STYLES = [
    "Neutral", "Proud", "Old", "Elated", "Strutting", "Rushed", "BigSteps", "Heavyset", "ArmsBySide", "Angry",
    "Depressed", "March",
]

# Joints the network predicts. Nose, eyes and ears are virtual joints at fixed
# offsets in the frame of the BVH Head joint (+z forward, +y up, +x left; cm).
JOINTS = [
    "Hips", "Chest2", "Neck", "Head", "HeadTop", "Nose", "RightEye", "LeftEye", "RightEar", "LeftEar",
    "RightShoulder", "RightElbow", "RightWrist", "LeftShoulder", "LeftElbow", "LeftWrist",
    "RightHip", "RightKnee", "RightAnkle", "RightToe", "LeftHip", "LeftKnee", "LeftAnkle", "LeftToe",
]
VIRTUAL_HEAD_OFFSETS = {
    "Nose": (0.0, 8.0, 10.5),
    "RightEye": (-3.2, 10.5, 8.5),
    "LeftEye": (3.2, 10.5, 8.5),
    "RightEar": (-7.5, 8.5, 0.0),
    "LeftEar": (7.5, 8.5, 0.0),
}
GATING_JOINTS = ["RightAnkle", "LeftAnkle", "RightToe", "LeftToe", "RightWrist", "LeftWrist"]

JOINT_COUNT = len(JOINTS)
STYLE_COUNT = len(STYLES)
TRAJECTORY_SIZE = len(TRAJECTORY_OFFSETS) * 6
POSE_SIZE = JOINT_COUNT * 6
MOTION_INPUT_SIZE = TRAJECTORY_SIZE + POSE_SIZE + STYLE_COUNT
GATING_INPUT_SIZE = len(GATING_JOINTS) * 3 + len(FUTURE_OFFSETS) * 4 + STYLE_COUNT
OUTPUT_SIZE = 3 + POSE_SIZE + len(FUTURE_OFFSETS) * 6


def mirror_indices() -> list[int]:
    def swap(name: str) -> str:
        if name.startswith("Right"):
            return "Left" + name[5:]
        if name.startswith("Left"):
            return "Right" + name[4:]
        return name

    return [JOINTS.index(swap(name)) for name in JOINTS]


def to_local(vectors: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Rotates world xz vectors [..., 2] into the root frame of facing angle (facing = (sin, cos))."""
    c, s = np.cos(angle), np.sin(angle)
    x, z = vectors[..., 0], vectors[..., 1]
    return np.stack([x * c - z * s, x * s + z * c], axis=-1)


def to_world(vectors: np.ndarray, angle: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    x, z = vectors[..., 0], vectors[..., 1]
    return np.stack([x * c + z * s, -x * s + z * c], axis=-1)


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def joint_positions(motion: bvh.Motion) -> np.ndarray:
    """World positions in metres [frames, JOINT_COUNT, 3] at FPS."""
    skeleton = motion.skeleton
    positions, rotations = bvh.forward_kinematics(motion)
    head = skeleton.index("Head")
    result = []
    for name in JOINTS:
        if name in skeleton.names:
            result.append(positions[:, skeleton.index(name)])
        else:
            offset = skeleton.end_sites["Head"] if name == "HeadTop" else np.array(VIRTUAL_HEAD_OFFSETS[name])
            result.append(positions[:, head] + np.einsum("fij,j->fi", rotations[:, head], offset))
    step = SOURCE_FPS // FPS
    return np.stack(result, axis=1)[::step] / 100.0


def gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    radius = int(3 * sigma)
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.concatenate([np.repeat(values[:1], radius, 0), values, np.repeat(values[-1:], radius, 0)])
    return np.stack([np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(values.shape[1])], axis=1)


def root_trajectory(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Root position xz [frames, 2] and facing angle [frames]."""
    index = JOINTS.index
    across = (positions[:, index("LeftHip")] - positions[:, index("RightHip")]) + (
        positions[:, index("LeftShoulder")] - positions[:, index("RightShoulder")]
    )
    forward = np.cross(across, np.array([0.0, 1.0, 0.0]))[:, [0, 2]]
    forward = gaussian_smooth(forward, 3.0)
    angle = np.arctan2(forward[:, 0], forward[:, 1])
    return positions[:, index("Hips"), :][:, [0, 2]], angle


def frame_features(positions: np.ndarray, style: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Motion inputs, gating inputs and outputs for every usable frame of one clip."""
    root, angle = root_trajectory(positions)
    velocity_world = np.zeros_like(positions)
    velocity_world[1:] = (positions[1:] - positions[:-1]) * FPS
    velocity_world[0] = velocity_world[1]
    root_velocity = np.zeros_like(root)
    root_velocity[1:] = (root[1:] - root[:-1]) * FPS
    root_velocity[0] = root_velocity[1]
    one_hot = np.zeros(STYLE_COUNT)
    one_hot[style] = 1.0

    def pose(frame: int) -> np.ndarray:
        local = positions[frame].copy()
        local[:, [0, 2]] = to_local(local[:, [0, 2]] - root[frame], angle[frame])
        velocity = velocity_world[frame].copy()
        velocity[:, [0, 2]] = to_local(velocity[:, [0, 2]], angle[frame])
        return np.concatenate([local, velocity], axis=1)  # [JOINT_COUNT, 6]

    def trajectory(frame: int, offsets: list[int]) -> np.ndarray:
        samples = []
        for offset in offsets:
            sample = frame + offset
            position = to_local(root[sample] - root[frame], angle[frame])
            direction = to_local(np.array([np.sin(angle[sample]), np.cos(angle[sample])]), angle[frame])
            velocity = to_local(root_velocity[sample], angle[frame])
            samples.append(np.concatenate([position, direction, velocity]))
        return np.stack(samples)  # [len(offsets), 6]

    motion_inputs, gating_inputs, outputs = [], [], []
    first = -TRAJECTORY_OFFSETS[0]
    last = len(positions) - 1 - TRAJECTORY_OFFSETS[-1] - 1
    gating = [JOINTS.index(name) for name in GATING_JOINTS]
    for frame in range(first, last):
        current = pose(frame)
        current_trajectory = trajectory(frame, TRAJECTORY_OFFSETS)
        motion_inputs.append(np.concatenate([current_trajectory.ravel(), current.ravel(), one_hot]))
        future = current_trajectory[-len(FUTURE_OFFSETS) :, 2:6]
        gating_inputs.append(np.concatenate([current[gating, 3:6].ravel(), future.ravel(), one_hot]))
        delta = to_local(root[frame + 1] - root[frame], angle[frame])
        rotation = wrap_angle(angle[frame + 1] - angle[frame])
        outputs.append(
            np.concatenate(
                [delta, [rotation], pose(frame + 1).ravel(), trajectory(frame + 1, FUTURE_OFFSETS).ravel()]
            )
        )
    return np.array(motion_inputs), np.array(gating_inputs), np.array(outputs)


def mirror(positions: np.ndarray) -> np.ndarray:
    mirrored = positions[:, mirror_indices()].copy()
    mirrored[..., 0] *= -1.0
    return mirrored


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the training database from 100STYLE BVH files.")
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--data", type=Path, default=root / "build" / "character_animation" / "100STYLE")
    parser.add_argument("--output", type=Path, default=root / "build" / "character_animation" / "database.npz")
    args = parser.parse_args()

    motion_inputs, gating_inputs, outputs, clips = [], [], [], []
    for style, name in enumerate(STYLES):
        for path in sorted((args.data / name).glob("*.bvh")):
            positions = joint_positions(bvh.read(path))
            for variant in (positions, mirror(positions)):
                x, g, y = frame_features(variant, style)
                motion_inputs.append(x)
                gating_inputs.append(g)
                outputs.append(y)
                clips.append(len(x))
            print(f"{path.name}: {len(positions)} frames")
    np.savez_compressed(
        args.output,
        motion_inputs=np.concatenate(motion_inputs).astype(np.float32),
        gating_inputs=np.concatenate(gating_inputs).astype(np.float32),
        outputs=np.concatenate(outputs).astype(np.float32),
        clip_lengths=np.array(clips),
    )
    print(f"wrote {args.output}: {sum(clips)} samples")


if __name__ == "__main__":
    main()
