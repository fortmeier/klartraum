# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Write the controller's initial state and reference rollouts for Klartraum.

Runs controller.py with ONNX Runtime on a scripted control sequence (stand,
walk forward, turn right) and writes, next to the model:

    initial_state_f32.bin        state of a Neutral idle frame (controller.STATE_SIZE floats)
    rollout_controls_f32.bin     [frames, CONTROL_SIZE]
    rollout_motion_inputs_f32.bin, rollout_gating_inputs_f32.bin, rollout_outputs_f32.bin
    rollout_joints_f32.bin       [frames, JOINT_COUNT, 3] world positions in metres
    rollout_openpose_f32.bin     [frames, 18, 4] COCO-18 keypoints (w = visibility)

It also prints plausibility metrics of the rollout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort

import bvh
import controller
import features

ROOT = Path(__file__).resolve().parents[2]
INITIAL_CLIP = "Neutral/Neutral_ID.bvh"
INITIAL_FRAME = 120


def scripted_controls(frames: int, speed: float = 1.3, style: str = "Neutral") -> np.ndarray:
    """Stand for a third of a second, walk forward, then turn right towards +x while walking."""
    controls = np.zeros((frames, controller.CONTROL_SIZE), dtype=np.float32)
    walk_start, turn_start = frames // 5, frames // 2
    for frame in range(frames):
        heading = 0.0
        if frame >= turn_start:
            heading = 0.5 * np.pi * min(1.0, (frame - turn_start) / max(1, frames - turn_start - 1))
        facing = np.array([np.sin(heading), np.cos(heading)])
        velocity = facing * speed if frame >= walk_start else np.zeros(2)
        controls[frame] = (velocity[0], velocity[1], facing[0], facing[1], features.STYLES.index(style), 0.6, 0, 0)
    return controls


def plausibility(joints: np.ndarray, controls: np.ndarray) -> dict[str, float]:
    index = features.JOINTS.index
    bones = [("RightHip", "RightKnee"), ("RightKnee", "RightAnkle"), ("LeftHip", "LeftKnee"),
             ("LeftKnee", "LeftAnkle"), ("RightShoulder", "RightElbow"), ("RightElbow", "RightWrist"),
             ("LeftShoulder", "LeftElbow"), ("LeftElbow", "LeftWrist")]
    lengths = np.stack([np.linalg.norm(joints[:, index(a)] - joints[:, index(b)], axis=1) for a, b in bones], 1)
    bone_deviation = float(np.max(np.abs(lengths / lengths[:1] - 1.0)))
    sliding = []
    for toe in ("RightToe", "LeftToe"):
        heights = joints[:, index(toe), 1]
        horizontal = np.linalg.norm(np.diff(joints[:, index(toe), :][:, [0, 2]], axis=0), axis=1)
        contact = heights[1:] < np.min(heights) + 0.03
        if contact.any():
            sliding.append(float(np.mean(horizontal[contact])))
    hips = joints[:, index("Hips")][:, [0, 2]]
    walking = np.linalg.norm(controls[:, :2], axis=1) > 0
    speed = np.linalg.norm(np.diff(hips, axis=0), axis=1) * features.FPS
    last = slice(len(controls) // 2, None)
    return {
        "max bone length deviation": bone_deviation,
        "mean toe sliding in contact (m/frame)": max(sliding) if sliding else float("nan"),
        "mean speed while walking, second half (m/s)": float(np.mean(speed[last][walking[1:][last]])),
        "commanded speed (m/s)": float(np.max(np.linalg.norm(controls[:, :2], axis=1))),
        "final hips height (m)": float(joints[-1, index("Hips"), 1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "data" / "character")
    parser.add_argument("--data", type=Path, default=ROOT / "build" / "character_animation" / "100STYLE")
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--long-frames", type=int, default=600, help="frames of the longer stability rollout")
    args = parser.parse_args()

    positions = features.joint_positions(bvh.read(args.data / INITIAL_CLIP))
    state = controller.initial_state(positions, INITIAL_FRAME)
    session = ort.InferenceSession(str(args.model_dir / "controller.onnx"), providers=["CPUExecutionProvider"])
    controls = scripted_controls(args.frames)
    _, records = controller.rollout(session, state, controls)

    state.tofile(args.model_dir / "initial_state_f32.bin")
    controls.tofile(args.model_dir / "rollout_controls_f32.bin")
    for name, values in records.items():
        values.tofile(args.model_dir / f"rollout_{name}_f32.bin")
    print(f"wrote fixtures for {args.frames} frames to {args.model_dir}")
    for name, value in plausibility(records["joints"], controls).items():
        print(f"  {name}: {value:.4f}")

    long_controls = scripted_controls(args.long_frames)
    _, long_records = controller.rollout(session, state, long_controls)
    print(f"{args.long_frames}-frame rollout:")
    for name, value in plausibility(long_records["joints"], long_controls).items():
        print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()
