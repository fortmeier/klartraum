# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Minimal BVH reader with forward kinematics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Skeleton:
    names: list[str] = field(default_factory=list)
    parents: list[int] = field(default_factory=list)
    offsets: list[np.ndarray] = field(default_factory=list)
    channels: list[list[str]] = field(default_factory=list)
    end_sites: dict[str, np.ndarray] = field(default_factory=dict)  # joint name -> end offset

    def index(self, name: str) -> int:
        return self.names.index(name)


@dataclass
class Motion:
    skeleton: Skeleton
    frame_time: float
    values: np.ndarray  # [frames, channels]


def read(path: Path) -> Motion:
    tokens = path.read_text().split()
    skeleton = Skeleton()
    position = 0
    stack: list[int] = []
    last_joint = -1
    pending_end = False
    while tokens[position] != "MOTION":
        token = tokens[position]
        if token in ("ROOT", "JOINT"):
            skeleton.names.append(tokens[position + 1])
            skeleton.parents.append(stack[-1] if stack else -1)
            skeleton.channels.append([])
            skeleton.offsets.append(np.zeros(3))
            last_joint = len(skeleton.names) - 1
            position += 2
        elif token == "End":
            pending_end = True
            position += 2
        elif token == "{":
            stack.append(last_joint if not pending_end else -2)
            position += 1
        elif token == "}":
            stack.pop()
            pending_end = False
            position += 1
        elif token == "OFFSET":
            offset = np.array([float(value) for value in tokens[position + 1 : position + 4]])
            if pending_end:
                skeleton.end_sites[skeleton.names[[s for s in stack if s >= 0][-1]]] = offset
            else:
                skeleton.offsets[last_joint] = offset
            position += 4
        elif token == "CHANNELS":
            count = int(tokens[position + 1])
            skeleton.channels[last_joint] = tokens[position + 2 : position + 2 + count]
            position += 2 + count
        else:
            position += 1
    frames = int(tokens[position + 2])
    frame_time = float(tokens[position + 5])
    values = np.array(tokens[position + 6 :], dtype=np.float64)
    values = values.reshape(frames, -1)
    return Motion(skeleton, frame_time, values)


def _axis_rotation(axis: str, degrees: np.ndarray) -> np.ndarray:
    radians = np.radians(degrees)
    c, s = np.cos(radians), np.sin(radians)
    one, zero = np.ones_like(c), np.zeros_like(c)
    if axis == "X":
        rows = [[one, zero, zero], [zero, c, -s], [zero, s, c]]
    elif axis == "Y":
        rows = [[c, zero, s], [zero, one, zero], [-s, zero, c]]
    else:
        rows = [[c, -s, zero], [s, c, zero], [zero, zero, one]]
    return np.stack([np.stack(row, axis=-1) for row in rows], axis=-2)


def forward_kinematics(motion: Motion) -> tuple[np.ndarray, np.ndarray]:
    """World positions [frames, joints, 3] and rotations [frames, joints, 3, 3] of every joint."""
    skeleton = motion.skeleton
    frames = motion.values.shape[0]
    count = len(skeleton.names)
    positions = np.zeros((frames, count, 3))
    rotations = np.zeros((frames, count, 3, 3))
    column = 0
    for joint in range(count):
        local_rotation = np.broadcast_to(np.eye(3), (frames, 3, 3)).copy()
        local_position = np.broadcast_to(skeleton.offsets[joint], (frames, 3)).copy()
        for channel in skeleton.channels[joint]:
            values = motion.values[:, column]
            column += 1
            if channel.endswith("position"):
                local_position[:, "XYZ".index(channel[0])] = values
            else:
                local_rotation = local_rotation @ _axis_rotation(channel[0], values)
        parent = skeleton.parents[joint]
        if parent < 0:
            rotations[:, joint] = local_rotation
            positions[:, joint] = local_position
        else:
            rotations[:, joint] = rotations[:, parent] @ local_rotation
            positions[:, joint] = positions[:, parent] + np.einsum("fij,fj->fi", rotations[:, parent], local_position)
    return positions, rotations
