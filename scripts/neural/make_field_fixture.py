# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "torchvision", "numpy", "pillow", "onnx", "pyyaml"]
# ///

"""Writes the test fixture of klartraum's FieldStyle to data/onnx/field_fixture*: a small FieldStyle of
scripts/neural/train_watercolor_field.py with fixed random weights, exported as FieldStyle loads it (.onnx, .bin),
and one frame of 96 x 64:

- _images.bin: the colour, classes, coverage and depth images as four BGRA8 images (random colours; random classes;
  coverages of 1, 0.7, 0.35 and 0.02; depth over black in its three channels, times the coverage, that puts the
  surface points inside the field's box);
- _frame.bin: 28 floats: the camera (clip from world, row-major, 16), the eye (3), near and far of the depth (2) and
  the conditions (7);
- _input.bin: the network's input planes for that frame (pack()), float32 NCHW (1 x 41 x 64 x 96);
- _output.bin: PyTorch's output for that frame, float32 NCHW (1 x 3 x 64 x 96).

    uv run scripts/neural/make_field_fixture.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_watercolor import PALETTE  # noqa: E402
from train_watercolor_field import (FieldStyle, batch, depth_window, export_runtime, inputs,  # noqa: E402
                                    pack, runtime_view)

WIDTH, HEIGHT = 96, 64


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Camera from world, right-handed, looking along -z (as glm::lookAt)."""
    f = (target - eye) / np.linalg.norm(target - eye)
    s = np.cross(f, [0.0, 1.0, 0.0])
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def projection(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Vulkan clip from camera, depth 0 to 1, y down (as the dataset's)."""
    f = 1.0 / math.tan(fov_y / 2)
    return np.array([[f / aspect, 0, 0, 0], [0, -f, 0, 0], [0, 0, far / (near - far), near * far / (near - far)],
                     [0, 0, -1, 0]])


def main() -> None:
    out = Path(__file__).resolve().parents[2] / "data" / "onnx"
    rng = np.random.default_rng(11)
    low, high = np.array([-1.0, -0.5, -1.0]), np.array([1.0, 1.5, 1.0])
    torch.manual_seed(5)
    model = FieldStyle(low, high, fine=0.2, paper=(20, 28))  # smaller than the frame: the tile repeats
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.3)
    model.eval()

    eye, target = np.array([2.5, 1.5, 3.0]), np.array([0.0, 0.4, 0.0])
    view, proj = look_at(eye, target), projection(math.radians(40), WIDTH / HEIGHT, 0.05, 60.0)
    camera = {"camera": {"view": view.flatten().tolist(), "projection": proj.flatten().tolist(),
                         "position": eye.tolist()}}
    near, far = depth_window(camera, low, high)

    colour = rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    kinds = rng.integers(0, len(PALETTE), (HEIGHT, WIDTH))
    alpha = np.choose(rng.integers(0, 4, (HEIGHT, WIDTH)), [1.0, 0.7, 0.35, 0.02])
    classes = np.round(PALETTE[kinds] * 255).astype(np.uint8)
    coverage = np.repeat(np.round(alpha * 255).astype(np.uint8)[..., None], 3, -1)
    grey = rng.uniform(0.05, 0.95, (HEIGHT, WIDTH))
    depth = np.stack([np.round(alpha * (grey + k / 765) * 255) for k in range(3)], -1).astype(np.uint8)
    conditions = rng.uniform(-1, 1, 7).astype(np.float32)

    view_dict = runtime_view(colour, classes, coverage, depth, {**camera, "time_of_day": 0.0, "season": 0.0,
                                                      "light": {"intensity": 0.0, "source": "sun"}, "exposure": 0.0,
                                                      "background": 0.0}, near, far)
    view_dict["conditions"] = conditions
    x = batch([inputs(view_dict, (slice(0, HEIGHT), slice(0, WIDTH)), torch.device("cpu"))])
    with torch.no_grad():
        y = model(x)
        packed = pack(model, x)
    difference = export_runtime(model, x, out / "field_fixture")
    assert difference < 1e-4, difference
    surface = float(x["surface"].mean())
    print(f"{surface:.0%} of the pixels show a surface; exported network within {difference:.1e}")

    opaque = np.full((HEIGHT, WIDTH, 1), 255, np.uint8)
    images = [np.concatenate([image[..., ::-1], opaque], -1) for image in (colour, classes, coverage, depth)]
    (out / "field_fixture_images.bin").write_bytes(np.stack(images).tobytes())
    frame = np.concatenate([(proj @ view).flatten(), eye, [near, far], conditions]).astype(np.float32)
    (out / "field_fixture_frame.bin").write_bytes(frame.tobytes())
    (out / "field_fixture_input.bin").write_bytes(packed.numpy().astype(np.float32).tobytes())
    (out / "field_fixture_output.bin").write_bytes(y.numpy().astype(np.float32).tobytes())
    print(f"wrote {out / 'field_fixture'}*")


if __name__ == "__main__":
    main()
