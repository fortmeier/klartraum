# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "torchvision", "numpy", "pillow", "onnx", "pyyaml"]
# ///

"""Writes the test fixtures of klartraum's StyleTransfer: small networks of scripts/neural/train_watercolor.py with
fixed random weights (rgb: the U-Net on the rendering; gbuffers: the U-Net on the rendering, its classes and depth;
spade: the G-buffers modulating the decoder) to data/onnx/watercolor_<kind>_fixture*:

- *.onnx: the network at 96 x 64 (the half of a 192 x 128 frame);
- *_images.bin: its frame as three BGRA8 images of 192 x 128 (colour, classes, depth): random bytes, the classes
  random colours of the class palette;
- *_output.bin: PyTorch's output for them, float32 NCHW (1 x 3 x 64 x 96), the input made as StyleTransfer makes it
  (the colour and depth averaged over 2 x 2 pixels, the class of pixel (2x + 1, 2y + 1) one-hot);
- *_upsampled.bin: that output resized bilinearly to 192 x 128 and clamped to [0, 1], float32 NCHW.

    uv run scripts/neural/make_watercolor_fixture.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from train_watercolor import PALETTE, SpadeUNet, UNet, export_onnx  # noqa: E402

FRAME_WIDTH, FRAME_HEIGHT = 192, 128
WIDTH, HEIGHT = FRAME_WIDTH // 2, FRAME_HEIGHT // 2


def network_input(colour: np.ndarray, classes: np.ndarray, depth: np.ndarray, with_gbuffers: bool) -> torch.Tensor:
    """The network's input from the frame's BGRA images, as StyleTransfer makes it."""
    def average(bgra: np.ndarray) -> np.ndarray:
        rgb = bgra[..., [2, 1, 0]].astype(np.float32) / 255.0
        return rgb.reshape(HEIGHT, 2, WIDTH, 2, 3).mean((1, 3))

    planes = [average(colour)]
    if with_gbuffers:
        picked = classes[1::2, 1::2][..., [2, 1, 0]].astype(np.float32) / 255.0
        nearest = np.argmin(((picked[:, :, None, :] - PALETTE[None, None]) ** 2).sum(-1), -1)
        planes.append(np.eye(len(PALETTE), dtype=np.float32)[nearest])
        planes.append(average(depth)[..., :1])
    return torch.from_numpy(np.concatenate(planes, -1)).permute(2, 0, 1)[None].contiguous()


def main() -> None:
    out = Path(__file__).resolve().parents[2] / "data" / "onnx"
    generator = np.random.default_rng(7)
    colour = generator.integers(0, 256, (FRAME_HEIGHT, FRAME_WIDTH, 4), dtype=np.uint8)
    palette_bytes = np.round(PALETTE * 255).astype(np.uint8)[:, [2, 1, 0]]
    classes = np.concatenate([palette_bytes[generator.integers(0, len(PALETTE), (FRAME_HEIGHT, FRAME_WIDTH))],
                              np.full((FRAME_HEIGHT, FRAME_WIDTH, 1), 255, np.uint8)], -1)
    grey = generator.integers(0, 256, (FRAME_HEIGHT, FRAME_WIDTH, 1), dtype=np.uint8)
    depth = np.concatenate([grey, grey, grey, np.full_like(grey, 255)], -1)
    images = np.stack([colour, classes, depth]).tobytes()
    width = (8, 8, 8, 8)
    for kind, model, with_gbuffers in (("rgb", UNet(3, width), False), ("gbuffers", UNet(11, width), True),
                                       ("spade", SpadeUNet(8, width), True)):
        torch.manual_seed(3)
        for p in model.parameters():
            torch.nn.init.normal_(p, std=0.3)
        model.eval()
        x = network_input(colour, classes, depth, with_gbuffers)
        with torch.no_grad():
            y = model(x)
            upsampled = F.interpolate(y, scale_factor=2.0, mode="bilinear", align_corners=False).clamp(0, 1)
        stem = out / f"watercolor_{kind}_fixture"
        export_onnx(model, x, Path(f"{stem}.onnx"))
        Path(f"{stem}_images.bin").write_bytes(images)
        Path(f"{stem}_output.bin").write_bytes(y.numpy().astype(np.float32).tobytes())
        Path(f"{stem}_upsampled.bin").write_bytes(upsampled.numpy().astype(np.float32).tobytes())
        print(f"wrote {stem}*")


if __name__ == "__main__":
    main()
