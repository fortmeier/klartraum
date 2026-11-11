# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Reference gradients for the training elements of Klartraum (tests/test_photo_enhancer_training.cpp): for every
case, random inputs and an upstream gradient, and PyTorch autograd's gradients, as float32 files
`<out>/<case>_<tensor>.bin` with the shapes in `<out>/<case>.txt` (one line per tensor: name and dimensions).

    uv run --project scripts/qwen_image python scripts/neural/make_training_fixtures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

OUT = Path(__file__).resolve().parents[2] / "build/TestingOutput/training_fixtures"


def save(out: Path, case: str, tensors: dict[str, torch.Tensor]) -> None:
    lines = []
    for name, tensor in tensors.items():
        tensor.detach().float().contiguous().numpy().tofile(out / f"{case}_{name}.bin")
        lines.append(name + " " + " ".join(str(d) for d in tensor.shape))
    (out / f"{case}.txt").write_text("\n".join(lines) + "\n")


def conv_case(out: Path, case: str, n: int, cin: int, cout: int, size: tuple[int, int], kernel: int, stride: int,
              pad: int, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, cin, *size, generator=generator, requires_grad=True)
    w = (torch.randn(cout, cin, kernel, kernel, generator=generator) * 0.3).requires_grad_(True)
    b = torch.randn(cout, generator=generator, requires_grad=True)
    y = F.conv2d(x, w, b, stride=stride, padding=pad)
    dy = torch.randn(y.shape, generator=generator)
    y.backward(dy)
    save(out, case, {"x": x, "w": w, "b": b, "y": y, "dy": dy, "dx": x.grad, "dw": w.grad, "db": b.grad})


def activation_case(out: Path, case: str, function, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    x = (torch.randn(3, 4, 5, 6, generator=generator) * 2.0).requires_grad_(True)
    y = function(x)
    dy = torch.randn(y.shape, generator=generator)
    y.backward(dy)
    save(out, case, {"x": x, "y": y, "dy": dy, "dx": x.grad})


def channel_norm_case(out: Path, seed: int) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_photo_enhancer import ChannelNorm

    generator = torch.Generator().manual_seed(seed)
    norm = ChannelNorm(6)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(6, generator=generator))
        norm.bias.copy_(torch.randn(6, generator=generator))
    x = (torch.randn(2, 6, 5, 7, generator=generator) * 3.0 + 1.0).requires_grad_(True)
    y = norm(x)
    dy = torch.randn(y.shape, generator=generator)
    y.backward(dy)
    save(out, "channel_norm", {"x": x, "weight": norm.weight, "bias": norm.bias, "y": y, "dy": dy, "dx": x.grad,
                               "dweight": norm.weight.grad, "dbias": norm.bias.grad})


def upsample_case(out: Path, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 3, 5, 7, generator=generator, requires_grad=True)
    y = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
    dy = torch.randn(y.shape, generator=generator)
    y.backward(dy)
    save(out, "upsample2x", {"x": x, "y": y, "dy": dy, "dx": x.grad})


def film_case(out: Path, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 4, 5, 6, generator=generator, requires_grad=True)
    scale = torch.randn(2, 4, generator=generator, requires_grad=True)
    shift = torch.randn(2, 4, generator=generator, requires_grad=True)
    y = x * scale[:, :, None, None] + shift[:, :, None, None]
    dy = torch.randn(y.shape, generator=generator)
    y.backward(dy)
    save(out, "film", {"x": x, "scale": scale, "shift": shift, "y": y, "dy": dy, "dx": x.grad,
                       "dscale": scale.grad, "dshift": shift.grad})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    conv_case(args.out, "conv3x3", 2, 5, 7, (9, 11), 3, 1, 1, 1)
    conv_case(args.out, "conv3x3_stride2", 2, 5, 7, (10, 12), 3, 2, 1, 2)
    conv_case(args.out, "conv1x1", 2, 6, 3, (8, 8), 1, 1, 0, 3)
    conv_case(args.out, "conv4x4_stride2", 2, 4, 6, (12, 10), 4, 2, 1, 4)
    activation_case(args.out, "elu", F.elu, 5)
    activation_case(args.out, "leaky_relu", lambda x: F.leaky_relu(x, 0.2), 6)
    channel_norm_case(args.out, 7)
    upsample_case(args.out, 8)
    film_case(args.out, 9)
    print(f"fixtures in {args.out}")


if __name__ == "__main__":
    main()
