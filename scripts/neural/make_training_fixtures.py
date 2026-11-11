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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    conv_case(args.out, "conv3x3", 2, 5, 7, (9, 11), 3, 1, 1, 1)
    conv_case(args.out, "conv3x3_stride2", 2, 5, 7, (10, 12), 3, 2, 1, 2)
    conv_case(args.out, "conv1x1", 2, 6, 3, (8, 8), 1, 1, 0, 3)
    conv_case(args.out, "conv4x4_stride2", 2, 4, 6, (12, 10), 4, 2, 1, 4)
    print(f"fixtures in {args.out}")


if __name__ == "__main__":
    main()
