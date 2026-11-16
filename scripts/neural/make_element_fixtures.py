# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Reference values of Klartraum's training elements (tests/test_training_elements.cpp): for every case, random
inputs and an upstream gradient, and PyTorch's outputs and autograd's gradients, as float32 files
`<out>/<case>_<tensor>.bin` with the shapes in `<out>/<case>.txt` (one line per tensor: name and dimensions).

    uv run --project scripts/qwen_image python scripts/neural/make_element_fixtures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# The PyTorch references of the elements (training_reference.py) sit next to this script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

OUT = Path(__file__).resolve().parents[2] / "build/TestingOutput/training_fixtures"


def save(out: Path, case: str, tensors: dict[str, torch.Tensor]) -> None:
    lines = []
    for name, tensor in tensors.items():
        tensor.detach().float().contiguous().numpy().tofile(out / f"{case}_{name}.bin")
        lines.append(name + " " + " ".join(str(d) for d in tensor.shape))
    (out / f"{case}.txt").write_text("\n".join(lines) + "\n")


def activation_case(out: Path, case: str, function, seed: int) -> None:
    """An activation and its input gradient."""
    generator = torch.Generator().manual_seed(seed)
    x = (torch.randn(3, 4, 5, 6, generator=generator) * 2.0).requires_grad_(True)
    y = function(x)
    dy = torch.randn(y.shape, generator=generator)
    y.backward(dy)
    save(out, case, {"x": x, "y": y, "dy": dy, "dx": x.grad})


def linear_case(out: Path, seed: int) -> None:
    """A linear layer (nn.Linear 5 -> 7) on a batch of 3 and its input, weight and bias gradients."""
    generator = torch.Generator().manual_seed(seed)
    layer = torch.nn.Linear(5, 7)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(7, 5, generator=generator))
        layer.bias.copy_(torch.randn(7, generator=generator))
    x = torch.randn(3, 5, generator=generator, requires_grad=True)
    y = layer(x)
    dy = torch.randn(3, 7, generator=generator)
    y.backward(dy)
    save(out, "linear", {"x": x, "w": layer.weight, "b": layer.bias, "y": y, "dy": dy, "dx": x.grad,
                         "dw": layer.weight.grad, "db": layer.bias.grad})


def conv_case(out: Path, case: str, n: int, cin: int, cout: int, size: tuple[int, int], kernel: int, stride: int,
              pad: int, seed: int) -> None:
    """A 2D convolution and its input, weight and bias gradients."""
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
    activation_case(args.out, "elu", F.elu, 5)
    activation_case(args.out, "leaky_relu", lambda x: F.leaky_relu(x, 0.2), 6)
    linear_case(args.out, 12)
    conv_case(args.out, "conv3x3", 2, 5, 7, (9, 11), 3, 1, 1, 1)
    conv_case(args.out, "conv3x3_stride2", 2, 5, 7, (10, 12), 3, 2, 1, 2)
    conv_case(args.out, "conv1x1", 2, 6, 3, (8, 8), 1, 1, 0, 3)
    conv_case(args.out, "conv4x4_stride2", 2, 4, 6, (12, 10), 4, 2, 1, 4)
    print(f"fixtures in {args.out}")


if __name__ == "__main__":
    main()
