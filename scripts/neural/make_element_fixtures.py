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


def channel_norm_case(out: Path, seed: int) -> None:
    """ChannelNorm of 6 channels and its input, scale and shift gradients."""
    from training_reference import ChannelNorm

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
    """A bilinear upsampling by two (align_corners=False) and its input gradient."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 3, 5, 7, generator=generator, requires_grad=True)
    y = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
    dy = torch.randn(y.shape, generator=generator)
    y.backward(dy)
    save(out, "upsample2x", {"x": x, "y": y, "dy": dy, "dx": x.grad})


def max_pool_case(out: Path, seed: int) -> None:
    """A 2 x 2 max-pool and its input gradient."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 3, 6, 8, generator=generator, requires_grad=True)
    y = F.max_pool2d(x, 2)
    dy = torch.randn(y.shape, generator=generator)
    y.backward(dy)
    save(out, "max_pool", {"x": x, "y": y, "dy": dy, "dx": x.grad})


def film_case(out: Path, seed: int) -> None:
    """FiLM (a scale and shift per sample and channel) and its input, scale and shift gradients."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 4, 5, 6, generator=generator, requires_grad=True)
    scale = torch.randn(2, 4, generator=generator, requires_grad=True)
    shift = torch.randn(2, 4, generator=generator, requires_grad=True)
    y = x * scale[:, :, None, None] + shift[:, :, None, None]
    dy = torch.randn(y.shape, generator=generator)
    y.backward(dy)
    save(out, "film", {"x": x, "scale": scale, "shift": shift, "y": y, "dy": dy, "dx": x.grad,
                       "dscale": scale.grad, "dshift": shift.grad})


def clip_case(out: Path, seed: int) -> None:
    """Two gradients clipped to a global norm of 1 (clip_grad_norm_)."""
    generator = torch.Generator().manual_seed(seed)
    a = torch.nn.Parameter(torch.zeros(300))
    b = torch.nn.Parameter(torch.zeros(70))
    a.grad = torch.randn(300, generator=generator) * 0.2
    b.grad = torch.randn(70, generator=generator) * 0.2
    before = {"ga": a.grad.clone(), "gb": b.grad.clone()}
    torch.nn.utils.clip_grad_norm_([a, b], 1.0)
    save(out, "clip", {**before, "clipped_a": a.grad, "clipped_b": b.grad})


def photometric_case(out: Path, seed: int) -> None:
    """0.5 L1 + 5 L1 of 8 x 8 block means of an image against a target, and the image's gradient."""
    generator = torch.Generator().manual_seed(seed)
    image = torch.rand(2, 3, 16, 20, generator=generator, requires_grad=True)
    target = torch.rand(2, 3, 16, 20, generator=generator)
    loss = 0.5 * F.l1_loss(image, target) + 5.0 * F.l1_loss(F.avg_pool2d(image, 8), F.avg_pool2d(target, 8))
    loss.backward()
    save(out, "photometric", {"image": image, "target": target, "dimage": image.grad, "loss": loss.reshape(1)})


def vgg_case(out: Path, seed: int) -> None:
    """The VGG-16 perceptual distance of an image to a target and the image's gradient."""
    from training_reference import Vgg

    generator = torch.Generator().manual_seed(seed)
    vgg = Vgg().eval()
    image = torch.rand(2, 3, 32, 32, generator=generator, requires_grad=True)
    target = torch.rand(2, 3, 32, 32, generator=generator)
    distance = vgg.distance(image, target)
    distance.backward()
    save(out, "vgg", {"image": image, "target": target, "dimage": image.grad, "loss": distance.reshape(1)})


def augment_backward_case(out: Path, seed: int) -> None:
    """DiffAugment with given numbers (training_reference.augment_with()) and its input gradient: pairs of 2 G-buffer
    channels and an image, a positive and a negative roll, cutouts in the middle and at the border."""
    from training_reference import augment_with

    generator = torch.Generator().manual_seed(seed)
    pair = torch.rand(3, 5, 12, 16, generator=generator, requires_grad=True)
    parameters = torch.tensor([[0.1, 1.3, 2.0, -3.0, 5.0, 7.0], [-0.15, 0.6, -1.0, 4.0, 0.0, 15.0],
                               [0.0, 1.0, 0.0, 0.0, 11.0, 2.0]])
    augmented = augment_with(pair, parameters)
    gradient = torch.randn(augmented.shape, generator=generator)
    (augmented * gradient).sum().backward()
    save(out, "augment_backward", {"pair": pair, "parameters": parameters, "augmented": augmented,
                                   "daugmented": gradient, "dpair": pair.grad})


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
    channel_norm_case(args.out, 7)
    upsample_case(args.out, 8)
    max_pool_case(args.out, 15)
    film_case(args.out, 9)
    clip_case(args.out, 13)
    photometric_case(args.out, 11)
    vgg_case(args.out, 15)
    augment_backward_case(args.out, 23)
    print(f"fixtures in {args.out}")


if __name__ == "__main__":
    main()
