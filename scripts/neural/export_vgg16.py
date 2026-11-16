# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Exports the convolutions of torchvision's VGG-16 (ImageNet weights) up to relu4_3, the layers of the photo
enhancer's perceptual loss (train_photo_enhancer.Vgg), for Klartraum's neural::VggLoss: `<out>/conv<k>.weight.bin`
[out][in][3][3] and `<out>/conv<k>.bias.bin` [out] (float32) for the ten convolutions k = 0..9, and
`<out>/layers.txt` with one line per convolution: its input and output channels and whether a 2 x 2 max-pool follows
it. The weights are torchvision's (BSD-3-Clause), trained on ImageNet.

    uv run --project scripts/qwen_image python scripts/neural/export_vgg16.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.models import VGG16_Weights, vgg16

OUT = Path(__file__).resolve().parents[2] / "build/models/vgg16"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:23].eval()
    lines, k = [], 0
    modules = list(features)
    for i, module in enumerate(modules):
        if isinstance(module, torch.nn.Conv2d):
            module.weight.detach().numpy().astype("float32").tofile(args.out / f"conv{k}.weight.bin")
            module.bias.detach().numpy().astype("float32").tofile(args.out / f"conv{k}.bias.bin")
            pooled = i + 2 < len(modules) and isinstance(modules[i + 2], torch.nn.MaxPool2d)
            lines.append(f"{module.in_channels} {module.out_channels} {int(pooled)}")
            k += 1
    (args.out / "layers.txt").write_text("\n".join(lines) + "\n")
    print(f"{k} convolutions in {args.out}")


if __name__ == "__main__":
    main()
