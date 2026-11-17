# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "pillow", "onnx"]
# ///

"""Trains an unconditional texture GAN on patches of real vegetation, to check that a GAN setup learns them before
it is used to train the tree enhancer.

As the Spatial GAN (Jetchev et al. 2016) the generator is fully convolutional: a spatial noise tensor (8 x 8 x 32
for a 64 px patch) upsampled three times by nearest resize and convolution (no transposed convolutions, which leave
checkerboards), with per-pixel noise added at every scale. The discriminator is the enhancer's PatchGAN on RGB.
For few real images both see the same differentiable augmentations (DiffAugment, Zhao et al. 2020: colour,
translation, cutout); the discriminator gets R1 regularisation on real patches every 16 steps; the losses are
non-saturating; samples come from an exponential moving average of the generator.

Writes every --every steps a grid of real (left) and generated (right) patches to <out-dir>/samples_NNNNNN.png and
prints statistics of both: mean colour, the detail energy (lod_metrics' detailError() energy) and the slope of the
radially averaged power spectrum.

    uv run scripts/neural/train_texture_gan.py <photos-dir> <out-dir> [--steps 20000]
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from train_tree_enhancer import RealCrops, diff_augment  # noqa: E402


class NoiseInjection(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.weight * torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)


class Generator(nn.Module):
    """Spatial noise (32 channels at 1/8 of the size) to RGB, fully convolutional."""

    def __init__(self, z: int = 32, width: int = 96) -> None:
        super().__init__()
        self.z = z

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), NoiseInjection(cout), nn.LeakyReLU(0.2),
                                 nn.Conv2d(cout, cout, 3, padding=1), NoiseInjection(cout), nn.LeakyReLU(0.2))

        self.start = block(z, width)
        self.up = nn.ModuleList([block(width, width), block(width, width // 2), block(width // 2, width // 2)])
        self.rgb = nn.Conv2d(width // 2, 3, 1)

    def noise(self, batch: int, size: int, device: torch.device) -> torch.Tensor:
        return torch.randn(batch, self.z, size // 8, size // 8, device=device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.start(z)
        for block in self.up:
            x = block(F.interpolate(x, scale_factor=2, mode="nearest"))
        return torch.sigmoid(self.rgb(x))


class Discriminator(nn.Module):
    """A PatchGAN on RGB with a logit per patch of about 16 px; its mean is the patch's score."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
                                 nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
                                 nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
                                 nn.Conv2d(256, 1, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x * 2.0 - 1.0).mean(dim=(1, 2, 3))


def statistics(x: torch.Tensor) -> tuple[np.ndarray, float, float]:
    """Mean colour, mean Laplacian energy of the luminance and the slope of the radially averaged power spectrum."""
    x = x.detach().float().cpu()
    luminance = (x * torch.tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
    kernel = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]).view(1, 1, 3, 3)
    energy = float((F.conv2d(luminance, kernel) ** 2).mean())
    spectrum = torch.fft.fftshift(torch.fft.fft2(luminance[:, 0] - luminance[:, 0].mean(dim=(1, 2), keepdim=True)),
                                  dim=(1, 2)).abs().pow(2).mean(dim=0).numpy()
    h, w = spectrum.shape
    yy, xx = np.indices(spectrum.shape)
    radius = np.hypot(yy - h // 2, xx - w // 2).astype(int)
    radial = np.bincount(radius.ravel(), spectrum.ravel()) / np.maximum(np.bincount(radius.ravel()), 1)
    r = np.arange(2, min(h, w) // 2)
    slope = float(np.polyfit(np.log(r), np.log(radial[r] + 1e-12), 1)[0])
    return x.mean(dim=(0, 2, 3)).numpy(), energy, slope


def grid(real: torch.Tensor, fake: torch.Tensor, path: Path, columns: int = 8) -> None:
    def sheet(x: torch.Tensor) -> np.ndarray:
        x = x[: columns * columns].detach().cpu().clamp(0, 1)
        n, c, h, w = x.shape
        rows = math.ceil(n / columns)
        out = torch.zeros(c, rows * h, columns * w)
        for i in range(n):
            out[:, (i // columns) * h:(i // columns + 1) * h, (i % columns) * w:(i % columns + 1) * w] = x[i]
        return (out.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    left, right = sheet(real), sheet(fake)
    gap = np.full((left.shape[0], 16, 3), 255, np.uint8)
    sheet_image = np.concatenate([left, gap, right], axis=1)
    Image.fromarray(sheet_image).resize((2 * sheet_image.shape[1], 2 * sheet_image.shape[0]), Image.NEAREST).save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("photos", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--r1", type=float, default=1.0, help="weight of the R1 penalty")
    parser.add_argument("--every", type=int, default=1000)
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rng = np.random.default_rng(1)
    real = RealCrops(args.photos, crop=args.size)

    generator, discriminator = Generator().to(device), Discriminator().to(device)
    ema = copy.deepcopy(generator).eval()
    g_opt = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.0, 0.99))
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.0, 0.99))
    augment = (lambda x: x) if args.no_augment else diff_augment
    fixed = generator.noise(64, args.size, device)
    start = time.time()
    for step in range(1, args.steps + 1):
        x = real.sample(args.batch, rng).to(device)
        z = generator.noise(args.batch, args.size, device)
        # Discriminator: non-saturating loss on augmented real and generated patches, R1 every 16 steps.
        fake = generator(z).detach()
        d_loss = F.softplus(-discriminator(augment(x))).mean() + F.softplus(discriminator(augment(fake))).mean()
        if step % 16 == 0:
            x.requires_grad_(True)
            score = discriminator(augment(x))
            (grad,) = torch.autograd.grad(score.sum(), x, create_graph=True)
            d_loss = d_loss + args.r1 * 16 * 0.5 * grad.pow(2).sum(dim=(1, 2, 3)).mean()
        d_opt.zero_grad()
        d_loss.backward()
        d_opt.step()
        # Generator.
        g_loss = F.softplus(-discriminator(augment(generator(z)))).mean()
        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()
        with torch.no_grad():
            for e, p in zip(ema.parameters(), generator.parameters()):
                e.lerp_(p, 0.001)
        if step % args.every == 0 or step == args.steps:
            with torch.no_grad():
                samples = ema(fixed)
            reference = real.sample(64, np.random.default_rng(2))
            grid(reference, samples, args.out_dir / f"samples_{step:06d}.png")
            (rc, re, rs), (fc, fe, fs) = statistics(reference), statistics(samples)
            print(f"step {step}: d {d_loss.item():.3f}, g {g_loss.item():.3f}; real colour {np.round(rc, 3)}, "
                  f"energy {re:.4f}, spectrum slope {rs:.2f}; generated colour {np.round(fc, 3)}, energy {fe:.4f}, "
                  f"slope {fs:.2f}; {time.time() - start:.0f} s", flush=True)
    torch.save({"generator": ema.state_dict()}, args.out_dir / "texture_gan.pt")


if __name__ == "__main__":
    main()
