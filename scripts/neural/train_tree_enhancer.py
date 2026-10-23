# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "pillow", "onnx"]
# ///

"""Trains the small CNN that enhances the Gaussian renderer's distant trees.

Reads the view pairs of tree_neural_dataset_example (lr_*, fa_*, fb_*, hf_*, mask_*): the network sees the
Gaussian image and its features (8 channels) and predicts a correction added to the image. Its target is the
reference where the far trees' mask says so and the Gaussian image elsewhere, so it learns to change only the
distant trees. Every tenth view is held out for validation. Writes the weights, the ONNX model and the
validation numbers (the error of 4 px block means against the target on the far trees' pixels, of the input
and of the output) to the output directory.

    uv run scripts/neural/train_tree_enhancer.py <dataset-dir> <out-dir> [--steps 20000]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

INPUT_CHANNELS = 8


def load_view(directory: Path, name: str) -> dict[str, np.ndarray]:
    def rgb(prefix: str) -> np.ndarray:
        return np.array(Image.open(directory / f"{prefix}_{name}.ppm").convert("RGB"), dtype=np.uint8)

    return {
        "lr": rgb("lr"),
        "fa": rgb("fa"),
        "fb": rgb("fb"),
        "hf": rgb("hf"),
        "mask": np.array(Image.open(directory / f"mask_{name}.pgm"), dtype=np.uint8),
    }


def to_tensors(view: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The network's input (8, H, W), its target (3, H, W) and the far trees' mask (1, H, W), all in [0, 1]."""
    lr = torch.from_numpy(view["lr"]).permute(2, 0, 1).float() / 255.0
    fa = torch.from_numpy(view["fa"]).permute(2, 0, 1).float() / 255.0
    fb = torch.from_numpy(view["fb"][..., :2]).permute(2, 0, 1).float() / 255.0
    hf = torch.from_numpy(view["hf"]).permute(2, 0, 1).float() / 255.0
    mask = torch.from_numpy(view["mask"]).unsqueeze(0).float() / 255.0
    inputs = torch.cat([lr, fa, fb], dim=0)
    target = mask * hf + (1.0 - mask) * lr
    return inputs, target, mask


class Enhancer(nn.Module):
    """A three-level U-Net: 16, 32 and 48 channels; the output is added to the input image."""

    def __init__(self, width: tuple[int, int, int] = (16, 32, 48)) -> None:
        super().__init__()
        a, b, c = width

        def block(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1), nn.ReLU(),
                                 nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU())

        self.enc0 = block(INPUT_CHANNELS, a)
        self.enc1 = block(a, b, stride=2)
        self.enc2 = block(b, c, stride=2)
        self.up1 = nn.ConvTranspose2d(c, b, 2, stride=2)
        self.dec1 = block(2 * b, b)
        self.up0 = nn.ConvTranspose2d(b, a, 2, stride=2)
        self.dec0 = block(2 * a, a)
        self.out = nn.Conv2d(a, 3, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        # The input image's RGB, taken by a fixed 1 x 1 convolution with a zero bias (exported as a Conv with bias,
        # as OnnxNetwork runs it).
        self.pick = nn.Conv2d(INPUT_CHANNELS, 3, 1)
        with torch.no_grad():
            self.pick.weight.zero_()
            self.pick.bias.zero_()
            for c in range(3):
                self.pick.weight[c, c] = 1.0
        self.pick.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        d1 = self.dec1(torch.cat([self.up1(e2), e1], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), e0], dim=1))
        return self.pick(x) + self.out(d0)


def constants_as_initializers(model):
    """Moves the values of Constant nodes into the graph's initializers and lists every initializer as a graph
    input, as OnnxNetwork reads them."""
    import onnx

    graph = model.graph
    keep = []
    for node in graph.node:
        if node.op_type == "Constant" and len(node.attribute) == 1 and node.attribute[0].name == "value":
            tensor = onnx.TensorProto()
            tensor.CopyFrom(node.attribute[0].t)
            tensor.name = node.output[0]
            graph.initializer.append(tensor)
        else:
            keep.append(node)
    del graph.node[:]
    graph.node.extend(keep)
    # OnnxNetwork makes tensors of the graph's inputs (and fills them from the initializers of the same name):
    # every initializer is listed as an input, too.
    listed = {i.name for i in graph.input}
    for tensor in graph.initializer:
        if tensor.name not in listed:
            graph.input.append(onnx.helper.make_tensor_value_info(tensor.name, tensor.data_type, list(tensor.dims)))
    return model


class EnhancerV2(nn.Module):
    """Cheaper at full resolution: two strided 3 x 3 convolutions down to a quarter, resizes up (nearest neighbour
    or bilinear, `upsample`), and at full resolution one 3 x 3 convolution of 16 to 8 channels and a 1 x 1 to RGB;
    the output is added to the input image."""

    def __init__(self, upsample: str = "nearest") -> None:
        super().__init__()
        self.upsample = upsample
        self.down1 = nn.Sequential(nn.Conv2d(INPUT_CHANNELS, 24, 3, stride=2, padding=1), nn.ReLU(),
                                   nn.Conv2d(24, 24, 3, padding=1), nn.ReLU())
        self.down2 = nn.Sequential(nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.ReLU(),
                                   nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                                   nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.mid = nn.Sequential(nn.Conv2d(32 + 24, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 8, 1), nn.ReLU())
        self.full = nn.Sequential(nn.Conv2d(8 + INPUT_CHANNELS, 8, 3, padding=1), nn.ReLU())
        self.out = nn.Conv2d(8, 3, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.pick = nn.Conv2d(INPUT_CHANNELS, 3, 1)
        with torch.no_grad():
            self.pick.weight.zero_()
            self.pick.bias.zero_()
            for c in range(3):
                self.pick.weight[c, c] = 1.0
        self.pick.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        m = self.mid(torch.cat([self.up(d2), d1], dim=1))
        f = self.full(torch.cat([self.up(m), x], dim=1))
        return self.pick(x) + self.out(f)

    def up(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample == "nearest":
            return F.interpolate(x, scale_factor=2, mode="nearest")
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)


# v3: v2 with bilinear upsampling, which leaves no blocks of 2 x 2 or 4 x 4 pixels for an adversarial loss to
# sharpen into a grid.
ARCHITECTURES = {"v1": Enhancer, "v2": EnhancerV2, "v3": lambda: EnhancerV2(upsample="bilinear")}


def texture(x: torch.Tensor) -> torch.Tensor:
    """The fine structure of RGB crops: their luminance minus its 5 x 5 box blur, scaled by the crop's mean
    luminance (so brightness and exposure do not count), without any colour."""
    weights = torch.tensor([0.2126, 0.7152, 0.0722], device=x.device).view(1, 3, 1, 1)
    luminance = (x * weights).sum(dim=1, keepdim=True)
    blurred = F.avg_pool2d(F.pad(luminance, (2, 2, 2, 2), mode="replicate"), 5, stride=1)
    return (luminance - blurred) / (luminance.mean(dim=(2, 3), keepdim=True) + 0.05)


def diff_augment(x: torch.Tensor) -> torch.Tensor:
    """DiffAugment's colour, translation (1/8) and cutout (1/2) on a batch, differentiable and per sample."""
    b = x.shape[0]
    dev = x.device
    x = x + (torch.rand(b, 1, 1, 1, device=dev) - 0.5)
    mean = x.mean(dim=1, keepdim=True)
    x = (x - mean) * (torch.rand(b, 1, 1, 1, device=dev) * 2) + mean
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - mean) * (torch.rand(b, 1, 1, 1, device=dev) + 0.5) + mean
    # Translation by up to 1/8 of the size, padding with zeros.
    h, w = x.shape[2:]
    sx, sy = w // 8, h // 8
    tx = torch.randint(-sx, sx + 1, (b, 1, 1), device=dev)
    ty = torch.randint(-sy, sy + 1, (b, 1, 1), device=dev)
    gy, gx = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing="ij")
    gx = torch.clamp(gx.unsqueeze(0) + tx + 1, 0, w + 1)
    gy = torch.clamp(gy.unsqueeze(0) + ty + 1, 0, h + 1)
    padded = F.pad(x, (1, 1, 1, 1))
    x = padded.permute(0, 2, 3, 1)[torch.arange(b, device=dev).view(b, 1, 1), gy, gx].permute(0, 3, 1, 2)
    # Cutout: a square of half the size set to zero.
    ch, cw = h // 2, w // 2
    ox = torch.randint(0, w + (1 - cw % 2), (b, 1, 1), device=dev)
    oy = torch.randint(0, h + (1 - ch % 2), (b, 1, 1), device=dev)
    gy, gx = torch.meshgrid(torch.arange(ch, device=dev), torch.arange(cw, device=dev), indexing="ij")
    gx = torch.clamp(gx.unsqueeze(0) + ox - cw // 2, 0, w - 1)
    gy = torch.clamp(gy.unsqueeze(0) + oy - ch // 2, 0, h - 1)
    mask = torch.ones(b, h, w, device=dev)
    mask[torch.arange(b, device=dev).view(b, 1, 1), gy, gx] = 0
    return x * mask.unsqueeze(1)


class PatchDiscriminator(nn.Module):
    """Judges 64 x 64 crops as real photographs of vegetation or enhanced renderings, on RGB or on their texture
    alone (texture(), so colour stays with the other losses): a logit per patch of about 16 px, averaged into the
    crop's score. The setup train_texture_gan.py learns vegetation with from noise."""

    def __init__(self, input: str = "rgb") -> None:
        super().__init__()
        self.input = input
        channels = 3 if input == "rgb" else 1
        self.net = nn.Sequential(nn.Conv2d(channels, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
                                 nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
                                 nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.LeakyReLU(0.2),
                                 nn.Conv2d(256, 1, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x * 2.0 - 1.0 if self.input == "rgb" else texture(x)).mean(dim=(1, 2, 3))


class RealCrops:
    """Crops of vegetation from real photographs (a directory of JPEGs, e.g. download_real_forest.py's): each
    photo scaled to 512 px wide, crops of 64 to 256 px resized to 64, centred where green dominates (green above
    red and blue by a margin) and taken only where most of the crop is vegetation."""

    def __init__(self, directory: Path, crop: int = 64) -> None:
        self.crop = crop
        self.images, self.centres = [], []
        for path in sorted(directory.glob("*.jpg")):
            try:
                image = Image.open(path).convert("RGB")
            except Exception:
                continue
            w, h = image.size
            image = image.resize((512, max(int(512 * h / w), 130)))
            pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            r, g, b = pixels
            green = (g > r + 0.02) & (g > b + 0.02)
            if green.float().mean() < 0.3:
                continue
            centres = torch.nonzero(green[64:-64, 64:-64]) + 64
            if len(centres) == 0:
                continue
            self.images.append(pixels)
            self.centres.append((centres, green))
        print(f"{len(self.images)} photographs with enough vegetation in {directory}")

    def sample(self, count: int, rng: np.random.Generator) -> torch.Tensor:
        out = []
        while len(out) < count:
            k = int(rng.integers(len(self.images)))
            centres, green = self.centres[k]
            # 64 to 256 px of a 512 px wide photograph: leaves a few pixels small or smaller, as far away.
            size = int(rng.integers(self.crop, 4 * self.crop + 1))
            cy, cx = centres[int(rng.integers(len(centres)))].tolist()
            y0, x0 = max(0, cy - size // 2), max(0, cx - size // 2)
            image = self.images[k]
            if y0 + size > image.shape[1] or x0 + size > image.shape[2]:
                continue
            if green[y0:y0 + size, x0:x0 + size].float().mean() < 0.7:
                continue
            crop = image[:, y0:y0 + size, x0:x0 + size].unsqueeze(0)
            out.append(F.interpolate(crop, size=(self.crop, self.crop), mode="area")[0])
        return torch.stack(out)


def fake_crops(images: torch.Tensor, masks: torch.Tensor, count: int, rng: np.random.Generator,
               crop: int = 64) -> torch.Tensor:
    """`count` crops of `crop` px of rendered images, centred on their far trees (where the mask is above 0.5)."""
    out = []
    _, _, h, w = images.shape
    for _ in range(count):
        k = int(rng.integers(images.shape[0]))
        idx = torch.nonzero(masks[k, 0] > 0.5)
        if len(idx) == 0:
            y, x = int(rng.integers(h - crop + 1)), int(rng.integers(w - crop + 1))
        else:
            cy, cx = idx[int(rng.integers(len(idx)))].tolist()
            y, x = int(np.clip(cy - crop // 2, 0, h - crop)), int(np.clip(cx - crop // 2, 0, w - crop))
        out.append(images[k, :, y:y + crop, x:x + crop])
    return torch.stack(out)


def detail_loss(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, block: int = 8,
                floor: float = 1e-4) -> torch.Tensor:
    """detail_error() as a loss: the mean |log2| of the ratio of the blocks' Laplacian energies, over the blocks
    of the far trees."""
    weights = torch.tensor([0.2126, 0.7152, 0.0722], device=a.device).view(1, 3, 1, 1)
    kernel = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], device=a.device).view(1, 1, 3, 3)

    def energy(x: torch.Tensor) -> torch.Tensor:
        luminance = (x * weights).sum(dim=1, keepdim=True)
        laplacian = F.conv2d(F.pad(luminance, (1, 1, 1, 1), mode="replicate"), kernel)
        return F.avg_pool2d(laplacian * laplacian, block)

    keep = (F.avg_pool2d(mask, block) > 0.25).float()
    ratio = torch.log2((energy(a) + floor) / (energy(b) + floor)).abs()
    return (ratio * keep).sum() / keep.sum().clamp(min=1.0)


def block_error(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    """The mean absolute difference of 4 px block means (RGB averaged), over the blocks the mask touches."""
    da = F.avg_pool2d(a, 4) - F.avg_pool2d(b, 4)
    weight = F.max_pool2d(mask, 4) > 0.0
    per_block = da.abs().mean(dim=1, keepdim=True)
    return float(per_block[weight].mean()) if weight.any() else 0.0


def detail_error(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor, block: int = 8, floor: float = 1e-4) -> float:
    """lod_metrics' detailError(): per 8 px block the energy of the luminance's Laplacian, compared as
    |log2(a's / b's)|, averaged over the blocks the mask touches (and lit in either image)."""
    weights = torch.tensor([0.2126, 0.7152, 0.0722], device=a.device).view(1, 3, 1, 1)
    kernel = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], device=a.device).view(1, 1, 3, 3)

    def energy(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        luminance = (x * weights).sum(dim=1, keepdim=True)
        laplacian = F.conv2d(F.pad(luminance, (1, 1, 1, 1), mode="replicate"), kernel)
        return F.avg_pool2d(laplacian * laplacian, block), F.avg_pool2d(luminance, block)

    ea, la = energy(a)
    eb, lb = energy(b)
    keep = (F.max_pool2d(mask, block) > 0.0) & (torch.maximum(la, lb) > 0.01)
    ratio = torch.log2((ea + floor) / (eb + floor)).abs()
    return float(ratio[keep].mean()) if keep.any() else 0.0


def crops(inputs: list[torch.Tensor], targets: list[torch.Tensor], masks: list[torch.Tensor], size: int,
          count: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`count` crops of `size` px, three quarters of them centred on a far-tree pixel."""
    xs, ys, ms = [], [], []
    for _ in range(count):
        k = int(rng.integers(len(inputs)))
        _, h, w = inputs[k].shape
        mask = masks[k][0]
        if rng.random() < 0.75 and bool((mask > 0.5).any()):
            idx = torch.nonzero(mask > 0.5)
            cy, cx = idx[int(rng.integers(len(idx)))].tolist()
            y = int(np.clip(cy - size // 2, 0, h - size))
            x = int(np.clip(cx - size // 2, 0, w - size))
        else:
            y = int(rng.integers(h - size + 1))
            x = int(rng.integers(w - size + 1))
        xs.append(inputs[k][:, y:y + size, x:x + size])
        ys.append(targets[k][:, y:y + size, x:x + size])
        ms.append(masks[k][:, y:y + size, x:x + size])
    return torch.stack(xs), torch.stack(ys), torch.stack(ms)


def gradient_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    dxa, dxb = a[..., :, 1:] - a[..., :, :-1], b[..., :, 1:] - b[..., :, :-1]
    dya, dyb = a[..., 1:, :] - a[..., :-1, :], b[..., 1:, :] - b[..., :-1, :]
    return (dxa - dxb).abs().mean() + (dya - dyb).abs().mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--arch", choices=sorted(ARCHITECTURES), default="v1")
    parser.add_argument("--detail", type=float, default=0.0, help="weight of the detail loss")
    parser.add_argument("--real", type=Path, help="directory of real photographs for the adversarial loss")
    parser.add_argument("--adversarial", type=float, default=0.0, help="weight of the adversarial loss")
    parser.add_argument("--adversarial-start", type=int, default=2000, help="step from which it counts")
    parser.add_argument("--discriminator", choices=["rgb", "texture"], default="rgb",
                        help="what the discriminator sees of a crop")
    parser.add_argument("--r1", type=float, default=1.0, help="weight of the discriminator's R1 penalty")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    names = sorted(p.stem[3:] for p in args.dataset.glob("hf_*.ppm"))
    train, val = [n for n in names if int(n) % 10 != 9], [n for n in names if int(n) % 10 == 9]
    print(f"{len(train)} training views, {len(val)} validation views on {device}")

    def load(group: list[str]):
        tensors = [to_tensors(load_view(args.dataset, n)) for n in group]
        return [t[0] for t in tensors], [t[1] for t in tensors], [t[2] for t in tensors]

    tx, ty, tm = load(train)
    vx, vy, vm = load(val)

    model = ARCHITECTURES[args.arch]().to(device)
    (args.out_dir / "config.json").write_text(json.dumps({"arch": args.arch, "detail": args.detail,
                                                          "adversarial": args.adversarial,
                                                          "discriminator": args.discriminator}))
    discriminator = real = None
    if args.adversarial > 0.0:
        real = RealCrops(args.real)
        discriminator = PatchDiscriminator(args.discriminator).to(device)
        d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.0, 0.99))
    print(f"{sum(p.numel() for p in model.parameters())} parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, s / 500) * 0.5 * (1.0 + math.cos(math.pi * s / args.steps)))
    rng = np.random.default_rng(1)

    def validate() -> tuple[float, float, float, float]:
        model.eval()
        before = after = detail_before = detail_after = 0.0
        with torch.no_grad():
            for x, y, m in zip(vx, vy, vm):
                out = model(x.unsqueeze(0).to(device)).clamp(0.0, 1.0).cpu()
                before += block_error(x[:3].unsqueeze(0), y.unsqueeze(0), m.unsqueeze(0))
                after += block_error(out, y.unsqueeze(0), m.unsqueeze(0))
                detail_before += detail_error(x[:3].unsqueeze(0), y.unsqueeze(0), m.unsqueeze(0))
                detail_after += detail_error(out, y.unsqueeze(0), m.unsqueeze(0))
        model.train()
        n = max(len(vx), 1)
        return before / n, after / n, detail_before / n, detail_after / n

    history = []
    start = time.time()
    for step in range(1, args.steps + 1):
        x, y, m = crops(tx, ty, tm, args.crop, args.batch, rng)
        x, y, m = x.to(device), y.to(device), m.to(device)
        out = model(x)
        # The far trees weigh most; elsewhere the network learns to leave the image alone.
        weight = 0.2 + m
        loss = ((out - y).abs() * weight).mean() + 0.5 * gradient_l1(out, y)
        if args.detail > 0.0:
            loss = loss + args.detail * detail_loss(out, y, m)
        adversarial = 0.0
        if discriminator is not None and step >= args.adversarial_start:
            # The discriminator learns real vegetation from enhanced far trees (non-saturating loss, both seen through
            # the same DiffAugment, R1 on real crops every 16 steps); the enhancer learns to make its far trees look
            # real to it.
            fake = fake_crops(out, m, 2 * args.batch, rng)
            real_batch = real.sample(2 * args.batch, rng).to(device)
            d_loss = (F.softplus(-discriminator(diff_augment(real_batch))).mean()
                      + F.softplus(discriminator(diff_augment(fake.detach()))).mean())
            if step % 16 == 0:
                real_batch.requires_grad_(True)
                (grad,) = torch.autograd.grad(discriminator(diff_augment(real_batch)).sum(), real_batch,
                                              create_graph=True)
                d_loss = d_loss + args.r1 * 16 * 0.5 * grad.pow(2).sum(dim=(1, 2, 3)).mean()
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            adversarial = F.softplus(-discriminator(diff_augment(fake))).mean()
            loss = loss + args.adversarial * adversarial
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % 500 == 0 or step == args.steps:
            before, after, detail_before, detail_after = validate()
            history.append({"step": step, "loss": loss.item(), "val_input": before, "val_output": after,
                            "val_detail_input": detail_before, "val_detail_output": detail_after})
            print(f"step {step}: loss {loss.item():.4f}; validation block error input {before:.4f}, "
                  f"output {after:.4f} ({100.0 * (1.0 - after / max(before, 1e-9)):.1f} % lower); detail error "
                  f"input {detail_before:.3f}, output {detail_after:.3f}; {time.time() - start:.0f} s", flush=True)

    torch.save(model.state_dict(), args.out_dir / "enhancer.pt")
    (args.out_dir / "history.json").write_text(json.dumps(history, indent=1))
    model.eval().cpu()
    for height, width in ((720, 1280),):
        dummy = torch.zeros(1, INPUT_CHANNELS, height, width)
        path = args.out_dir / f"enhancer_{width}x{height}.onnx"
        torch.onnx.export(model, dummy, path, input_names=["input"], output_names=["output"], opset_version=17,
                          dynamo=False)
        # OnnxNetwork needs the shapes of the intermediate tensors.
        onnx.save(onnx.shape_inference.infer_shapes(constants_as_initializers(onnx.load(path))), path)
    print(f"model in {args.out_dir}")


if __name__ == "__main__":
    main()
