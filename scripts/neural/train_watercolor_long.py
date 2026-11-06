# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "torchvision", "numpy", "pillow", "onnx", "pyyaml"]
# ///

"""Trains the watercolor network of train_watercolor.py at the rendering's full size, for hours, with checkpoints.

The network (a U-Net of five levels, no whole-image normalization, no noise) sees the rendering, its classes (one
channel per class) and its depth (techdemo001_example --dataset) at full size (or half with --scale 2) and predicts
the watercolor image (the style images, resized to the rendering's size). Training takes random crops; the losses are
L1, a VGG perceptual loss, a patch discriminator and the temporal loss of train_watercolor.py on windows of the
dataset's sequences. The learning rate falls along a cosine over the time budget (--hours). Every --checkpoint hours
and at the end, a checkpoint goes to <out>/checkpoint_<hours>h/: the weights, the ONNX model at the frame's size, the
held-out numbers (L1, perceptual distance, flicker ratio, as train_watercolor.py; every 13th pair is held out) and
held-out comparison images; every --video hours and at the end, training pauses while
techdemo001_example renders the demo with that model (--style) and ffmpeg encodes it into the checkpoint.

    uv run scripts/neural/train_watercolor_long.py --renders DIR --styles DIR --gbuffers DIR --out DIR
        [--hours 8] [--checkpoint 1.5] [--video 3]
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from train_watercolor import PALETTE, PatchDiscriminator, Perceptual, export_onnx  # noqa: E402

CLASSES = len(PALETTE)
REPO = Path(__file__).resolve().parents[2]


class DeepUNet(nn.Module):
    """A U-Net with len(widths) - 1 levels, ReLU, bilinear halving, nearest upsampling and convolutions, without
    normalization; the output is a sigmoid image."""

    def __init__(self, inputs: int, widths: tuple[int, ...] = (32, 64, 128, 192, 256)) -> None:
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(), nn.Conv2d(cout, cout, 3, padding=1),
                                 nn.ReLU())

        self.encoders = nn.ModuleList([block(inputs, widths[0])] +
                                      [block(widths[i], widths[i + 1]) for i in range(len(widths) - 2)])
        self.mid = block(widths[-2], widths[-1])
        self.decoders = nn.ModuleList([block(widths[i + 1] + widths[i], widths[i])
                                       for i in reversed(range(len(widths) - 1))])
        self.out = nn.Conv2d(widths[0] + inputs, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        h = x
        for i, encoder in enumerate(self.encoders):
            h = encoder(h if i == 0 else F.interpolate(h, scale_factor=0.5, mode="bilinear", align_corners=False))
            skips.append(h)
        h = self.mid(F.interpolate(h, scale_factor=0.5, mode="bilinear", align_corners=False))
        for decoder, skip in zip(self.decoders, reversed(skips)):
            h = decoder(torch.cat([F.interpolate(h, scale_factor=2.0, mode="nearest"), skip], 1))
        return torch.sigmoid(self.out(torch.cat([h, x], 1)))


def read_rgb(path: Path, size: tuple[int, int], resample) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize(size, resample), np.uint8)


def read_view(folder: Path, suffix: str, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The rendering (rgb), the class index and the depth of a view, uint8, at `size`: averaged over blocks and the
    class of the pixel StyleTransfer picks, as it makes its input."""
    rgb = read_rgb(folder / f"image_{suffix}.png", size, Image.BOX)
    colours = read_rgb(folder / f"class_{suffix}.png", size, Image.NEAREST).astype(np.float32) / 255.0
    classes = np.argmin(((colours[:, :, None] - PALETTE[None, None]) ** 2).sum(-1), -1).astype(np.uint8)
    depth = read_rgb(folder / f"depth_{suffix}.png", size, Image.BOX)[..., 0]
    return rgb, classes, depth


def to_input(rgb: np.ndarray, classes: np.ndarray, depth: np.ndarray) -> torch.Tensor:
    """The network's input (11, H, W) from a view's uint8 arrays."""
    x = np.concatenate([rgb.astype(np.float32) / 255.0, np.eye(CLASSES, dtype=np.float32)[classes],
                        depth[..., None].astype(np.float32) / 255.0], -1)
    return torch.from_numpy(x).permute(2, 0, 1).contiguous()


def matrix(values: list[float]) -> np.ndarray:
    return np.array(values, np.float64).reshape(4, 4)


def reproject_window(later: dict, earlier: dict, depth_later: np.ndarray, depth_earlier: np.ndarray, near: float,
                     far: float, top: int, left: int, size: int, width: int, height: int):
    """For the pixels of a size x size window of the later frame: where the same surface points lie in the earlier
    frame (pixel coordinates) and whether they are seen there (not background, not occluded)."""
    ys, xs = np.mgrid[top:top + size, left:left + size]
    ndc = np.stack([(xs + 0.5) / width * 2 - 1, (ys + 0.5) / height * 2 - 1], -1)
    inverse = np.linalg.inv(matrix(later["camera"]["projection"]) @ matrix(later["camera"]["view"]))
    ones = np.ones((size, size, 1))
    far_points = np.concatenate([ndc, ones, ones], -1) @ inverse.T
    eye = np.array(later["camera"]["position"])
    direction = far_points[..., :3] / far_points[..., 3:] - eye
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    d = depth_later[top:top + size, left:left + size].astype(np.float64) / 255.0
    points = eye + direction * (near + d * (far - near))[..., None]
    clip = np.concatenate([points, ones], -1) @ (matrix(earlier["camera"]["projection"]) @
                                                 matrix(earlier["camera"]["view"])).T
    ndc_earlier = clip[..., :2] / clip[..., 3:]
    px = (ndc_earlier[..., 0] * 0.5 + 0.5) * width - 0.5
    py = (ndc_earlier[..., 1] * 0.5 + 0.5) * height - 0.5
    inside = (px >= 0) & (px <= width - 1) & (py >= 0) & (py <= height - 1)
    ix = np.clip(np.round(px).astype(int), 0, width - 1)
    iy = np.clip(np.round(py).astype(int), 0, height - 1)
    seen = near + depth_earlier[iy, ix].astype(np.float64) / 255.0 * (far - near)
    to_earlier = np.linalg.norm(points - np.array(earlier["camera"]["position"]), axis=-1)
    mask = (d < 0.99) & inside & (np.abs(seen - to_earlier) < 0.15)
    return px, py, mask


class Data:
    def __init__(self, renders: Path, styles: Path, gbuffers: Path, scale: int) -> None:
        self.width, self.height = 1280 // scale, 720 // scale
        size = (self.width, self.height)
        self.pairs = []
        for style in sorted(styles.glob("*.png")):
            suffix = style.stem.split("_")[-1]
            if not (gbuffers / f"class_{suffix}.png").exists():
                continue
            rgb, classes, depth = read_view(gbuffers, suffix, size)
            self.pairs.append({"name": style.stem, "rgb": rgb, "classes": classes, "depth": depth,
                               "style": read_rgb(style, size, Image.LANCZOS)})
        self.held = [p for i, p in enumerate(self.pairs) if i % 13 == 12]
        self.train = [p for i, p in enumerate(self.pairs) if i % 13 != 12]
        description = yaml.safe_load((gbuffers / "sequences" / "sequences.yaml").read_text())
        self.near, self.far = description["depth_near"], description["depth_far"]
        self.sequences = []
        for sequence in description["sequences"]:
            folder = gbuffers / "sequences" / sequence["name"]
            frames = []
            for frame in sequence["frames"]:
                suffix = frame["file"].split("_")[-1].split(".")[0]
                rgb, classes, depth = read_view(folder, suffix, size)
                frames.append({"rgb": rgb, "classes": classes, "depth": depth, "camera": frame})
            self.sequences.append(frames)
        held_count = max(len(self.sequences) // 5, 1)
        self.held_sequences = self.sequences[-held_count:]
        self.train_sequences = self.sequences[:-held_count]

    def crop_batch(self, count: int, size: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for _ in range(count):
            p = self.train[rng.integers(len(self.train))]
            top, left = rng.integers(self.height - size + 1), rng.integers(self.width - size + 1)
            window = (slice(top, top + size), slice(left, left + size))
            x = to_input(p["rgb"][window], p["classes"][window], p["depth"][window])
            y = torch.from_numpy(p["style"][window].astype(np.float32) / 255.0).permute(2, 0, 1)
            if rng.random() < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            x[:3] = (x[:3] * (1.0 + 0.1 * (rng.random() - 0.5))).clamp(0, 1)
            xs.append(x)
            ys.append(y)
        return torch.stack(xs), torch.stack(ys)

    def temporal_window(self, sequences: list, size: int, rng: np.random.Generator):
        """A window of a frame and the matching region of the frame before: the two inputs, the grid that samples
        the earlier output at the later window's points, and their mask."""
        frames = sequences[rng.integers(len(sequences))]
        f = rng.integers(len(frames) - 1)
        earlier, later = frames[f], frames[f + 1]
        top, left = rng.integers(self.height - size + 1), rng.integers(self.width - size + 1)
        px, py, mask = reproject_window(later["camera"], earlier["camera"], later["depth"], earlier["depth"],
                                        self.near, self.far, top, left, size, self.width, self.height)
        # The region of the earlier frame those points fall in, a little larger, aligned to 32 pixels.
        x0 = int(max(0, np.floor(px[mask].min()) - 8)) if mask.any() else left
        y0 = int(max(0, np.floor(py[mask].min()) - 8)) if mask.any() else top
        x1 = int(min(self.width, np.ceil(px[mask].max()) + 9)) if mask.any() else left + size
        y1 = int(min(self.height, np.ceil(py[mask].max()) + 9)) if mask.any() else top + size
        w = min(self.width - (x0 - x0 % 32), (x1 - x0 + x0 % 32 + 31) // 32 * 32)
        h = min(self.height - (y0 - y0 % 32), (y1 - y0 + y0 % 32 + 31) // 32 * 32)
        x0, y0 = x0 - x0 % 32, y0 - y0 % 32
        w, h = w - w % 32, h - h % 32
        region = (slice(y0, y0 + h), slice(x0, x0 + w))
        window = (slice(top, top + size), slice(left, left + size))
        x_earlier = to_input(earlier["rgb"][region], earlier["classes"][region], earlier["depth"][region])
        x_later = to_input(later["rgb"][window], later["classes"][window], later["depth"][window])
        grid = np.stack([(px - x0 + 0.5) / w * 2 - 1, (py - y0 + 0.5) / h * 2 - 1], -1).astype(np.float32)
        return x_earlier, x_later, torch.from_numpy(grid), torch.from_numpy(mask.astype(np.float32))


def temporal_change(model: nn.Module, data: Data, sequences: list, size: int, rng: np.random.Generator,
                    device: torch.device):
    x0, x1, grid, mask = data.temporal_window(sequences, size, rng)
    x0, x1, grid, mask = x0[None].to(device), x1[None].to(device), grid[None].to(device), mask[None, None].to(device)
    o0, o1 = model(x0), model(x1)
    warp = lambda image: F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return o1 - warp(o0), x1[:, :3] - warp(x0[:, :3]), mask


def evaluate(model: nn.Module, data: Data, perceptual: nn.Module, device: torch.device, folder: Path) -> dict:
    model.eval()
    metrics = {"held_out": [p["name"] for p in data.held]}
    with torch.no_grad():
        l1 = l1_rendering = per = per_rendering = 0.0
        rows = []
        for p in data.held:
            x = to_input(p["rgb"], p["classes"], p["depth"])[None].to(device)
            y = torch.from_numpy(p["style"].astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
            out = model(x)
            l1 += float(F.l1_loss(out, y)) / len(data.held)
            l1_rendering += float(F.l1_loss(x[:, :3], y)) / len(data.held)
            per += float(perceptual(out, y)) / len(data.held)
            per_rendering += float(perceptual(x[:, :3], y)) / len(data.held)
            rows.append(torch.cat([x[0, :3], out[0], y[0]], 2).cpu())
        metrics["network"] = {"l1": l1, "perceptual": per}
        metrics["rendering"] = {"l1": l1_rendering, "perceptual": per_rendering}
        rng = np.random.default_rng(5)
        output_change = rendering_change = 0.0
        for _ in range(40):
            change, r_change, mask = temporal_change(model, data, data.held_sequences, 384, rng, device)
            output_change += float((mask * change.abs()).sum() / (3 * mask.sum() + 1))
            rendering_change += float((mask * r_change.abs()).sum() / (3 * mask.sum() + 1))
        metrics["flicker ratio"] = output_change / max(rendering_change, 1e-9)
    # Held-out comparison (rendering, output, target), and a full-size crop of each output for the texture.
    for k in range(0, len(rows), 5):
        grid = (torch.cat(rows[k:k + 5], 1).permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(grid).save(folder / f"held_out_{k // 5}.jpg", quality=90)
    model.train()
    return metrics


def checkpoint(model, data, perceptual, device, out: Path, hours: float, step: int, video: bool, args) -> None:
    folder = out / f"checkpoint_{hours:.1f}h"
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), folder / "watercolor.pt")
    metrics = evaluate(model, data, perceptual, device, folder)
    metrics.update({"hours": hours, "step": step})
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"checkpoint {hours:.1f} h, step {step}: {json.dumps({k: metrics[k] for k in ('network', 'flicker ratio')})}",
          flush=True)
    export = DeepUNet(11).eval()
    export.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
    export_onnx(export, torch.zeros(1, 11, data.height, data.width), folder / "watercolor.onnx")
    if video:
        # The demo with this model, the training paused (the GPU is shared).
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()
        render = folder / "demo"
        subprocess.run([str(REPO / "build/examples/techdemo001_example"), "--style", str(folder / "watercolor.onnx"),
                        "--out-dir", str(render)], cwd=REPO, check=False, stdout=subprocess.DEVNULL)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "30", "-i",
                        str(render / "frames" / "frame_%04d.ppm"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf",
                        "18", str(folder / f"techdemo001_watercolor_{hours:.1f}h.mp4")], check=False)
        shutil.rmtree(render / "frames", ignore_errors=True)
        print(f"video {folder / f'techdemo001_watercolor_{hours:.1f}h.mp4'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--renders", type=Path, required=True, help="unused, the renderings come from --gbuffers")
    parser.add_argument("--styles", type=Path, required=True)
    parser.add_argument("--gbuffers", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--checkpoint", type=float, default=1.5)
    parser.add_argument("--video", type=float, default=3.0)
    parser.add_argument("--crop", type=int, default=384)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--temporal", type=float, default=1.0)
    parser.add_argument("--gan", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data = Data(args.renders, args.styles, args.gbuffers, args.scale)
    print(f"{len(data.pairs)} pairs at {data.width} x {data.height}: {len(data.train)} for training, "
          f"{len(data.held)} held out; {len(data.sequences)} sequences, {len(data.held_sequences)} held out", flush=True)
    model = DeepUNet(11).to(device)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters", flush=True)
    perceptual = Perceptual().to(device)
    discriminator = PatchDiscriminator(14).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    budget = args.hours * 3600.0
    started = time.time()
    paused = 0.0  # seconds spent on checkpoints, not counted as training time
    next_checkpoint, next_video = args.checkpoint, args.video
    step = 0
    while True:
        trained = time.time() - started - paused
        if trained >= budget:
            break
        lr = 1e-5 + 0.5 * (2e-4 - 1e-5) * (1 + math.cos(math.pi * trained / budget))
        for group in optimizer.param_groups + d_optimizer.param_groups:
            group["lr"] = lr
        x, y = data.crop_batch(args.batch, args.crop, rng)
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = F.l1_loss(out, y) + 0.2 * perceptual(out, y)
        if trained > 0.1 * budget:
            loss = loss + args.gan * F.softplus(-discriminator(torch.cat([x, out], 1))).mean()
        if args.temporal > 0:
            change, rendering_change, mask = temporal_change(model, data, data.train_sequences, 256, rng, device)
            loss = loss + args.temporal * (mask * (change - rendering_change).abs()).sum() / (3 * mask.sum() + 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        real = discriminator(torch.cat([x, y], 1))
        fake = discriminator(torch.cat([x, out.detach()], 1))
        d_loss = F.softplus(-real).mean() + F.softplus(fake).mean()
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        if step % 200 == 0:
            print(f"step {step}: loss {loss.item():.4f}, lr {lr:.2e} ({trained / 3600:.2f} h)", flush=True)
        step += 1
        hours = (time.time() - started - paused) / 3600.0
        if hours >= next_checkpoint and hours < args.hours:
            pause = time.time()
            checkpoint(model, data, perceptual, device, args.out, next_checkpoint, step, hours >= next_video, args)
            if hours >= next_video:
                next_video += args.video
            next_checkpoint += args.checkpoint
            paused += time.time() - pause
    checkpoint(model, data, perceptual, device, args.out, args.hours, step, True, args)
    print("done", flush=True)


if __name__ == "__main__":
    main()
