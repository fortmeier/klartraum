# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "torchvision", "numpy", "pillow", "onnx", "pyyaml"]
# ///

"""Trains an image-to-image network that turns renderings of the bonsai scene into its watercolor style.

Reads pairs by file name: the renderings (techdemo001_example --dataset, 1280 x 720) and their watercolor versions
(any size of the same aspect). Every eighth pair is held out. The network works at half the rendering's size. It sees
the rendering and, with --gbuffers (a techdemo001_example --dataset directory with class_XXXX.png and depth_XXXX.png),
the surfaces' classes (one channel per class) and their distance from the camera, and predicts the watercolor image.
It has no normalization layers over the whole image (whose statistics would shift as the camera moves) and no noise
inputs, both for temporal stability, and uses only operators klartraum's OnnxNetwork runs.

Models: "unet" (the G-buffers, if any, as extra input channels) and "spade" (the G-buffers modulate the decoder's
features, after Park et al. 2019, without its normalization). Losses: L1 and a VGG perceptual loss; optionally a
patch discriminator (--gan) and, with the G-buffers' sequences (techdemo001_example --sequences), a temporal loss
(--temporal W): between consecutive frames, the output may change at a surface point only as much as the rendering
does, the points matched by reprojecting each pixel with its depth and the two cameras (occluded ones left out).

Writes the weights, the ONNX model, held-out numbers (L1 and perceptual distance of the output and of two baselines:
the rendering itself and a per-channel affine colour transfer fitted on the training pairs; with sequences, the
flicker ratio: the output's change between consecutive frames at matched points over the rendering's, on held-out
sequences) and held-out comparison images.

    uv run scripts/neural/train_watercolor.py --renders DIR --styles DIR --out DIR [--gbuffers DIR] [--model unet]
        [--steps 4000] [--gan] [--temporal 1.0]
"""

from __future__ import annotations

import argparse
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

WIDTH, HEIGHT = 640, 360  # the network's working size, half the rendering's
# The class colours of the G-buffer (tree::treeClassColor()): background, pot, soil, stem, leaf, object, lantern.
PALETTE = np.array([[0, 0, 0], [0.2, 0.4, 1], [0.6, 0.4, 0], [1, 0, 0], [0, 1, 0], [1, 0.6, 0], [1, 1, 1]], np.float32)
CLASSES = len(PALETTE)


def load_image(path: Path, resample=Image.BOX) -> np.ndarray:
    """An image at the network's size; BOX averages the 2 x 2 blocks of a rendering, as klartraum's StyleTransfer
    does."""
    return np.asarray(Image.open(path).convert("RGB").resize((WIDTH, HEIGHT), resample), np.float32) / 255.0


def load_gbuffers(folder: Path, suffix: str) -> np.ndarray:
    """The classes as one channel per class (nearest palette colour) and the depth grey, HWC."""
    colours = load_image(folder / f"class_{suffix}.png", Image.NEAREST)
    nearest = np.argmin(((colours[:, :, None, :] - PALETTE[None, None]) ** 2).sum(-1), -1)
    onehot = np.eye(CLASSES, dtype=np.float32)[nearest]
    depth = load_image(folder / f"depth_{suffix}.png")[..., :1]
    return np.concatenate([onehot, depth], -1)


def load_pairs(renders: Path, styles: Path, gbuffers: Path | None,
               with_gbuffers: bool = True) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """The pairs present in both folders as (name, input, style), float32 HWC at WIDTH x HEIGHT: the input is the
    rendering, with the G-buffers appended if given."""
    pairs = []
    for style_path in sorted(styles.glob("*.png")):
        render_path = renders / style_path.name
        if not render_path.exists():
            continue
        x = load_image(render_path)
        if gbuffers is not None and with_gbuffers:
            x = np.concatenate([x, load_gbuffers(gbuffers, style_path.stem.split("_")[-1])], -1)
        pairs.append((style_path.stem, x, load_image(style_path, Image.LANCZOS)))
    return pairs


def matrix(values: list[float]) -> np.ndarray:
    return np.array(values, np.float64).reshape(4, 4)


def reprojection(later: dict, earlier: dict, depth_later: np.ndarray, depth_earlier: np.ndarray, near: float,
                 far: float) -> tuple[np.ndarray, np.ndarray]:
    """For every pixel of the later frame, where the same surface point lies in the earlier frame (a grid_sample grid
    in [-1, 1]) and whether it is seen there too (not background, not occluded)."""
    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH]
    ndc = np.stack([(xs + 0.5) / WIDTH * 2 - 1, (ys + 0.5) / HEIGHT * 2 - 1], -1)
    inverse = np.linalg.inv(matrix(later["camera"]["projection"]) @ matrix(later["camera"]["view"]))
    far_points = np.concatenate([ndc, np.ones((HEIGHT, WIDTH, 1)), np.ones((HEIGHT, WIDTH, 1))], -1) @ inverse.T
    eye = np.array(later["camera"]["position"])
    direction = far_points[..., :3] / far_points[..., 3:] - eye
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    distance = near + depth_later * (far - near)
    points = eye + direction * distance[..., None]
    clip = np.concatenate([points, np.ones((HEIGHT, WIDTH, 1))], -1) @ (
        matrix(earlier["camera"]["projection"]) @ matrix(earlier["camera"]["view"])).T
    grid = clip[..., :2] / clip[..., 3:]
    # Seen in the earlier frame: inside it, at the distance its depth shows there.
    px = np.clip(((grid[..., 0] * 0.5 + 0.5) * WIDTH).astype(int), 0, WIDTH - 1)
    py = np.clip(((grid[..., 1] * 0.5 + 0.5) * HEIGHT).astype(int), 0, HEIGHT - 1)
    seen_distance = near + depth_earlier[py, px] * (far - near)
    to_earlier = np.linalg.norm(points - np.array(earlier["camera"]["position"]), axis=-1)
    mask = (depth_later < 0.99) & (np.abs(grid) < 1).all(-1) & (np.abs(seen_distance - to_earlier) < 0.1)
    return grid.astype(np.float32), mask.astype(np.float32)


def load_sequences(gbuffers: Path, with_gbuffers: bool) -> list[dict]:
    """The sequences' frames as inputs and, for each consecutive pair, the reprojection of the later into the
    earlier frame."""
    description = yaml.safe_load((gbuffers / "sequences" / "sequences.yaml").read_text())
    near, far = description["depth_near"], description["depth_far"]
    sequences = []
    for sequence in description["sequences"]:
        folder = gbuffers / "sequences" / sequence["name"]
        frames = sequence["frames"]
        inputs, depths = [], []
        for frame in frames:
            suffix = frame["file"].split("_")[-1].split(".")[0]
            x = load_image(folder / frame["file"])
            g = load_gbuffers(folder, suffix)
            inputs.append(np.concatenate([x, g], -1) if with_gbuffers else x)
            depths.append(g[..., -1])
        pairs = [reprojection(frames[f + 1], frames[f], depths[f + 1], depths[f], near, far)
                 for f in range(len(frames) - 1)]
        sequences.append({"inputs": np.stack(inputs), "grids": np.stack([p[0] for p in pairs]),
                          "masks": np.stack([p[1] for p in pairs])})
    return sequences


def block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(), nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU())


def down(x: torch.Tensor) -> torch.Tensor:
    # Halving bilinearly averages each 2 x 2 block, as average pooling does; it exports as a Resize, which
    # klartraum's OnnxNetwork runs.
    return F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)


def up(x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(x, scale_factor=2.0, mode="nearest")


class UNet(nn.Module):
    """A U-Net with three levels, ReLU, nearest upsampling and convolutions, without normalization; the output is a
    sigmoid image."""

    def __init__(self, inputs: int = 3, width: tuple[int, int, int, int] = (32, 64, 96, 128)) -> None:
        super().__init__()
        a, b, c, d = width
        self.enc0, self.enc1, self.enc2, self.mid = block(inputs, a), block(a, b), block(b, c), block(c, d)
        self.dec2, self.dec1, self.dec0 = block(d + c, c), block(c + b, b), block(b + a, a)
        self.out = nn.Conv2d(a + inputs, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x)
        e1 = self.enc1(down(e0))
        e2 = self.enc2(down(e1))
        m = self.mid(down(e2))
        d2 = self.dec2(torch.cat([up(m), e2], 1))
        d1 = self.dec1(torch.cat([up(d2), e1], 1))
        d0 = self.dec0(torch.cat([up(d1), e0], 1))
        return torch.sigmoid(self.out(torch.cat([d0, x], 1)))


class Modulation(nn.Module):
    """Scales and shifts features per pixel by the G-buffers (SPADE without the normalization)."""

    def __init__(self, features: int, conditions: int, hidden: int = 32) -> None:
        super().__init__()
        self.shared = nn.Sequential(nn.Conv2d(conditions, hidden, 3, padding=1), nn.ReLU())
        self.scale = nn.Conv2d(hidden, features, 3, padding=1)
        self.shift = nn.Conv2d(hidden, features, 3, padding=1)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        h = self.shared(g)
        return x * (1 + self.scale(h)) + self.shift(h)


class SpadeUNet(UNet):
    """The U-Net on the rendering, its decoder features modulated by the G-buffers at every level."""

    def __init__(self, conditions: int, width: tuple[int, int, int, int] = (32, 64, 96, 128)) -> None:
        super().__init__(3, width)
        a, b, c, d = width
        self.conditions = conditions
        self.mod2, self.mod1, self.mod0 = Modulation(c, conditions), Modulation(b, conditions), Modulation(a, conditions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb, g = x[:, :3], x[:, 3:]
        g1, g2 = down(g), down(down(g))
        e0 = self.enc0(rgb)
        e1 = self.enc1(down(e0))
        e2 = self.enc2(down(e1))
        m = self.mid(down(e2))
        d2 = self.mod2(self.dec2(torch.cat([up(m), e2], 1)), g2)
        d1 = self.mod1(self.dec1(torch.cat([up(d2), e1], 1)), g1)
        d0 = self.mod0(self.dec0(torch.cat([up(d1), e0], 1)), g)
        return torch.sigmoid(self.out(torch.cat([d0, rgb], 1)))


class Perceptual(nn.Module):
    """Distance of VGG16 features (relu1_2, relu2_2, relu3_3) of two images in [0, 1]."""

    def __init__(self) -> None:
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
        for p in features.parameters():
            p.requires_grad_(False)
        self.slices = nn.ModuleList([features[:4], features[4:9], features[9:16]])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x, y = (a - self.mean) / self.std, (b - self.mean) / self.std
        loss = 0.0
        for s in self.slices:
            x, y = s(x), s(y)
            loss = loss + F.l1_loss(x, y)
        return loss


class PatchDiscriminator(nn.Module):
    def __init__(self, inputs: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(inputs, 64, 4, 2, 1), nn.LeakyReLU(0.2), nn.Conv2d(64, 128, 4, 2, 1),
                                 nn.LeakyReLU(0.2), nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2),
                                 nn.Conv2d(256, 1, 3, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def crops(inputs: torch.Tensor, targets: torch.Tensor, size: int, count: int,
          generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Random crops (with random horizontal flips and a little exposure jitter of the rendering) of random pairs."""
    n, _, h, w = inputs.shape
    xs, ys = [], []
    for _ in range(count):
        i = int(torch.randint(n, (1,), generator=generator))
        top = int(torch.randint(h - size + 1, (1,), generator=generator))
        left = int(torch.randint(w - size + 1, (1,), generator=generator))
        x = inputs[i, :, top:top + size, left:left + size]
        y = targets[i, :, top:top + size, left:left + size]
        if torch.rand(1, generator=generator).item() < 0.5:
            x, y = x.flip(-1), y.flip(-1)
        gain = 1.0 + 0.1 * (torch.rand(1, generator=generator).item() - 0.5)
        x = torch.cat([(x[:3] * gain).clamp(0, 1), x[3:]], 0)
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)


def warp(image: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=False)


def temporal_change(model: nn.Module, sequence: dict, f: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """The change of the output and of the rendering from frame f to f + 1 at matched points, and the mask."""
    x0 = torch.from_numpy(sequence["inputs"][f]).permute(2, 0, 1)[None].to(device)
    x1 = torch.from_numpy(sequence["inputs"][f + 1]).permute(2, 0, 1)[None].to(device)
    grid = torch.from_numpy(sequence["grids"][f])[None].to(device)
    mask = torch.from_numpy(sequence["masks"][f])[None, None].to(device)
    o0, o1 = model(x0), model(x1)
    return o1 - warp(o0, grid), x1[:, :3] - warp(x0[:, :3], grid), mask


def export_onnx(model: nn.Module, example: torch.Tensor, path: Path) -> None:
    """Exports the model as klartraum's OnnxNetwork reads it: the constants as initializers, every tensor's shape
    inferred."""
    import onnx

    from train_tree_enhancer import constants_as_initializers

    torch.onnx.export(model, example, path, input_names=["input"], output_names=["output"], opset_version=17,
                      dynamo=False)
    onnx.save(onnx.shape_inference.infer_shapes(constants_as_initializers(onnx.load(path))), path)


def colour_transfer(train_inputs: np.ndarray, train_targets: np.ndarray):
    """A per-channel affine map fitted by least squares on the training pairs: the colour-only baseline."""
    coefficients = []
    for c in range(3):
        a, b = np.polyfit(train_inputs[..., c].ravel(), train_targets[..., c].ravel(), 1)
        coefficients.append((a, b))
    return lambda image: np.stack([np.clip(a * image[..., c] + b, 0, 1) for c, (a, b) in enumerate(coefficients)],
                                  -1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--renders", type=Path, required=True)
    parser.add_argument("--styles", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gbuffers", type=Path, help="the dataset directory with class_ and depth_ images")
    parser.add_argument("--model", choices=["unet", "spade"], default="unet")
    parser.add_argument("--rgb-only", action="store_true",
                        help="the rendering alone as input, --gbuffers only for the temporal loss's sequences")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--crop", type=int, default=192)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--gan", action="store_true", help="add a patch discriminator")
    parser.add_argument("--temporal", type=float, default=0.0, help="weight of the temporal loss (needs --gbuffers)")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(args.seed)

    pairs = load_pairs(args.renders, args.styles, args.gbuffers, not args.rgb_only)
    held = [p for i, p in enumerate(pairs) if i % 8 == 7]
    train = [p for i, p in enumerate(pairs) if i % 8 != 7]
    print(f"{len(pairs)} pairs: {len(train)} for training, {len(held)} held out ({', '.join(p[0] for p in held)})")
    to_tensor = lambda images: torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).contiguous()
    train_x, train_y = to_tensor([p[1] for p in train]), to_tensor([p[2] for p in train])
    held_x, held_y = to_tensor([p[1] for p in held]), to_tensor([p[2] for p in held])
    channels = train_x.shape[1]
    sequences = []
    if args.gbuffers is not None and (args.gbuffers / "sequences" / "sequences.yaml").exists():
        sequences = load_sequences(args.gbuffers, not args.rgb_only)
        print(f"{len(sequences)} sequences, the last {max(len(sequences) // 5, 1)} held out")
    held_sequences = sequences[-max(len(sequences) // 5, 1):] if sequences else []
    train_sequences = sequences[:len(sequences) - len(held_sequences)]

    if args.model == "spade":
        model = SpadeUNet(channels - 3).to(device)
    else:
        model = UNet(channels).to(device)
    perceptual = Perceptual().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))
    discriminator = PatchDiscriminator(channels + 3).to(device) if args.gan else None
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999)) if args.gan else None
    generator = torch.Generator().manual_seed(args.seed)
    started = time.time()
    for step in range(args.steps):
        x, y = crops(train_x, train_y, args.crop, args.batch, generator)
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = F.l1_loss(out, y) + 0.2 * perceptual(out, y)
        if discriminator is not None and step > args.steps // 4:
            loss = loss + 0.02 * F.softplus(-discriminator(torch.cat([x, out], 1))).mean()
        if args.temporal > 0 and train_sequences:
            sequence = train_sequences[int(torch.randint(len(train_sequences), (1,), generator=generator))]
            f = int(torch.randint(len(sequence["grids"]), (1,), generator=generator))
            change, rendering_change, mask = temporal_change(model, sequence, f, device)
            # The output may change at a surface point as the rendering does there, no more.
            loss = loss + args.temporal * (mask * (change - rendering_change).abs()).sum() / (3 * mask.sum() + 1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if discriminator is not None:
            real = discriminator(torch.cat([x, y], 1))
            fake = discriminator(torch.cat([x, out.detach()], 1))
            d_loss = F.softplus(-real).mean() + F.softplus(fake).mean()
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
        if step % 100 == 0 or step == args.steps - 1:
            print(f"step {step}: loss {loss.item():.4f} ({time.time() - started:.0f} s)", flush=True)

    torch.save(model.state_dict(), args.out / "watercolor.pt")
    # Held-out evaluation against the two baselines.
    model.eval()
    baseline = colour_transfer(np.stack([p[1][..., :3] for p in train]), np.stack([p[2] for p in train]))
    metrics = {"model": args.model, "inputs": channels, "gan": args.gan, "temporal": args.temporal,
               "pairs": len(pairs), "train": len(train), "held_out": [p[0] for p in held]}
    with torch.no_grad():
        outputs = torch.cat([model(held_x[i:i + 1].to(device)).cpu() for i in range(len(held))])
        colour = to_tensor([baseline(p[1][..., :3]) for p in held])
        for name, images in (("rendering", held_x[:, :3]), ("colour transfer", colour), ("network", outputs)):
            metrics[name] = {"l1": float(F.l1_loss(images, held_y)),
                             "perceptual": float(perceptual(images.to(device), held_y.to(device)))}
        if held_sequences:
            output_change, rendering_change = 0.0, 0.0
            for sequence in held_sequences:
                for f in range(len(sequence["grids"])):
                    change, r_change, mask = temporal_change(model, sequence, f, device)
                    output_change += float((mask * change.abs()).sum() / (3 * mask.sum() + 1))
                    rendering_change += float((mask * r_change.abs()).sum() / (3 * mask.sum() + 1))
            metrics["flicker ratio"] = output_change / max(rendering_change, 1e-9)
    print(json.dumps(metrics, indent=2))
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    rows = [torch.cat([held_x[i, :3], colour[i], outputs[i], held_y[i]], 2) for i in range(len(held))]
    grid = (torch.cat(rows, 1).permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(grid).save(args.out / "held_out.png")
    model.cpu()
    export_onnx(model, torch.zeros(1, channels, HEIGHT, WIDTH), args.out / "watercolor.onnx")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
