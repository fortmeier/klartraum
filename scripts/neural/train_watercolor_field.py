# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "torchvision", "numpy", "pillow", "pyyaml"]
# ///

"""Trains a watercolor stylization that is anchored in the world instead of the screen.

Every pixel is styled on its own from what the renderer knows about it, so the result changes between frames only as
much as the rendering does:

- a feature field on a grid in world space (two resolutions), looked up at the surface point the pixel shows (from
  its depth and camera): the style's washes and strokes stay on the objects;
- a learned paper texture fixed to the screen, like the paper of a painting;
- a few features of the geometry around the pixel (a small convolution on the classes and depth only, for outlines);
- the rendering's colour, the pixel's class and the view's conditions (time of day, season, light);

and a per-pixel MLP turns them into the watercolor colour. The data is the dataset of techdemo001_example --dataset
(renderings, classes, depth, cameras) and the style images paired by name; every 13th pair is held out. Results go to
--out: metrics.json (held-out L1 and perceptual distance as train_watercolor_long.py, and the flicker ratio on the
dataset's held-out sequences), held-out comparison images and the weights.

    uv run scripts/neural/train_watercolor_field.py --gbuffers DIR --styles DIR --out DIR [--steps 6000]
"""

from __future__ import annotations

import argparse
import json
import math
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
from train_watercolor import PALETTE, Perceptual  # noqa: E402

CLASSES = len(PALETTE)
WIDTH, HEIGHT = 1280, 720
CONDITIONS = 7


def read_rgb(path: Path, resample=Image.BOX) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if image.size != (WIDTH, HEIGHT):
        image = image.resize((WIDTH, HEIGHT), resample)
    return np.asarray(image, np.uint8)


def matrix(values: list[float]) -> np.ndarray:
    return np.array(values, np.float64).reshape(4, 4)


def conditions(view: dict) -> np.ndarray:
    """The view's conditions: time of day and season on circles, the directional light, exposure and background."""
    t, s = view["time_of_day"], view["season"] / 4.0
    light = view["light"]["intensity"] * (1.0 if view["light"]["source"] == "sun" else 0.3)
    return np.array([math.sin(2 * math.pi * t), math.cos(2 * math.pi * t), math.sin(2 * math.pi * s),
                     math.cos(2 * math.pi * s), light, view["exposure"] / 3.0, view["background"]], np.float32)


def surface_points(view: dict, distance: np.ndarray, window: tuple[slice, slice], width: int,
                   height: int) -> np.ndarray:
    """The world point every pixel of `window` of a width x height frame shows (h, w, 3), from its distance from the
    camera (`distance` of the window; background pixels get some point) and the view's camera."""
    camera = view["camera"]
    ys, xs = np.mgrid[window]
    ndc = np.stack([(xs + 0.5) / width * 2 - 1, (ys + 0.5) / height * 2 - 1], -1)
    inverse = np.linalg.inv(matrix(camera["projection"]) @ matrix(camera["view"]))
    ones = np.ones(ys.shape + (1,))
    far_points = np.concatenate([ndc, ones, ones], -1) @ inverse.T
    eye = np.array(camera["position"])
    direction = far_points[..., :3] / far_points[..., 3:] - eye
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    finite = np.where(np.isfinite(distance), distance, 0.0).astype(np.float64)
    return (eye + direction * finite[..., None]).astype(np.float32)


def decode_distance(classes: np.ndarray, coverage: np.ndarray, depth: np.ndarray, near: float,
                    far: float) -> np.ndarray:
    """The distance from the camera of every pixel's surface (infinity where it shows none), from the class indices,
    the coverage image (grey) and the depth image over black between near and far (three channels, the second and
    third one and two thirds of a step later: tree::treeGBufferColor()), as FieldStyle decodes them. A pixel shows a
    surface where its class is not background and the surfaces cover more than 5 % of it; partly covered pixels show
    the depth times the coverage, so it is divided out."""
    alpha = coverage.astype(np.float32) / 255.0
    mean = depth.astype(np.float32).mean(-1) / 255.0
    d = np.clip(mean / np.maximum(alpha, 1e-6) - 1.0 / 255.0, 0.0, 1.0)
    return np.where((classes > 0) & (alpha > 0.05), near + d * (far - near), np.inf).astype(np.float32)


def class_indices(colours: np.ndarray) -> np.ndarray:
    """The class of every pixel of a class image (uint8 RGB): the one whose colour is nearest."""
    c = colours.astype(np.float32) / 255.0
    return np.argmin(((c[:, :, None] - PALETTE[None, None]) ** 2).sum(-1), -1).astype(np.uint8)


def read_view(folder: Path, view: dict, near: float, far: float) -> dict:
    """A view's rendering, class indices, distance from the camera (from the field's depth if the dataset has it,
    infinity for background) and conditions (surface points: points())."""
    classes = class_indices(read_rgb(folder / view["classes"], Image.NEAREST))
    if "field_depth" in view:
        coverage = read_rgb(folder / view["coverage"], Image.NEAREST)[..., 0]
        depth = read_rgb(folder / view["field_depth"], Image.NEAREST)
        distance = decode_distance(classes, coverage, depth, view["field_depth_near"], view["field_depth_far"])
    else:
        grey = read_rgb(folder / view["depth"])[..., 0]
        distance = np.where((grey < 255) & (classes > 0), near + grey.astype(np.float32) / 255.0 * (far - near),
                            np.inf).astype(np.float32)
    return {"rgb": read_rgb(folder / view["file"]), "classes": classes, "distance": distance,
            "conditions": conditions(view), "view": view}


def depth_window(view: dict, low: np.ndarray, high: np.ndarray) -> tuple[float, float]:
    """The distances from the camera the field's box spans: the window the runtime's depth G-buffer covers
    (fieldDepthWindow())."""
    eye = np.array(view["camera"]["position"])
    corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    nearest = np.linalg.norm(eye - np.clip(eye, low, high))
    return max(float(nearest), 0.05), float(np.linalg.norm(corners - eye, axis=-1).max())


def points(view: dict, window: tuple[slice, slice] | None = None) -> np.ndarray:
    """The surface points of a window (default: all) of a read view."""
    height, width = view["distance"].shape
    window = window or (slice(0, height), slice(0, width))
    return surface_points(view["view"], view["distance"][window], window, width, height)


class Data:
    def __init__(self, gbuffers: Path, styles: Path) -> None:
        description = yaml.safe_load((gbuffers / "dataset.yaml").read_text())
        self.near, self.far = description["depth_near"], description["depth_far"]
        self.pairs = []
        for view in description["images"]:
            style = styles / view["file"]
            if style.exists():
                pair = read_view(gbuffers, view, self.near, self.far)
                pair["name"] = style.stem
                pair["style"] = read_rgb(style, Image.LANCZOS)
                self.pairs.append(pair)
        self.held = [p for i, p in enumerate(self.pairs) if i % 13 == 12]
        self.train = [p for i, p in enumerate(self.pairs) if i % 13 != 12]
        sequences = yaml.safe_load((gbuffers / "sequences" / "sequences.yaml").read_text())
        self.sequences = []
        for sequence in sequences["sequences"]:
            folder = gbuffers / "sequences" / sequence["name"]
            self.sequences.append([read_view(folder, frame, self.near, self.far) for frame in sequence["frames"]])
        self.held_sequences = self.sequences[-max(len(self.sequences) // 5, 1):]
        if "field_box_low" in description:
            # The box the dataset's field depth was drawn for.
            self.low = np.array(description["field_box_low"], np.float64)
            self.high = np.array(description["field_box_high"], np.float64)
        else:
            # The surface points of the training views, robustly, with a margin; well inside surfaces only, since at
            # outlines the depth is blended with the background's.
            inside = lambda p: np.isfinite(p["distance"]) & (np.abs(np.diff(p["distance"], axis=1, append=0)) < 0.2)
            seen = np.concatenate([points(p)[inside(p)][::50] for p in self.train])
            low, high = np.percentile(seen, 0.5, axis=0), np.percentile(seen, 99.5, axis=0)
            self.low, self.high = low - 0.1, high + 0.1


def inputs(view: dict, window: tuple[slice, slice], device: torch.device) -> dict:
    """The model's inputs for a window of a view, as tensors on `device`."""
    rows, cols = window
    height, width = view["distance"].shape
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    ys, xs = np.mgrid[rows, cols]
    distance = view["distance"][window]
    finite = np.isfinite(distance)
    return {"rgb": to(view["rgb"][window]).permute(2, 0, 1).float() / 255.0,
            "classes": to(view["classes"][window]).long(),
            "surface": to(finite.astype(np.float32)),
            "depth": to(np.clip((np.where(finite, distance, 16.0) - 2.0) / 14.0, 0.0, 1.0).astype(np.float32)),
            "points": to(points(view, window)).permute(2, 0, 1),
            "pixels": to(np.stack([(xs + 0.5) / width * 2 - 1, (ys + 0.5) / height * 2 - 1], -1).astype(np.float32)),
            "conditions": to(view["conditions"])}


def batch(items: list[dict]) -> dict:
    return {k: torch.stack([item[k] for item in items]) for k in items[0]}


class Grid(nn.Module):
    """Features on a regular grid over the box [low, high], looked up trilinearly at points (gathered from the eight
    corners, so it runs anywhere)."""

    def __init__(self, low: np.ndarray, high: np.ndarray, cell: float, channels: int) -> None:
        super().__init__()
        self.register_buffer("low", torch.tensor(low, dtype=torch.float32))
        size = np.maximum(np.ceil((high - low) / cell).astype(int) + 1, 2)
        self.size = [int(v) for v in size]
        self.register_buffer("scale", torch.tensor((size - 1) / (high - low), dtype=torch.float32))
        self.features = nn.Parameter(torch.zeros(int(np.prod(size)), channels))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # points (N, 3) -> features (N, C)
        g = ((points - self.low) * self.scale).clamp(min=0)
        sx, sy, sz = self.size
        g = torch.minimum(g, torch.tensor([sx - 1.001, sy - 1.001, sz - 1.001], device=g.device))
        base = g.floor()
        f = g - base
        i = base.long()
        out = 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    index = ((i[:, 2] + dz) * sy + (i[:, 1] + dy)) * sx + (i[:, 0] + dx)
                    w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * \
                        (f[:, 2] if dz else 1 - f[:, 2])
                    out = out + w[:, None] * self.features[index]
        return out


class FieldStyle(nn.Module):
    def __init__(self, low: np.ndarray, high: np.ndarray, fine: float = 0.02,
                 paper: tuple[int, int] = (HEIGHT // 2, WIDTH // 2)) -> None:
        super().__init__()
        self.coarse = Grid(low, high, 4 * fine, 8)
        self.fine = Grid(low, high, fine, 8)
        self.paper = nn.Parameter(torch.zeros(1, 4, *paper))
        self.edges = nn.Sequential(nn.Conv2d(CLASSES + 1, 16, 3, padding=1), nn.ReLU(),
                                   nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(), nn.Conv2d(16, 8, 3, padding=1))
        width = 3 + CLASSES + 16 + 4 + 8 + CONDITIONS
        self.mlp = nn.Sequential(nn.Conv2d(width, 64, 1), nn.ReLU(), nn.Conv2d(64, 64, 1), nn.ReLU(),
                                 nn.Conv2d(64, 64, 1), nn.ReLU(), nn.Conv2d(64, 3, 1))

    def forward(self, x: dict) -> torch.Tensor:
        rgb, classes = x["rgb"], x["classes"]
        b, _, h, w = rgb.shape
        onehot = F.one_hot(classes, CLASSES).permute(0, 3, 1, 2).float()
        surface = x["surface"][:, None]
        points = x["points"].permute(0, 2, 3, 1).reshape(-1, 3)
        field = torch.cat([self.coarse(points), self.fine(points)], -1)
        field = field.reshape(b, h, w, 16).permute(0, 3, 1, 2) * surface
        # The paper shows where no surface is; on the surfaces, the field carries the grain, anchored to them.
        paper = F.grid_sample(self.paper.expand(b, -1, -1, -1), x["pixels"], mode="bilinear",
                              align_corners=False) * (1.0 - surface)
        edges = self.edges(torch.cat([onehot, x["depth"][:, None]], 1))
        cond = x["conditions"][:, :, None, None].expand(-1, -1, h, w)
        out = self.mlp(torch.cat([rgb, onehot, field, paper, edges, cond], 1))
        return torch.sigmoid(out + torch.logit(rgb.clamp(0.02, 0.98)))


# --- runtime -------------------------------------------------------------------------------------------------------
#
# klartraum's FieldStyle packs every pixel's inputs into INPUT_CHANNELS planes in a shader (the grid and paper look-ups
# included) and runs the rest as an ONNX network: the edge convolutions and the MLP, with the input's channels picked
# by zero weights. runtime_view() makes a view from the frame's images as that shader reads them.

INPUT_CHANNELS = 41  # rgb 3, classes 7, depth 1, field 16, paper 4, conditions 7, logit of the rgb 3
FIELD_MAGIC = 0x444C464B  # "KFLD"


def runtime_view(colour: np.ndarray, classes: np.ndarray, coverage: np.ndarray, depth: np.ndarray, view: dict,
                 near: float, far: float) -> dict:
    """A view as FieldStyle reads it from the frame's images (uint8 RGB): the colour, the classes, the coverage and
    the depth over black between near and far."""
    indices = class_indices(classes)
    return {"rgb": colour, "classes": indices, "distance": decode_distance(indices, coverage[..., 0], depth, near, far),
            "conditions": conditions(view), "view": view}


def pack(model: "FieldStyle", x: dict) -> torch.Tensor:
    """The INPUT_CHANNELS planes FieldStyle's shader writes for a batch of inputs."""
    surface = x["surface"][:, None]
    b, _, h, w = x["rgb"].shape
    onehot = F.one_hot(x["classes"], CLASSES).permute(0, 3, 1, 2).float()
    flat = x["points"].permute(0, 2, 3, 1).reshape(-1, 3)
    field = torch.cat([model.coarse(flat), model.fine(flat)], -1).reshape(b, h, w, 16).permute(0, 3, 1, 2) * surface
    paper = F.grid_sample(model.paper.expand(b, -1, -1, -1), x["pixels"], mode="bilinear",
                          align_corners=False) * (1.0 - surface)
    cond = x["conditions"][:, :, None, None].expand(-1, -1, h, w)
    logit = torch.logit(x["rgb"].clamp(0.02, 0.98))
    return torch.cat([x["rgb"], onehot, x["depth"][:, None], field, paper, cond, logit], 1)


class Exported(nn.Module):
    """FieldStyle after the shader's look-ups: the packed planes in, the watercolor colour out."""

    def __init__(self, model: "FieldStyle") -> None:
        super().__init__()
        e = model.edges
        first = nn.Conv2d(INPUT_CHANNELS, 16, 3, padding=1)
        with torch.no_grad():
            first.weight.zero_()
            first.weight[:, 3:11] = e[0].weight  # the classes and the depth
            first.bias.copy_(e[0].bias)
        self.edges = nn.Sequential(first, *list(e)[1:])
        m = model.mlp[0]
        joined = nn.Conv2d(INPUT_CHANNELS + 8, 64, 1)
        # Columns of the MLP's input (rgb, classes, field, paper, edges, conditions) in the packed planes + edges.
        columns = list(range(0, 10)) + list(range(11, 31)) + list(range(41, 49)) + list(range(31, 38))
        with torch.no_grad():
            joined.weight.zero_()
            joined.weight[:, columns] = m.weight
            joined.bias.copy_(m.bias)
        self.mlp = nn.Sequential(joined, *list(model.mlp)[1:])
        self.logit = nn.Conv2d(INPUT_CHANNELS, 3, 1)
        with torch.no_grad():
            self.logit.weight.zero_()
            self.logit.bias.zero_()
            for c in range(3):
                self.logit.weight[c, 38 + c] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(torch.cat([x, self.edges(x)], 1)) + self.logit(x))


def export_runtime(model: "FieldStyle", example: dict, stem: Path) -> float:
    """Writes <stem>.bin (the grids and the paper) and <stem>.onnx (Exported at the example's size), and returns the
    largest difference between Exported on the packed example and the model."""
    from train_watercolor import export_onnx

    stem.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    example = {k: v.cpu() for k, v in example.items()}
    exported = Exported(model).eval()
    with torch.no_grad():
        packed = pack(model, example)
        difference = float((exported(packed) - model(example)).abs().max())
    export_onnx(exported, packed, Path(f"{stem}.onnx"))
    header = np.array([FIELD_MAGIC, 1, 8, *model.coarse.size, *model.fine.size, *model.paper.shape[1:]], np.uint32)
    box = np.concatenate([model.coarse.low.numpy(), model.coarse.low.numpy() + (np.array(model.coarse.size) - 1) /
                          model.coarse.scale.numpy()]).astype(np.float32)
    with open(f"{stem}.bin", "wb") as out:
        out.write(header.tobytes())
        out.write(box.tobytes())
        for grid in (model.coarse, model.fine):
            out.write(grid.features.detach().numpy().astype(np.float32).tobytes())
        out.write(model.paper.detach()[0].numpy().astype(np.float32).tobytes())
    return difference


def flicker_ratio(model: nn.Module, data: Data, device: torch.device, excess: bool = False) -> float:
    """The mean change of the output between consecutive frames of the held-out sequences at the same surface points
    (reprojected with depth and cameras, where seen in both), relative to that of the rendering.

    With `excess`, relative to the change of the best affine colour mapping of the rendering to the output (fitted
    per frame): a brighter style scales the rendering's own changes, which is no flicker; 1 means the output changes
    no more than that mapping of the rendering would."""
    out_change = in_change = 0.0
    full = (slice(0, HEIGHT), slice(0, WIDTH))
    with torch.no_grad():
        for frames in data.held_sequences:
            outputs = [model(batch([inputs(f, full, device)]))[0].cpu().numpy() for f in frames]
            for k in range(len(frames) - 1):
                earlier, later = frames[k], frames[k + 1]
                later_points, earlier_points = points(later), points(earlier)
                clip = np.concatenate([later_points.astype(np.float64), np.ones((HEIGHT, WIDTH, 1))], -1) @ (
                    matrix(earlier["view"]["camera"]["projection"]) @ matrix(earlier["view"]["camera"]["view"])).T
                ndc = clip[..., :2] / clip[..., 3:]
                px = np.clip(((ndc[..., 0] * 0.5 + 0.5) * WIDTH).astype(int), 0, WIDTH - 1)
                py = np.clip(((ndc[..., 1] * 0.5 + 0.5) * HEIGHT).astype(int), 0, HEIGHT - 1)
                seen = earlier_points[py, px]
                mask = (later["classes"] > 0) & (np.abs(ndc) < 1).all(-1) & \
                       (np.linalg.norm(seen - later_points, axis=-1) < 0.1)
                if mask.sum() < 100:
                    continue
                o1, o0 = outputs[k + 1], outputs[k][:, py, px]
                r1 = later["rgb"].transpose(2, 0, 1) / 255.0
                r0 = earlier["rgb"][py, px].transpose(2, 0, 1) / 255.0
                if excess:
                    m = affine_fit(r1, o1, mask)
                    r1, r0 = affine_apply(m, r1), affine_apply(m, r0)
                out_change += float(np.abs(o1 - o0)[:, mask].mean())
                in_change += float(np.abs(r1 - r0)[:, mask].mean())
    return out_change / max(in_change, 1e-9)


def affine_fit(rgb: np.ndarray, out: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The affine colour transform (4 x 3) that best maps the rendering (3, H, W) to `out` on `mask`."""
    a = np.concatenate([rgb[:, mask].T, np.ones((int(mask.sum()), 1))], 1)
    m, *_ = np.linalg.lstsq(a, out[:, mask].T, rcond=None)
    return m


def affine_apply(m: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    flat = np.concatenate([rgb.reshape(3, -1).T, np.ones((rgb.shape[1] * rgb.shape[2], 1))], 1) @ m
    return flat.T.reshape(rgb.shape)


def evaluate(model: nn.Module, data: Data, perceptual: nn.Module, device: torch.device, folder: Path) -> dict:
    model.eval()
    full = (slice(0, HEIGHT), slice(0, WIDTH))
    l1 = l1_rendering = per = per_rendering = 0.0
    rows = []
    with torch.no_grad():
        for p in data.held:
            x = batch([inputs(p, full, device)])
            y = torch.from_numpy(p["style"]).to(device).permute(2, 0, 1)[None].float() / 255.0
            out = model(x)
            l1 += float(F.l1_loss(out, y)) / len(data.held)
            l1_rendering += float(F.l1_loss(x["rgb"], y)) / len(data.held)
            per += float(perceptual(out, y)) / len(data.held)
            per_rendering += float(perceptual(x["rgb"], y)) / len(data.held)
            rows.append(torch.cat([x["rgb"][0], out[0], y[0]], 2).cpu())
    metrics = {"held_out": [p["name"] for p in data.held], "network": {"l1": l1, "perceptual": per},
               "rendering": {"l1": l1_rendering, "perceptual": per_rendering},
               "flicker ratio": flicker_ratio(model, data, device),
               "excess flicker": flicker_ratio(model, data, device, excess=True)}
    # Where the change comes from: the same model without the fine grid, and without both grids.
    saved = [model.fine.features.data.clone(), model.coarse.features.data.clone()]
    model.fine.features.data.zero_()
    metrics["flicker ratio without fine grid"] = flicker_ratio(model, data, device)
    model.coarse.features.data.zero_()
    metrics["flicker ratio without grids"] = flicker_ratio(model, data, device)
    model.fine.features.data.copy_(saved[0])
    model.coarse.features.data.copy_(saved[1])
    folder.mkdir(parents=True, exist_ok=True)
    for k in range(0, len(rows), 5):
        grid = (torch.cat(rows[k:k + 5], 1).permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(grid).save(folder / f"held_out_{k // 5}.jpg", quality=90)
    model.train()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gbuffers", type=Path, required=True)
    parser.add_argument("--styles", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--crop", type=int, default=320)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--perceptual", type=float, default=0.05)
    parser.add_argument("--fine", type=float, default=0.02, help="cell size of the fine grid")
    args = parser.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    data = Data(args.gbuffers, args.styles)
    args.out.mkdir(parents=True, exist_ok=True)
    log = open(args.out / "log.txt", "a")
    say = lambda text: (print(text, flush=True), log.write(text + "\n"), log.flush())
    say(f"{len(data.train)} training pairs, {len(data.held)} held out, {len(data.sequences)} sequences; field box "
        f"{np.round(data.low, 2).tolist()} to {np.round(data.high, 2).tolist()}")
    model = FieldStyle(data.low, data.high, args.fine).to(device)
    perceptual = Perceptual().to(device)
    grids = [model.coarse.features, model.fine.features, model.paper]
    others = [p for n, p in model.named_parameters() if not any(p is g for g in grids)]
    optimizer = torch.optim.Adam([{"params": grids, "lr": 1e-2}, {"params": others, "lr": 1e-3}])
    base = [g["lr"] for g in optimizer.param_groups]
    start = time.time()
    for step in range(args.steps):
        factor = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for group, lr in zip(optimizer.param_groups, base):
            group["lr"] = lr * factor
        items, targets = [], []
        for _ in range(args.batch):
            p = data.train[rng.integers(len(data.train))]
            top, left = rng.integers(HEIGHT - args.crop + 1), rng.integers(WIDTH - args.crop + 1)
            window = (slice(top, top + args.crop), slice(left, left + args.crop))
            items.append(inputs(p, window, device))
            targets.append(torch.from_numpy(np.ascontiguousarray(p["style"][window])).to(device).permute(2, 0, 1))
        x, y = batch(items), torch.stack(targets).float() / 255.0
        out = model(x)
        l1 = F.l1_loss(out, y)
        loss = l1 + args.perceptual * perceptual(out, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 200 == 0:
            say(f"step {step}: l1 {float(l1.detach()):.4f}, loss {float(loss.detach()):.4f}, "
                f"{time.time() - start:.0f} s")
    metrics = evaluate(model, data, perceptual, device, args.out)
    metrics["steps"] = args.steps
    say(json.dumps(metrics))
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    torch.save({"model": model.state_dict(), "low": data.low.tolist(), "high": data.high.tolist(),
                "fine": args.fine}, args.out / "field.pt")
    p = data.held[0]
    difference = export_runtime(model, batch([inputs(p, (slice(0, HEIGHT), slice(0, WIDTH)), torch.device("cpu"))]),
                                args.out / "field")
    say(f"exported to {args.out}/field.onnx and field.bin; largest difference of the exported network: "
        f"{difference:.2e}")


if __name__ == "__main__":
    main()
