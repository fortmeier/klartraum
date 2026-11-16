# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Reprojection between frames of the bonsai fly-in (tree_bonsai_example --fly-in --gbuffers).

A backward warp: every pixel of the target frame is lifted to the world with the target's depth G-buffer and camera,
projected into the source frame, and the source image is sampled there. A pixel is valid where the source sees the
same point (its depth there agrees with the projected distance) and is inside the source's view; background pixels
(class black) are not valid. Images may have another size than the G-buffers when they are a scaled and
vertically cropped version of them (e.g. 1280x720 -> 832x468 -> rows 2..466 = 832x464, the Qwen-Image frames).

    uv run --project scripts/qwen_image python scripts/qwen_image/flyin_warp.py check build/TestingOutput/qwen_flyin
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


def read_ppm(path: Path) -> np.ndarray:
    data = path.read_bytes()
    # Four header tokens (P6, width, height, maximum), then exactly one whitespace byte before the pixels.
    tokens, position = [], 0
    while len(tokens) < 4:
        while data[position:position + 1].isspace():
            position += 1
        end = position
        while not data[end:end + 1].isspace():
            end += 1
        tokens.append(data[position:end])
        position = end
    if tokens[0] != b"P6":
        raise ValueError(f"{path} is not a binary PPM")
    width, height = int(tokens[1]), int(tokens[2])
    pixels = np.frombuffer(data, dtype=np.uint8, count=width * height * 3, offset=position + 1)
    return pixels.reshape(height, width, 3)


def write_ppm(path: Path, image: np.ndarray) -> None:
    image = np.clip(np.rint(image), 0, 255).astype(np.uint8)
    path.write_bytes(f"P6\n{image.shape[1]} {image.shape[0]}\n255\n".encode() + image.tobytes())


class Flight:
    """Cameras and G-buffers of a rendered fly-in."""

    def __init__(self, directory: Path, renderer: str = "classical"):
        self.directory = directory
        self.renderer = renderer
        with open(directory / f"cameras_{renderer}.csv") as file:
            self.cameras = {int(row["frame"]): row for row in csv.DictReader(file)}
        first = read_ppm(self.frame_path(min(self.cameras), "class"))
        self.height, self.width = first.shape[:2]

    def frame_path(self, frame: int, kind: str = "") -> Path:
        middle = f"{kind}_" if kind else ""
        return self.directory / "frames" / f"{self.renderer}_{middle}{frame:04d}.ppm"

    def distance(self, frame: int) -> np.ndarray:
        """Distance from the camera per G-buffer pixel; NaN on the background."""
        depth = read_ppm(self.frame_path(frame, "depth")).astype(np.float64)
        covered = read_ppm(self.frame_path(frame, "class")).any(axis=2)
        d = depth.mean(axis=2) / 255.0 - 1.0 / 255.0
        far = float(self.cameras[frame]["depth_far"])
        return np.where(covered, np.clip(d, 0.0, 1.0) * far, np.nan)

    def basis(self, frame: int) -> tuple[np.ndarray, np.ndarray, float]:
        """Camera position, rotation rows (right, up, forward) and tan(fov/2), as glm::lookAt and perspective."""
        row = self.cameras[frame]
        eye = np.array([float(row[f"eye_{a}"]) for a in "xyz"])
        target = np.array([float(row[f"look_{a}"]) for a in "xyz"])
        forward = target - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        return eye, np.stack([right, up, forward]), math.tan(math.radians(float(row["fov_y_degrees"])) / 2)


class ImageGrid:
    """Maps pixels of an image (scaled to `scaled_height` and cropped from `top`) to normalized device coordinates."""

    def __init__(self, width: int, height: int, gbuffer_width: int, gbuffer_height: int, top: int = 0,
                 scaled_height: int | None = None):
        self.width, self.height, self.top = width, height, top
        self.scaled_height = scaled_height or height
        self.aspect = gbuffer_width / gbuffer_height

    def ndc(self) -> tuple[np.ndarray, np.ndarray]:
        y, x = np.mgrid[0:self.height, 0:self.width]
        u = (x + 0.5) / self.width * 2.0 - 1.0
        v = 1.0 - (y + self.top + 0.5) / self.scaled_height * 2.0  # up positive
        return u, v

    def pixel(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = (u + 1.0) / 2.0 * self.width - 0.5
        y = (1.0 - v) / 2.0 * self.scaled_height - 0.5 - self.top
        return x, y


def sample_gbuffer(values: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Nearest sample of a G-buffer image at normalized device coordinates (NaN outside)."""
    h, w = values.shape[:2]
    x = np.floor((u + 1.0) / 2.0 * w).astype(np.int64)
    y = np.floor((1.0 - v) / 2.0 * h).astype(np.int64)
    inside = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    result = np.full(u.shape, np.nan)
    result[inside] = values[y[inside], x[inside]]
    return result


def bilinear(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    top = image[y0, x0] * (1 - fx) + image[y0, x1] * fx
    bottom = image[y1, x0] * (1 - fx) + image[y1, x1] * fx
    return top * (1 - fy) + bottom * fy


def warp(flight: Flight, source: int, target: int, source_image: np.ndarray, grid: ImageGrid,
         tolerance: float = 0.03) -> tuple[np.ndarray, np.ndarray]:
    """The source image seen from the target frame, and the mask of valid pixels."""
    u, v = grid.ndc()
    eye_t, rotation_t, tan_t = flight.basis(target)
    distance_t = sample_gbuffer(flight.distance(target), u, v)
    rays = (u[..., None] * tan_t * grid.aspect * rotation_t[0] + v[..., None] * tan_t * rotation_t[1] +
            rotation_t[2])
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    points = eye_t + rays * distance_t[..., None]

    eye_s, rotation_s, tan_s = flight.basis(source)
    local = (points - eye_s) @ rotation_s.T  # right, up, forward
    in_front = local[..., 2] > 1e-9
    us = local[..., 0] / (local[..., 2] * tan_s * grid.aspect)
    vs = local[..., 1] / (local[..., 2] * tan_s)
    distance_s = np.linalg.norm(points - eye_s, axis=-1)
    seen = sample_gbuffer(flight.distance(source), us, vs)
    valid = (np.isfinite(distance_t) & in_front & (np.abs(us) <= 1) & (np.abs(vs) <= 1) &
             np.isfinite(seen) & (np.abs(seen - distance_s) <= tolerance * distance_s + 2e-3 * seen))
    x, y = grid.pixel(us, vs)
    warped = bilinear(source_image.astype(np.float64), np.nan_to_num(x), np.nan_to_num(y))
    return warped, valid


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * math.log10(255.0**2 / max(mse, 1e-12))


def check(args: argparse.Namespace) -> None:
    """Warps ground-truth frames t - k into t and compares with frame t where the warp is valid."""
    flight = Flight(args.directory)
    grid = ImageGrid(flight.width, flight.height, flight.width, flight.height)
    for target in args.frames:
        for k in args.offsets:
            source = target - k
            warped, valid = warp(flight, source, target, read_ppm(flight.frame_path(source)), grid)
            truth = read_ppm(flight.frame_path(target))
            identity = psnr(read_ppm(flight.frame_path(source))[valid], truth[valid])
            print(f"frame {target} from {source}: {valid.mean() * 100:5.1f} % valid, PSNR warped "
                  f"{psnr(warped[valid], truth[valid]):5.2f} dB (unwarped {identity:5.2f} dB)")
            if args.out:
                args.out.mkdir(parents=True, exist_ok=True)
                write_ppm(args.out / f"warp_{source:04d}_to_{target:04d}.ppm", np.where(valid[..., None], warped, 0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("check", help="warp ground-truth frames and compare")
    c.add_argument("directory", type=Path)
    c.add_argument("--frames", type=int, nargs="+", default=[60, 150, 300, 500, 700])
    c.add_argument("--offsets", type=int, nargs="+", default=[1, 4, 8])
    c.add_argument("--out", type=Path)
    args = parser.parse_args()
    check(args)


if __name__ == "__main__":
    main()
