# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Adds the surface-coordinate G-buffers of re-rendered flights (tree_bonsai_example --gbuffers, whose u, v and
16-bit depth frames the original flights lack) to the photo enhancer's samples: for every sample <flight>_<frame>
of every split, <split>/coords/<sample>.npz with, at the sample size (--size, default 832 x 464, prepare_photo_pairs.to_sample_size()
with nearest sampling), `uvid` [3, H, W] float32 (u, v and the 16-bit object id, each in [0, 1]) and `depth` [H, W]
float32 (the inverse depth from the 16-bit depth, stretched per frame as flyin_warp.depth_image()).

    uv run --project scripts/qwen_image python scripts/neural/add_surface_coordinates.py build/TestingOutput/gan_data \\
        build/TestingOutput/gan_data/flyins_uv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_photo_pairs  # noqa: E402
from prepare_photo_pairs import to_sample_size  # noqa: E402


def frame_image(flight: Path, kind: str, frame: int) -> np.ndarray:
    for suffix in (".png", ".ppm"):
        path = flight / "frames" / f"classical_{kind}_{frame:04d}{suffix}"
        if path.exists():
            return np.asarray(Image.open(path).convert("RGB")).astype(np.uint32)
    raise FileNotFoundError(f"{flight.name}: no {kind} frame {frame}")


def value16(image: np.ndarray) -> np.ndarray:
    return (image[..., 0] * 256 + image[..., 1]).astype(np.float32) / 65535.0


def surface_coordinates(flight: Path, frame: int, depth_far: float) -> tuple[np.ndarray, np.ndarray]:
    u_image, v_image = frame_image(flight, "u", frame), frame_image(flight, "v", frame)
    uvid = np.stack([value16(u_image), value16(v_image),
                     (u_image[..., 2] * 256 + v_image[..., 2]).astype(np.float32) / 65535.0])
    classes = frame_image(flight, "class", frame).any(axis=2)
    distance = value16(frame_image(flight, "depth16", frame)) * depth_far
    inverse = 1.0 / np.maximum(distance, 1e-9)
    depth = np.zeros(distance.shape, dtype=np.float32)
    if classes.any():
        a, b = np.percentile(inverse[classes], [2.0, prepare_photo_pairs.DEPTH_HIGH])
        depth = np.where(classes, np.clip((inverse - a) / max(b - a, 1e-12), 0.0, 1.0), 0.0).astype(np.float32)
    # Nearest sampling: coordinates and ids must not be blended between neighbouring surfaces.
    uvid = np.stack([to_sample_size(channel, Image.NEAREST) for channel in uvid])
    depth = to_sample_size(depth, Image.BILINEAR)
    return uvid, depth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path)
    parser.add_argument("flights", type=Path, help="directory of the re-rendered flights")
    parser.add_argument("--size", type=int, nargs=2, default=(prepare_photo_pairs.HEIGHT, prepare_photo_pairs.WIDTH),
                        metavar=("HEIGHT", "WIDTH"), help="the sample size, as for prepare_photo_pairs.py")
    parser.add_argument("--depth-high", type=float, default=prepare_photo_pairs.DEPTH_HIGH,
                        help="as for prepare_photo_pairs.py")
    args = parser.parse_args()
    prepare_photo_pairs.set_sample_size(*args.size)
    prepare_photo_pairs.set_depth_high(args.depth_high)
    cameras = {}
    for flight in args.flights.iterdir():
        csv_path = flight / "cameras_classical.csv"
        if csv_path.exists():
            with open(csv_path) as file:
                # A flight still being rendered may end in an incomplete line.
                cameras[flight.name] = {int(r["frame"]): float(r["depth_far"]) for r in csv.DictReader(file)
                                        if r.get("depth_far")}
    for split in ("train", "validation"):
        out = args.data / split / "coords"
        out.mkdir(parents=True, exist_ok=True)
        done = missing = 0
        for photo in sorted((args.data / split / "photo").glob("*.ppm")):
            flight_name, frame = photo.stem.rsplit("_", 1)
            if flight_name not in cameras or int(frame) not in cameras[flight_name]:
                missing += 1
                continue
            uvid, depth = surface_coordinates(args.flights / flight_name, int(frame), cameras[flight_name][int(frame)])
            np.savez_compressed(out / f"{photo.stem}.npz", uvid=uvid, depth=depth)
            done += 1
        print(f"{split}: {done} samples with coordinates, {missing} without a re-rendered frame")


if __name__ == "__main__":
    main()
