# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""A fly-in made of Qwen-Image photographs of keyframes (every `--stride`-th frame, e.g. the photo enhancer's
training photographs of a flight): every frame in between is its two neighbouring keyframes' photographs reprojected
into it with the flight's depth G-buffers and cameras (flyin_warp.warp()) and blended by time; where only one
keyframe sees a pixel, that one is used, where neither does, the nearer keyframe's photograph unwarped.

    uv run --project scripts/qwen_image python scripts/qwen_image/keyframe_flight.py \\
        build/TestingOutput/gan_data/flyins/seed11_az0.6 build/TestingOutput/gan_data/train/photo \\
        build/TestingOutput/qwen_keyframe_flight --first 24 --last 440
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

import flyin_warp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("flight", type=Path, help="fly-in with G-buffers of every frame")
    parser.add_argument("photos", type=Path, help="directory of <flight name>_<frame>.ppm photographs")
    parser.add_argument("out", type=Path)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--last", type=int, required=True, help="last keyframe")
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    flight = flyin_warp.Flight(args.flight)
    name = args.flight.name
    photo = {k: flyin_warp.read_ppm(args.photos / f"{name}_{k:04d}.ppm").astype(np.float64)
             for k in range(args.first, args.last + 1, args.stride)}
    height, width = next(iter(photo.values())).shape[:2]
    # The photographs are the frames scaled to `width` and cropped symmetrically in height.
    scaled_height = round(flight.height * width / flight.width)
    grid = flyin_warp.ImageGrid(width, height, flight.width, flight.height, top=(scaled_height - height) // 2,
                                scaled_height=scaled_height)
    frames = args.out / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    for n, t in enumerate(range(args.first, args.last + 1)):
        a = args.first + ((t - args.first) // args.stride) * args.stride
        if t == a:
            image = photo[a]
        else:
            b = a + args.stride
            warped_a, valid_a = flyin_warp.warp(flight, a, t, photo[a], grid)
            warped_b, valid_b = flyin_warp.warp(flight, b, t, photo[b], grid)
            wb = (t - a) / args.stride
            weight_a = valid_a * (1 - wb)
            weight_b = valid_b * wb
            total = weight_a + weight_b
            blended = (warped_a * weight_a[..., None] + warped_b * weight_b[..., None]) / np.maximum(total, 1e-9)[..., None]
            nearest = photo[a] if wb < 0.5 else photo[b]
            image = np.where((total > 0)[..., None], blended, nearest)
        flyin_warp.write_ppm(frames / f"frame_{n:04d}.ppm", image)
        if n % 50 == 0:
            print(f"  frame {t}", flush=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(args.fps), "-i",
                    str(frames / "frame_%04d.ppm"), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                    str(args.out / "qwen_flight.mp4")], check=True)
    print(f"video in {args.out / 'qwen_flight.mp4'}")


if __name__ == "__main__":
    main()
