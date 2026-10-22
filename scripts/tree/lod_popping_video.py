# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow"]
# ///

"""Labels the frames of tree_lod_popping_example and encodes them into a video.

Each frame gets the run's name, the distance, what draws the tree and the residual over both panels, and a strip
below with both runs' residuals over the whole flight and a cursor at the current frame. The frames are piped into
ffmpeg, nothing is written besides the video.

    uv run scripts/tree/lod_popping_video.py <out-dir> [--fps 30]
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

STRIP = 200
STATIC = (90, 170, 255)
DYNAMIC = (255, 150, 60)


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    with open(args.out_dir / "popping.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    frames = sorted((args.out_dir / "frames").glob("frame_*.ppm"))
    if len(frames) != len(rows):
        raise SystemExit(f"{len(frames)} frames but {len(rows)} rows in popping.csv")
    width, height = Image.open(frames[0]).size
    panel = width // 2

    static = [float(r["residual_static"]) for r in rows]
    dynamic = [float(r["residual_dynamic"]) for r in rows]
    top = max(max(static), max(dynamic), 1e-6)
    big, small = font(28), font(20)

    # The strip's curves, drawn once.
    strip = Image.new("RGB", (width, STRIP), (18, 18, 18))
    draw = ImageDraw.Draw(strip)
    left, right, low, high = 70, width - 20, STRIP - 30, 15

    def x_of(k: int) -> float:
        return left + (right - left) * k / max(len(rows) - 1, 1)

    def y_of(v: float) -> float:
        return low - (low - high) * v / top

    for k in range(1, len(rows)):
        if rows[k]["drawn_by"] != rows[k - 1]["drawn_by"]:
            draw.line([(x_of(k), high), (x_of(k), low)], fill=(60, 60, 60))
    for values, colour in ((static, STATIC), (dynamic, DYNAMIC)):
        draw.line([(x_of(k), y_of(v)) for k, v in enumerate(values)], fill=colour, width=2)
    draw.line([(left, low), (right, low)], fill=(120, 120, 120))
    draw.text((8, high - 4), f"{top:.3f}", font=small, fill=(160, 160, 160))
    draw.text((8, low - 20), "0", font=small, fill=(160, 160, 160))
    draw.text((left + 10, low + 4), "residual per frame (grey lines: what draws the tree changes)", font=small,
              fill=(160, 160, 160))

    encoder = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
         f"{width}x{height + STRIP}", "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt",
         "yuv420p", str(args.out_dir / "tree_lod_popping.mp4")],
        stdin=subprocess.PIPE)
    for k, (path, row) in enumerate(zip(frames, rows)):
        canvas = Image.new("RGB", (width, height + STRIP))
        canvas.paste(Image.open(path), (0, 0))
        canvas.paste(strip, (0, height))
        draw = ImageDraw.Draw(canvas)
        for side, (name, values, colour) in enumerate(
                (("static (no wind)", static, STATIC), ("dynamic (wind)", dynamic, DYNAMIC))):
            x = side * panel + 16
            draw.text((x, 12), name, font=big, fill=colour)
            draw.text((x, 48), f"{float(row['distance_m']):6.1f} m   {row['drawn_by']}", font=small,
                      fill=(230, 230, 230))
            draw.text((x, 74), f"residual {values[k]:.4f}", font=small, fill=(230, 230, 230))
        draw.line([(x_of(k), height + high), (x_of(k), height + low)], fill=(255, 255, 255), width=2)
        encoder.stdin.write(canvas.tobytes())
    encoder.stdin.close()
    encoder.wait()
    print(f"video in {args.out_dir / 'tree_lod_popping.mp4'}")


if __name__ == "__main__":
    main()
