# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow"]
# ///

"""Labels the frames of tree_lod_metrics_example and encodes them into a video.

The frames hold four panels: with levels of detail above, without below; static left, dynamic right. Each panel gets
its run's name, the distance, what draws the tree and its popping residual (and fidelity error, with levels of
detail). Below the panels two strips plot the whole flight with a cursor at the current frame: the popping residual
of the four runs and the fidelity error of the two runs with levels of detail. The frames are piped into ffmpeg,
nothing is written besides the video.

    uv run scripts/tree/lod_metrics_video.py <out-dir> [--fps 30]
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PLOT = 150
STATIC = (90, 170, 255)
DYNAMIC = (255, 150, 60)
STATIC_FREE = (40, 80, 120)
DYNAMIC_FREE = (130, 75, 30)


def font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def plot(rows: list[dict], width: int, curves: list[tuple[list[float], tuple[int, int, int]]], title: str,
         small: ImageFont.ImageFont) -> tuple[Image.Image, callable]:
    """A strip with `curves` over the frames, grey lines where what draws the tree changes; returns it and x(k)."""
    strip = Image.new("RGB", (width, PLOT), (18, 18, 18))
    draw = ImageDraw.Draw(strip)
    left, right, low, high = 80, width - 20, PLOT - 28, 12
    top = max(max(values) for values, _ in curves) or 1e-6

    def x_of(k: int) -> float:
        return left + (right - left) * k / max(len(rows) - 1, 1)

    for k in range(1, len(rows)):
        if rows[k]["drawn_by"] != rows[k - 1]["drawn_by"]:
            draw.line([(x_of(k), high), (x_of(k), low)], fill=(60, 60, 60))
    for values, colour in curves:
        draw.line([(x_of(k), low - (low - high) * v / top) for k, v in enumerate(values)], fill=colour, width=2)
    draw.line([(left, low), (right, low)], fill=(120, 120, 120))
    draw.text((8, high - 4), f"{top:.3f}", font=small, fill=(160, 160, 160))
    draw.text((8, low - 18), "0", font=small, fill=(160, 160, 160))
    draw.text((left + 10, low + 4), title, font=small, fill=(160, 160, 160))
    return strip, x_of


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    with open(args.out_dir / "metrics.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    frames = sorted((args.out_dir / "frames").glob("frame_*.ppm"))
    if len(frames) != len(rows):
        raise SystemExit(f"{len(frames)} frames but {len(rows)} rows in metrics.csv")
    width, height = Image.open(frames[0]).size
    panel_w, panel_h = width // 2, height // 2

    def column(name: str) -> list[float]:
        return [float(r[name]) for r in rows]

    popping = [column("residual_static"), column("residual_dynamic"), column("residual_free_static"),
               column("residual_free_dynamic")]
    fidelity = [column("fidelity_static"), column("fidelity_dynamic")]
    big, small = font(26), font(19)

    popping_strip, x_of = plot(rows, width, [(popping[2], STATIC_FREE), (popping[3], DYNAMIC_FREE),
                                              (popping[0], STATIC), (popping[1], DYNAMIC)],
                               "popping residual (dark: without levels of detail; grey lines: what draws the tree "
                               "changes)", small)
    fidelity_strip, _ = plot(rows, width, [(fidelity[0], STATIC), (fidelity[1], DYNAMIC)],
                             "fidelity error against the tree without levels of detail", small)

    encoder = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
         f"{width}x{height + 2 * PLOT}", "-r", str(args.fps), "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt",
         "yuv420p", str(args.out_dir / "tree_lod_metrics.mp4")],
        stdin=subprocess.PIPE)
    panels = (("with LOD, static", STATIC, 0), ("with LOD, wind", DYNAMIC, 1),
              ("without LOD, static", STATIC, 2), ("without LOD, wind", DYNAMIC, 3))
    for k, (path, row) in enumerate(zip(frames, rows)):
        canvas = Image.new("RGB", (width, height + 2 * PLOT))
        canvas.paste(Image.open(path), (0, 0))
        canvas.paste(popping_strip, (0, height))
        canvas.paste(fidelity_strip, (0, height + PLOT))
        draw = ImageDraw.Draw(canvas)
        for name, colour, run in panels:
            x, y = (run % 2) * panel_w + 16, (run // 2) * panel_h + 10
            drawn = row["drawn_by"] if run < 2 else "generated"
            draw.text((x, y), name, font=big, fill=colour)
            draw.text((x, y + 34), f"{float(row['distance_m']):6.1f} m   {drawn}", font=small, fill=(230, 230, 230))
            line = f"popping {popping[run][k]:.4f}"
            if run < 2:
                line += f"   fidelity error {fidelity[run][k]:.4f}"
            draw.text((x, y + 58), line, font=small, fill=(230, 230, 230))
        draw.line([(0, panel_h), (width, panel_h)], fill=(70, 70, 70), width=2)
        draw.line([(panel_w, 0), (panel_w, height)], fill=(70, 70, 70), width=2)
        for strip_top in (height, height + PLOT):
            draw.line([(x_of(k), strip_top + 12), (x_of(k), strip_top + PLOT - 28)], fill=(255, 255, 255), width=2)
        encoder.stdin.write(canvas.tobytes())
    encoder.stdin.close()
    encoder.wait()
    print(f"video in {args.out_dir / 'tree_lod_metrics.mp4'}")


if __name__ == "__main__":
    main()
