# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

# /// script
# requires-python = ">=3.10"
# dependencies = ["pillow"]
# ///

"""Shows the view pairs of tree_neural_dataset_example as a video, one view per second: the Gaussian image, its
feature images (coverage/tree/level of detail and size/depth), the classical reference and the far trees' mask.

    uv run scripts/neural/dataset_video.py <dataset-dir> <video.mp4> [--views 40]
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("--views", type=int, default=40)
    args = parser.parse_args()
    names = sorted(p.stem[3:] for p in args.dataset.glob("hf_*.ppm"))[: args.views]
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:
        font = ImageFont.load_default()
    w, h = 640, 360
    width, height = 3 * w, 2 * h
    encoder = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", "1", "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30", str(args.video)],
        stdin=subprocess.PIPE)
    for name in names:
        tiles = [("Gaussian (low fidelity)", "lr", "ppm"), ("features A: coverage, tree, LOD", "fa", "ppm"),
                 ("classical reference (high fidelity)", "hf", "ppm"), ("features B: size, depth", "fb", "ppm"),
                 ("far trees' mask", "mask", "pgm")]
        canvas = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(canvas)
        for index, (title, prefix, extension) in enumerate(tiles):
            image = Image.open(args.dataset / f"{prefix}_{name}.{extension}").convert("RGB").resize((w, h))
            x, y = (index % 3) * w, (index // 3) * h
            canvas.paste(image, (x, y))
            draw.text((x + 10, y + 8), title, font=font, fill=(255, 255, 255))
        draw.text((2 * w + 10, h + 8), f"view {name}", font=font, fill=(255, 255, 255))
        encoder.stdin.write(canvas.tobytes())
    encoder.stdin.close()
    encoder.wait()
    print(f"{len(names)} views in {args.video}")


if __name__ == "__main__":
    main()
