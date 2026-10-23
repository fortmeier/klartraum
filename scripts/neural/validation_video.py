# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "pillow", "onnx"]
# ///

"""Shows a trained tree enhancer on the held-out views of its dataset (every tenth), one view per two seconds:
the Gaussian input, the enhanced image and the classical target (the reference on the far trees, the input
elsewhere), each with its block error and detail error (lod_metrics' detailError()) against the target on the
far trees.

    uv run scripts/neural/validation_video.py <dataset-dir> <enhancer.pt> <video.mp4>
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
import json  # noqa: E402

from train_tree_enhancer import ARCHITECTURES, block_error, detail_error, load_view, to_tensors, with_noise  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("weights", type=Path)
    parser.add_argument("video", type=Path)
    args = parser.parse_args()
    config = args.weights.parent / "config.json"
    arch = json.loads(config.read_text())["arch"] if config.exists() else "v1"
    model = ARCHITECTURES[arch]()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()
    names = [p.stem[3:] for p in sorted(args.dataset.glob("hf_*.ppm")) if int(p.stem[3:]) % 10 == 9]
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:
        font = ImageFont.load_default()
    w, h = 640, 360
    encoder = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{3 * w}x{h}", "-r",
         "0.5", "-i", "-", "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30", str(args.video)],
        stdin=subprocess.PIPE)
    totals = np.zeros(4)
    for name in names:
        x, y, m = to_tensors(load_view(args.dataset, name))
        with torch.no_grad():
            out = model(with_noise(model, x.unsqueeze(0), torch.Generator().manual_seed(0))).clamp(0.0, 1.0)[0]
        before = block_error(x[:3].unsqueeze(0), y.unsqueeze(0), m.unsqueeze(0))
        after = block_error(out.unsqueeze(0), y.unsqueeze(0), m.unsqueeze(0))
        detail_before = detail_error(x[:3].unsqueeze(0), y.unsqueeze(0), m.unsqueeze(0))
        detail_after = detail_error(out.unsqueeze(0), y.unsqueeze(0), m.unsqueeze(0))
        totals += (before, after, detail_before, detail_after)
        canvas = Image.new("RGB", (3 * w, h))
        draw = ImageDraw.Draw(canvas)
        for k, (title, image) in enumerate(((f"Gaussian input  {before:.4f}, detail {detail_before:.2f}", x[:3]),
                                            (f"enhanced  {after:.4f}, detail {detail_after:.2f}", out),
                                            ("classical target", y))):
            pixels = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
            canvas.paste(Image.fromarray(pixels).resize((w, h)), (k * w, 0))
            draw.text((k * w + 10, 8), title, font=font, fill=(255, 255, 255))
        draw.text((10, h - 32), f"view {name}", font=font, fill=(255, 255, 255))
        encoder.stdin.write(canvas.tobytes())
    encoder.stdin.close()
    encoder.wait()
    n = len(names)
    print(f"{n} views: block error input {totals[0] / n:.4f}, enhanced {totals[1] / n:.4f}; detail error input "
          f"{totals[2] / n:.3f}, enhanced {totals[3] / n:.3f}; video in {args.video}")


if __name__ == "__main__":
    main()
