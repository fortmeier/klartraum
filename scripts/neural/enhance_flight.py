# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Runs a trained photo enhancer (train_photo_enhancer.py's checkpoint) on every frame of a fly-in rendered with
G-buffers and encodes the result as videos: <out>/enhanced.mp4 and <out>/side_by_side.mp4 (rendering | enhanced).
The inputs are prepared as for training (prepare_photo_pairs.py): the frame at 832 x 464, its inverse depth and
its classes. Frames go to <out>/frames/.

    uv run --project scripts/qwen_image python scripts/neural/enhance_flight.py build/TestingOutput/qwen_flyin \\
        build/TestingOutput/gan_runs/v1/checkpoint.pt build/TestingOutput/gan_runs/v1/flight
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qwen_image"))
import flyin_warp  # noqa: E402
from prepare_photo_pairs import classes_of, to_sample_size  # noqa: E402
from add_surface_coordinates import surface_coordinates  # noqa: E402
from train_photo_enhancer import Generator, TexturedModel, network_input  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("flight", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--every", type=int, default=1, help="use every n-th frame")
    parser.add_argument("--uv-flight", type=Path,
                        help="the flight re-rendered with the coordinate G-buffers (for checkpoints with a texture)")
    args = parser.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    state = torch.load(args.checkpoint, map_location=device)["ema"]
    textured = any(key.startswith("texture.") for key in state)
    if any(key.startswith("generator.") for key in state):
        model = TexturedModel(textured).to(device).eval()
    else:
        model = Generator().to(device).eval()  # checkpoints from before the texture
    model.load_state_dict(state)
    if textured and args.uv_flight is None:
        raise SystemExit("this checkpoint has a surface texture: give the coordinate flight with --uv-flight")
    uv_cameras = {}
    if textured:
        import csv

        with open(args.uv_flight / "cameras_classical.csv") as file:
            uv_cameras = {int(r["frame"]): float(r["depth_far"]) for r in csv.DictReader(file) if r.get("depth_far")}
    flight = flyin_warp.Flight(args.flight)
    frames_dir = args.out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(flight.cameras)[::args.every]
    for n, frame in enumerate(frames):
        render = to_sample_size(flyin_warp.read_ppm(flight.frame_path(frame)), Image.BICUBIC)
        depth = to_sample_size((flyin_warp.depth_image(flight, frame) * 255).round().astype(np.uint8), Image.BILINEAR)
        classes = to_sample_size(classes_of(flyin_warp.read_ppm(flight.frame_path(frame, "class"))), Image.NEAREST)
        raw = np.concatenate([render, depth[..., None], classes[..., None]], axis=2)
        raw = torch.from_numpy(raw).permute(2, 0, 1)[None].contiguous()
        with torch.no_grad():
            if textured:
                uvid, precise = surface_coordinates(args.uv_flight, frame, uv_cameras[frame])
                coords = torch.from_numpy(np.concatenate([uvid, precise[None]]))[None]
                x = network_input(raw, coords).to(device)
                y = model(x, coords[:, :3].to(device), raw[:, 4].long().to(device))
            else:
                y = model(network_input(raw).to(device))
            y = y[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        enhanced = (y * 255).round().astype(np.uint8)
        Image.fromarray(enhanced).save(frames_dir / f"enhanced_{n:04d}.png")
        Image.fromarray(np.concatenate([render, enhanced], axis=1)).save(frames_dir / f"side_{n:04d}.png")
        if n % 60 == 0:
            print(f"  frame {frame} ({n + 1}/{len(frames)})", flush=True)
    for name, pattern in (("enhanced", "enhanced_%04d.png"), ("side_by_side", "side_%04d.png")):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(args.fps // args.every), "-i",
                        str(frames_dir / pattern), "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                        str(args.out / f"{name}.mp4")], check=True)
    print(f"videos in {args.out}")


if __name__ == "__main__":
    main()
