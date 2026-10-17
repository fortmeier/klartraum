# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Composes chapter 8 of the tech demo: one screen per compute graph, with the chapter it renders playing as a
picture in picture in the upper right corner.

The screens are the pictures of draw_graphs.py; the chapter frames come from a render of techdemo000_example,
whose log says where each chapter starts. Chapter 8's frames continue the numbering of the render's frames, so
the whole video encodes from one sequence.

    uv run --with pillow scripts/techdemo000/compose_chapter8.py --run build/TestingOutput/techdemo000_raster_full
        [--graphs build/TestingOutput/techdemo000_graphs] [--seconds 7] [--fps 30]
"""

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw

WIDTH, HEIGHT = 1920, 1080
PIP_SIZE = (480, 270)
PIP_POS = (WIDTH - 70 - PIP_SIZE[0], 40)
FADE_SECONDS = 0.5

# The screens, in order: graph picture and the chapter (1-based) it shows.
SCREENS = [
    ("gaussian_splatting.png", 2),
    ("dynamic_lighting.png", 3),
    ("neural_animation.png", 4),
    ("sd15_chain.png", 6),
]


def chapter_starts(log: Path):
    """Frame where each chapter starts, from the render log ("chapter N (name) from frame F")."""
    starts = {}
    for line in log.read_text().splitlines():
        m = re.match(r"chapter (\d+) \(\w+\) from frame (\d+)", line)
        if m:
            starts[int(m.group(1))] = int(m.group(2))
    return starts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="render directory with frames/ and render.log")
    parser.add_argument("--graphs", default="build/TestingOutput/techdemo000_graphs")
    parser.add_argument("--seconds", type=float, default=7.0, help="per screen")
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    run, graphs = Path(args.run), Path(args.graphs)
    frames_dir = run / "frames"
    starts = chapter_starts(run / "render.log")
    existing = sorted(int(p.stem.split("_")[1]) for p in frames_dir.glob("frame_*.ppm"))
    if not existing:
        raise SystemExit(f"no frames in {frames_dir}")
    first = max(existing) + 1

    per_screen = round(args.seconds * args.fps)
    fade = round(FADE_SECONDS * args.fps)
    number = first
    for picture, chapter in SCREENS:
        screen = Image.open(graphs / picture).convert("RGB")
        start = starts[chapter]
        end = starts.get(chapter + 1, max(existing) + 1)
        for k in range(per_screen):
            source = frames_dir / f"frame_{start + k % (end - start):04d}.ppm"
            frame = screen.copy()
            clip = Image.open(source).convert("RGB").resize(PIP_SIZE, Image.LANCZOS)
            frame.paste(clip, PIP_POS)
            draw = ImageDraw.Draw(frame)
            x0, y0 = PIP_POS
            draw.rectangle([x0 - 1, y0 - 1, x0 + PIP_SIZE[0], y0 + PIP_SIZE[1]], outline=(200, 200, 200), width=1)
            # Fade in and out of black.
            level = min(1.0, (k + 1) / fade, (per_screen - k) / fade)
            if level < 1.0:
                frame = Image.blend(Image.new("RGB", frame.size), frame, level)
            frame.save(frames_dir / f"frame_{number:04d}.ppm")
            number += 1
        print(f"{picture}: frames {number - per_screen} to {number - 1}, chapter {chapter} from frame {start}")
    print(f"chapter 8: frames {first} to {number - 1}")


if __name__ == "__main__":
    main()
