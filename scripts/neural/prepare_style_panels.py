# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow", "pyyaml"]
# ///

"""Prepares the side panels of tech demo 001's days (techdemo001_example --panels DIR) from the watercolor training
data: a training sample of a summer day (its rendering, classes, depth and watercolor target) and, for each season, a
watercolor target by day and by night. They go to --out as binary PPM files at the panels' sizes:

- sample_rendering.ppm, sample_classes.ppm, sample_depth.ppm, sample_target.ppm (--sample-width x height);
- target_<season>_<day|night>.ppm for winter, spring, summer and fall (--target-width x height).

The views are the dataset's (techdemo001_example --dataset) whose season and time of day lie nearest to each panel's,
among those that show the bonsai large enough.

    uv run scripts/neural/prepare_style_panels.py --dataset DIR --styles DIR --out DIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

SEASONS = {"winter": 0.0, "spring": 1.25, "summer": 2.0, "fall": 3.25}
TIMES = {"day": 0.5, "night": 0.0}


def season_distance(a: float, b: float) -> float:
    d = abs(a - b) % 4.0
    return min(d, 4.0 - d)


def time_distance(a: float, b: float) -> float:
    d = abs(a - b) % 1.0
    return min(d, 1.0 - d)


def write_ppm(image: Image.Image, size: tuple[int, int], path: Path) -> None:
    rgb = np.asarray(image.convert("RGB").resize(size, Image.LANCZOS), np.uint8)
    with open(path, "wb") as out:
        out.write(f"P6\n{size[0]} {size[1]}\n255\n".encode())
        out.write(rgb.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--styles", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-width", type=int, default=224)
    parser.add_argument("--sample-height", type=int, default=126)
    parser.add_argument("--target-width", type=int, default=320)
    parser.add_argument("--target-height", type=int, default=180)
    args = parser.parse_args()
    views = [v for v in yaml.safe_load((args.dataset / "dataset.yaml").read_text())["images"]
             if (args.styles / v["file"]).exists()]
    # How much of each view the bonsai fills: the share of pixels with a class.
    for v in views:
        classes = np.asarray(Image.open(args.dataset / v["classes"]).convert("RGB"))
        v["filled"] = float((classes.max(-1) > 20).mean())
    large = [v for v in views if v["filled"] > 0.12] or views
    used: set[str] = set()

    def nearest(season: float, time: float) -> dict:
        choice = min((v for v in large if v["file"] not in used),
                     key=lambda v: season_distance(v["season"], season) + 2.0 * time_distance(v["time_of_day"], time))
        used.add(choice["file"])
        return choice

    args.out.mkdir(parents=True, exist_ok=True)
    sample = nearest(SEASONS["summer"], 0.55)
    size = (args.sample_width, args.sample_height)
    write_ppm(Image.open(args.dataset / sample["file"]), size, args.out / "sample_rendering.ppm")
    write_ppm(Image.open(args.dataset / sample["classes"]), size, args.out / "sample_classes.ppm")
    write_ppm(Image.open(args.dataset / sample["depth"]), size, args.out / "sample_depth.ppm")
    write_ppm(Image.open(args.styles / sample["file"]), size, args.out / "sample_target.ppm")
    print(f"sample: {sample['file']} (season {sample['season']:.2f}, {sample['clock']})")
    size = (args.target_width, args.target_height)
    for name, season in SEASONS.items():
        for kind, time in TIMES.items():
            view = nearest(season, time)
            write_ppm(Image.open(args.styles / view["file"]), size, args.out / f"target_{name}_{kind}.ppm")
            print(f"{name} {kind}: {view['file']} (season {view['season']:.2f}, {view['clock']})")


if __name__ == "__main__":
    main()
