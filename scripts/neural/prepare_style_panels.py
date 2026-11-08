# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow", "pyyaml"]
# ///

"""Prepares the side panels of tech demo 001's days (techdemo001_example --panels DIR) from the watercolor training
data: a training sample of a summer day (its rendering, classes, depth and watercolor target) and a gallery of six
watercolor targets of the seasons by day and by night. They go to --out as binary PPM files at the panels' sizes:

- sample_rendering.ppm, sample_classes.ppm, sample_depth.ppm, sample_target.ppm (--sample-width x height);
- gallery_0.ppm ... gallery_5.ppm (--target-width x height), watercolor targets of summer, fall and winter by day and
  of spring, fall and winter by night, each the view the bonsai fills most among those within 0.3 of its season;
  gallery.txt
  lists them as "file season time" lines.

The views are the dataset's (techdemo001_example --dataset); day panels take views of a white background, night panels
views of a black one (within 0.05; winter's midday is a little grey; not of the twilight between).

    uv run scripts/neural/prepare_style_panels.py --dataset DIR --styles DIR --out DIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

SEASONS = {"winter": 0.0, "spring": 1.25, "summer": 2.0, "fall": 3.25}
# Every season once, days to the left and nights to the right in the panels' 2 x 3 grid.
GALLERY = [("summer", "day"), ("spring", "night"), ("fall", "day"), ("fall", "night"), ("winter", "day"),
           ("winter", "night")]
BACKGROUNDS = {"day": 1.0, "night": 0.0}


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
    parser.add_argument("--target-width", type=int, default=224)
    parser.add_argument("--target-height", type=int, default=126)
    args = parser.parse_args()
    views = [v for v in yaml.safe_load((args.dataset / "dataset.yaml").read_text())["images"]
             if (args.styles / v["file"]).exists()]
    # How much of each view the bonsai fills: the share of pixels with a class.
    for v in views:
        classes = np.asarray(Image.open(args.dataset / v["classes"]).convert("RGB"))
        v["filled"] = float((classes.max(-1) > 20).mean())
    large = [v for v in views if v["filled"] > 0.12] or views
    used: set[str] = set()

    def nearest(season: float, time: float, background: float | None = None) -> dict:
        candidates = [v for v in large if v["file"] not in used and
                      (background is None or abs(v["background"] - background) < 0.01)]
        choice = min(candidates,
                     key=lambda v: season_distance(v["season"], season) + 2.0 * time_distance(v["time_of_day"], time))
        used.add(choice["file"])
        return choice

    args.out.mkdir(parents=True, exist_ok=True)
    sample = nearest(SEASONS["summer"], 0.55, BACKGROUNDS["day"])
    size = (args.sample_width, args.sample_height)
    write_ppm(Image.open(args.dataset / sample["file"]), size, args.out / "sample_rendering.ppm")
    write_ppm(Image.open(args.dataset / sample["classes"]), size, args.out / "sample_classes.ppm")
    write_ppm(Image.open(args.dataset / sample["depth"]), size, args.out / "sample_depth.ppm")
    write_ppm(Image.open(args.styles / sample["file"]), size, args.out / "sample_target.ppm")
    print(f"sample: {sample['file']} (season {sample['season']:.2f}, {sample['clock']})")
    # The gallery beside the seasons: six close views, three days and three nights, every season.
    size = (args.target_width, args.target_height)
    lines = []
    for k, (name, kind) in enumerate(GALLERY):
        candidates = [v for v in views if v["file"] not in used and
                      abs(v["background"] - BACKGROUNDS[kind]) < 0.05 and
                      season_distance(v["season"], SEASONS[name]) < 0.3]
        view = max(candidates, key=lambda v: v["filled"])
        used.add(view["file"])
        write_ppm(Image.open(args.styles / view["file"]), size, args.out / f"gallery_{k}.ppm")
        lines.append(f"gallery_{k}.ppm {name} {kind}")
        print(f"gallery {k}, {name} {kind}: {view['file']} (season {view['season']:.2f}, {view['clock']}, "
              f"fills {view['filled']:.0%})")
    (args.out / "gallery.txt").write_text("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
