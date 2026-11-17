# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Temporal stability of an enhanced fly-in (enhance_flight.py's frames): every enhanced frame t is compared with the
frame `--offset` frames earlier reprojected into it (flyin_warp.warp() with the flight's depth and cameras), over the
pixels both frames see; prints the mean absolute difference (in 8-bit steps) and that of the renderings themselves as
the floor of what moving light and wind leave.

    uv run --project scripts/qwen_image python scripts/neural/temporal_error.py build/TestingOutput/qwen_flyin \\
        build/TestingOutput/gan_runs/v6/hourly/<label>/flight/frames --every 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qwen_image"))
import flyin_warp  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("flight", type=Path)
    parser.add_argument("frames", type=Path, help="enhance_flight.py's frames directory")
    parser.add_argument("--every", type=int, default=1, help="the frame step enhance_flight.py used")
    parser.add_argument("--first", type=int, default=240, help="first flight frame (the forest from 240 on)")
    parser.add_argument("--pair", type=int, default=1, help="compare frames this many enhanced frames apart")
    args = parser.parse_args()
    flight = flyin_warp.Flight(args.flight)
    grid = flyin_warp.ImageGrid(832, 464, flight.width, flight.height, top=2, scaled_height=468)
    frames = sorted(flight.cameras)[::args.every]
    enhanced_errors, render_errors = [], []
    for n in range(args.pair, len(frames), args.pair):
        source, target = frames[n - args.pair], frames[n]
        if source < args.first:
            continue
        side = [np.asarray(Image.open(args.frames / f"side_{k:04d}.png").convert("RGB")).astype(np.float64)
                for k in (n - args.pair, n)]
        render = [s[:, :832] for s in side]
        enhanced = [s[:, 832:] for s in side]
        warped, valid = flyin_warp.warp(flight, source, target, enhanced[0], grid)
        enhanced_errors.append(np.abs(warped - enhanced[1])[valid].mean())
        warped, valid = flyin_warp.warp(flight, source, target, render[0], grid)
        render_errors.append(np.abs(warped - render[1])[valid].mean())
    print(f"{args.frames}: enhanced {np.mean(enhanced_errors):.2f}, rendering {np.mean(render_errors):.2f} "
          f"(mean |frame - reprojected previous| over {len(enhanced_errors)} pairs, 8-bit steps)")


if __name__ == "__main__":
    main()
