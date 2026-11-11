# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Prepares the training pairs of the photo enhancer from fly-ins rendered with G-buffers (tree_bonsai_example
--fly-in --gbuffers): every `--stride`-th frame of every flight becomes a sample at the Qwen-Image size (832 x 464,
the 1280 x 720 frame scaled to 832 x 468 and cropped by two rows at the top and bottom):

* ``<out>/<split>/render/<id>.ppm``: the rendering;
* ``<out>/<split>/depth/<id>.ppm``: the inverse depth as the depth ControlNet takes it (flyin_warp.depth_image());
* ``<out>/<split>/class/<id>.pgm``: the rendered class of every pixel (TREE_CLASS_*, nearest);
* ``<out>/<split>/depth_smooth/<id>.ppm``: with `--smooth-control`, the depth blurred by a Gaussian, as the batches'
  control (the 8-bit steps of the plain depth can show as contour lines the ControlNet paints);
* ``<out>/<split>/photo/<id>.ppm``: written later by qwen_image_example from the batch files;
* ``<out>/batches/<split>_NN.txt``: qwen_image_example --batch lists (text-to-image with the depth as control),
  chunks of `--chunk` samples whose order alternates between the flights.

    uv run --project scripts/qwen_image python scripts/neural/prepare_photo_pairs.py build/TestingOutput/gan_data \\
        --train FLIGHT... --validation FLIGHT...
"""

from __future__ import annotations

import argparse
import sys
from itertools import zip_longest
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qwen_image"))
import flyin_warp  # noqa: E402

WIDTH, HEIGHT = 832, 464  # the sample size; set_sample_size() changes it
# tree::treeClassColor() of TREE_CLASS_* 0..6: background, pot, soil, stem, leaf, object, lantern.
CLASS_COLOURS = np.array([[0, 0, 0], [51, 102, 255], [153, 102, 0], [255, 0, 0], [0, 255, 0], [255, 153, 0],
                          [255, 255, 255]], dtype=np.int32)


DEPTH_HIGH = 98.0  # the percentile of the inverse depth that becomes white; set_depth_high() changes it


def set_sample_size(height: int, width: int) -> None:
    global WIDTH, HEIGHT
    WIDTH, HEIGHT = width, height


def set_depth_high(percentile: float) -> None:
    global DEPTH_HIGH
    DEPTH_HIGH = percentile


def to_sample_size(image: np.ndarray, resample) -> np.ndarray:
    """The image scaled to the sample width and cropped to its height around the middle (1280 x 720 -> 832 x 468,
    cropped by two rows at the top and bottom to 832 x 464)."""
    scaled_height = round(image.shape[0] * WIDTH / image.shape[1])
    top = (scaled_height - HEIGHT) // 2
    picture = Image.fromarray(image).resize((WIDTH, scaled_height), resample)
    return np.asarray(picture)[top:top + HEIGHT]


def classes_of(rgb: np.ndarray) -> np.ndarray:
    distances = ((rgb[:, :, None, :].astype(np.int32) - CLASS_COLOURS[None, None]) ** 2).sum(-1)
    return distances.argmin(-1).astype(np.uint8)


def usable(classes: np.ndarray) -> bool:
    """As train_photo_enhancer.usable(): hardly any pot and at most 60 % background."""
    return (classes == 1).mean() <= 0.02 and (classes == 0).mean() <= 0.6


def prepare(flight_dir: Path, split_dir: Path, stride: int, offset: int, skip_unusable: bool = False,
            smooth: float = 0.0) -> list[str]:
    flight = flyin_warp.Flight(flight_dir)
    name = flight_dir.name
    ids = []
    for kind in ("render", "depth", "class", "photo") + (("depth_smooth",) if smooth > 0 else ()):
        (split_dir / kind).mkdir(parents=True, exist_ok=True)
    frames = sorted(flight.cameras)
    # Flights written with --frame-stride are already sparse.
    for frame in frames if len(frames) > 1 and frames[1] - frames[0] >= stride else frames[offset::stride]:
        sample = f"{name}_{frame:04d}"
        classes = to_sample_size(classes_of(flyin_warp.read_ppm(flight.frame_path(frame, "class"))), Image.NEAREST)
        if skip_unusable and not usable(classes):
            continue
        render = flyin_warp.read_ppm(flight.frame_path(frame))
        flyin_warp.write_ppm(split_dir / "render" / f"{sample}.ppm", to_sample_size(render, Image.BICUBIC))
        depth = (flyin_warp.depth_image(flight, frame, high=DEPTH_HIGH) * 255).round().astype(np.uint8)
        depth = to_sample_size(depth, Image.BILINEAR)
        flyin_warp.write_ppm(split_dir / "depth" / f"{sample}.ppm", np.repeat(depth[..., None], 3, axis=2))
        if smooth > 0:
            blurred = np.asarray(Image.fromarray(np.repeat(depth[..., None], 3, axis=2)).filter(
                ImageFilter.GaussianBlur(smooth)))
            flyin_warp.write_ppm(split_dir / "depth_smooth" / f"{sample}.ppm", blurred)
        Image.fromarray(classes).save(split_dir / "class" / f"{sample}.pgm")
        ids.append(sample)
    print(f"  {name}: {len(ids)} samples")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path)
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--validation", type=Path, nargs="*", default=[])
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=20)
    parser.add_argument("--tag", default="", help="part of the batch file names, to add flights to a dataset later")
    parser.add_argument("--skip-unusable", action="store_true", help="leave out views of the pot or mostly sky")
    parser.add_argument("--size", type=int, nargs=2, default=(HEIGHT, WIDTH), metavar=("HEIGHT", "WIDTH"),
                        help="the sample size (default: the Qwen-Image size of the landscape, 464 x 832)")
    parser.add_argument("--depth-high", type=float, default=DEPTH_HIGH,
                        help="percentile of the inverse depth that becomes white (100: the nearest point)")
    parser.add_argument("--smooth-control", type=float, default=0.0,
                        help="Gaussian radius in pixels of the batches' control depth (0: the plain depth)")
    args = parser.parse_args()
    set_sample_size(*args.size)
    set_depth_high(args.depth_high)
    batches = args.out / "batches"
    batches.mkdir(parents=True, exist_ok=True)
    for split, flights in (("train", args.train), ("validation", args.validation)):
        split_dir = args.out / split
        per_flight = [prepare(f, split_dir, args.stride, (3 * k) % args.stride, args.skip_unusable, args.smooth_control)
                      for k, f in enumerate(flights)]
        control = "depth_smooth" if args.smooth_control > 0 else "depth"
        # Alternate between the flights, so that every prefix of the batches is varied.
        order = [s for group in zip_longest(*per_flight) for s in group if s is not None]
        for n in range(0, len(order), args.chunk):
            lines = [f"- {split_dir}/photo/{s}.ppm {split_dir}/{control}/{s}.ppm" for s in order[n:n + args.chunk]]
            (batches / f"{split}{args.tag}_{n // args.chunk:02d}.txt").write_text("\n".join(lines) + "\n")
        print(f"{split}: {len(order)} samples in {(len(order) + args.chunk - 1) // args.chunk} batches")


if __name__ == "__main__":
    main()
