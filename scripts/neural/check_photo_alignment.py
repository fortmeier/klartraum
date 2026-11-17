# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Scores how well every Qwen-Image photograph of the photo enhancer's training data follows the structure of its
rendering: the Spearman rank correlation between the rendering's inverse depth (the ControlNet's input) and the
inverse depth Depth Anything V2 Metric Outdoor Small estimates from the photograph, over the pixels the rendering
covers. A photograph that invents its own scene (e.g. a horizon above a rendered slope) scores low.

Writes <split>/alignment.json ({sample: correlation}) for every split and prints the distribution; the training
(train_photo_enhancer.py --min-alignment) leaves out samples below a threshold.

    uv run --project scripts/qwen_image python scripts/neural/check_photo_alignment.py build/TestingOutput/gan_data
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForDepthEstimation

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "build/models/depth/da2-metric-outdoor-small"


def ranks(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    result = np.empty(len(values))
    result[order] = np.arange(len(values))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path)
    args = parser.parse_args()
    model = AutoModelForDepthEstimation.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    for split in ("train", "validation"):
        directory = args.data / split
        scores_path = directory / "alignment.json"
        scores = json.loads(scores_path.read_text()) if scores_path.exists() else {}
        for photo_path in sorted((directory / "photo").glob("*.ppm")):
            sample = photo_path.stem
            if sample in scores:
                continue
            photo = np.asarray(Image.open(photo_path).convert("RGB").resize((924, 518), Image.BICUBIC))
            x = torch.from_numpy(photo.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
            with torch.no_grad():
                metres = model(pixel_values=(x - mean) / std).predicted_depth[None]
            depth = np.asarray(Image.open(directory / "depth" / f"{sample}.ppm").convert("L"), dtype=np.float32)
            estimated = 1.0 / F.interpolate(metres, size=depth.shape, mode="bilinear")[0, 0].clamp(min=0.1).numpy()
            covered = depth > 0
            if covered.sum() < 1000:
                scores[sample] = -1.0
                continue
            # Every 7th pixel is plenty for a rank correlation.
            a, b = depth[covered][::7], estimated[covered][::7]
            scores[sample] = float(np.corrcoef(ranks(a), ranks(b))[0, 1])
        scores_path.write_text(json.dumps(scores, indent=1))
        values = np.array(list(scores.values()))
        if len(values):
            print(f"{split}: {len(values)} photographs, correlation quartiles "
                  f"{np.percentile(values, [10, 25, 50, 75]).round(2)}; below 0.5: {(values < 0.5).sum()}")


if __name__ == "__main__":
    main()
