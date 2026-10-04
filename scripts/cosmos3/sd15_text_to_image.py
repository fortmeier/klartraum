# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Generate one Stable Diffusion 1.5 image with the diffusers reference pipeline.

First step of the dashcam example (see README.md): the image becomes the
Cosmos3 conditioning frame. It runs in this uv project because the
``scripts/sd15_onnx`` project pins ``onnxruntime-gpu``, which has no macOS wheel.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline


MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SD15_NOTICE = """\
WARNING: Stable Diffusion is not part of Klartraum.
  This script downloads and processes third-party Stable Diffusion 1.5 model
  weights, which are distributed by their respective authors under their own
  license (CreativeML Open RAIL-M). Klartraum neither ships nor endorses these
  models. You are solely responsible for complying with the model license and
  for how you use the models and anything they generate.
"""


def main() -> None:
    print(SD15_NOTICE, file=sys.stderr, flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="blurry, distorted, low quality, cartoon, painting, text, watermark")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=12)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.float32, safety_checker=None)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    started = time.perf_counter()
    image = pipe(
        args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.size,
        width=args.size,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator("cpu").manual_seed(args.seed),
    ).images[0]
    print(f"generated in {time.perf_counter() - started:.1f}s", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
