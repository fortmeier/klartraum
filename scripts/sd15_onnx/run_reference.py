# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Run the Stable Diffusion 1.5 VAE encoder and decoder on lantern.jpg."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image, ImageOps

from disclaimer import print_stable_diffusion_notice


MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_INPUT = REPO_ROOT / "data" / "lantern.jpg"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "build" / "TestingOutput" / "sd15_onnx"


def load_image(path: Path, size: int) -> torch.Tensor:
    with Image.open(path) as image:
        image = ImageOps.fit(
            image.convert("RGB"),
            (size, size),
            method=Image.Resampling.LANCZOS,
        )
        pixels = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)


def save_image(tensor: torch.Tensor, path: Path) -> None:
    pixels = tensor.detach().float().cpu().squeeze(0).permute(1, 2, 0)
    pixels = ((pixels.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    print_stable_diffusion_notice()
    args = parse_args()
    if args.size <= 0 or args.size % 8:
        raise ValueError("--size must be a positive multiple of 8")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    dtype = torch.float32
    print(f"Loading {args.model} VAE on {device}")
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae", torch_dtype=dtype)
    vae = vae.eval().to(device)
    image = load_image(args.input, args.size).to(device=device, dtype=dtype)

    with torch.inference_mode():
        moments = vae.quant_conv(vae.encoder(image))
        latent = torch.chunk(moments, 2, dim=1)[0]
        decoded = vae.decoder(vae.post_quant_conv(latent))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_image(decoded, args.output_dir / "lantern_sd15_vae_reference.png")
    np.save(args.output_dir / "lantern_sd15_latent.npy", latent.float().cpu().numpy())
    print(f"image:   {tuple(image.shape)}")
    print(f"moments: {tuple(moments.shape)}")
    print(f"latent:  {tuple(latent.shape)}")
    print(f"decoded: {tuple(decoded.shape)}")
    print(f"Wrote reference artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
