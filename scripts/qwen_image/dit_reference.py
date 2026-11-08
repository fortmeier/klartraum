# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Reference outputs of Qwen-Image's transformer (the 60-block MMDiT) from a GGUF file.

Builds diffusers' modules (QwenImageTransformerBlock, QwenEmbedRope, QwenTimestepProjEmbeddings,
AdaLayerNormContinuous, ...) in float32 and runs them one block at a time on weights dequantized from the GGUF (the
whole model in float32 would need ~80 GB), for a small latent and the prompt embedding of text_reference.py. With
``--lora``, the Lightning LoRA is merged into the dequantized weights (W + alpha / rank * up @ down).

Writes into build/models/qwen-image/reference/ for tests/test_qwen_image_transformer.cpp:

* ``dit_config.txt``: latent height and width (in latent pixels), text tokens, sigma;
* ``dit_latent_f32.bin``: the packed latent [h/2 * w/2][64] (random, seeded); ``dit_text_f32.bin``: the prompt
  embedding [tokens][3584];
* ``dit_block0_image_f32.bin`` / ``dit_block0_text_f32.bin``: the image and text states after the first block;
* ``dit_output_f32.bin``: the velocity prediction [h/2 * w/2][64] (``dit_output_lora_f32.bin`` with ``--lora``).

With ``--control``, the ControlNet Union (the Q8_0 GGUF of convert_controlnet.py) conditions the transformer on a
depth image of the fly-in (flyin_warp.depth_image()) encoded by diffusers' VAE:

* ``dit_control_f32.bin``: the packed control latent [h/2 * w/2][64];
* ``dit_control_residuals_f32.bin``: the ControlNet's 5 residuals [5][h/2 * w/2][3072] (conditioning scale 1);
* ``dit_output_control_f32.bin``: the controlled velocity (``dit_output_control_lora_f32.bin`` with ``--lora``).

    uv run --project scripts/qwen_image python scripts/qwen_image/dit_reference.py [--lora] [--control]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import gguf
import numpy as np
import torch
from diffusers.models.normalization import AdaLayerNormContinuous, RMSNorm
from diffusers.models.transformers.transformer_qwenimage import (
    QwenEmbedRope,
    QwenImageTransformerBlock,
    QwenTimestepProjEmbeddings,
)
from safetensors import safe_open

import flyin_warp

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "build/models/qwen-image"
DIM, HEADS, HEAD_DIM, TEXT_DIM, TEMPLATE_TOKENS = 3072, 24, 128, 3584, 34


class Weights:
    """Dequantized GGUF tensors, optionally with a LoRA merged in."""

    def __init__(self, path: Path, lora: Path | None):
        self.tensors = {t.name: t for t in gguf.GGUFReader(path).tensors}
        self.lora = safe_open(lora, "pt") if lora else None
        self.lora_keys = set(self.lora.keys()) if self.lora else set()

    def get(self, name: str) -> torch.Tensor:
        tensor = self.tensors[name]
        values = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
        shape = [int(v) for v in reversed(tensor.shape)]
        weight = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32).reshape(shape))
        base = name.removesuffix(".weight")
        if name.endswith(".weight") and base + ".lora_down.weight" in self.lora_keys:
            down = self.lora.get_tensor(base + ".lora_down.weight").float()
            up = self.lora.get_tensor(base + ".lora_up.weight").float()
            alpha = float(self.lora.get_tensor(base + ".alpha"))
            weight = weight + (alpha / down.shape[0]) * (up @ down)
        return weight

    def load(self, module: torch.nn.Module, prefix: str) -> None:
        names = [name for name, _ in module.named_parameters()]
        module.load_state_dict({name: self.get(prefix + name) for name in names}, strict=True)


def control_residuals(args, latent: torch.Tensor, text: torch.Tensor, rotary) -> list[torch.Tensor]:
    """The ControlNet's residuals for a depth image of the fly-in, and the fixtures of its inputs and outputs."""
    from diffusers import AutoencoderKLQwenImage

    flight = flyin_warp.Flight(args.flight)
    depth = flyin_warp.depth_image(flight, 300)
    height, width = args.height * 8, args.width * 8
    picture = torch.from_numpy(depth).float().view(1, 1, *depth.shape)
    picture = torch.nn.functional.interpolate(picture, size=(height, width), mode="area").clamp(0, 1)
    picture = (picture * 2 - 1).expand(1, 3, height, width).unsqueeze(2)
    vae = AutoencoderKLQwenImage.from_pretrained(MODELS / "original", subfolder="vae", torch_dtype=torch.float32)
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1)
    with torch.no_grad():
        control = (vae.encode(picture).latent_dist.mode()[:, :, 0] - mean) / std
    rows, columns = args.height // 2, args.width // 2
    packed = control.view(1, 16, rows, 2, columns, 2).permute(0, 2, 4, 1, 3, 5).reshape(1, rows * columns, 64)
    packed[0].numpy().tofile(args.out / "dit_control_f32.bin")

    net = Weights(args.controlnet, None)
    time_embed = QwenTimestepProjEmbeddings(DIM)
    net.load(time_embed, "time_text_embed.")
    txt_norm = RMSNorm(TEXT_DIM, eps=1e-6)
    net.load(txt_norm, "txt_norm.")
    img_in, x_embedder, txt_in = torch.nn.Linear(64, DIM), torch.nn.Linear(64, DIM), torch.nn.Linear(TEXT_DIM, DIM)
    net.load(img_in, "img_in.")
    net.load(x_embedder, "controlnet_x_embedder.")
    net.load(txt_in, "txt_in.")
    latent = torch.from_numpy(np.fromfile(args.out / "dit_latent_f32.bin", dtype=np.float32)).view(1, -1, 64)
    with torch.no_grad():
        image = img_in(latent) + x_embedder(packed)
        temb = time_embed(torch.tensor([args.sigma]), image)
        context = txt_in(txt_norm(text))
        residuals = []
        for index in range(5):
            block = QwenImageTransformerBlock(DIM, HEADS, HEAD_DIM).eval()
            net.load(block, f"transformer_blocks.{index}.")
            context, image = block(image, context, None, temb, rotary)
            projection = torch.nn.Linear(DIM, DIM)
            net.load(projection, f"controlnet_blocks.{index}.")
            residuals.append(projection(image))
    torch.cat(residuals)[None].numpy().tofile(args.out / "dit_control_residuals_f32.bin")
    print(f"  control residuals: |r| max {max(float(r.abs().max()) for r in residuals):.1f}", flush=True)
    return residuals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--height", type=int, default=16, help="latent height (image height / 8)")
    parser.add_argument("--width", type=int, default=16, help="latent width (image width / 8)")
    parser.add_argument("--sigma", type=float, default=0.75)
    parser.add_argument("--blocks", type=int, default=60)
    parser.add_argument("--lora", action="store_true", help="merge the Lightning LoRA")
    parser.add_argument("--gguf", type=Path, default=MODELS / "transformer/qwen-image-Q4_K_S.gguf")
    parser.add_argument("--lora-file", type=Path,
                        default=MODELS / "lightning/Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors")
    parser.add_argument("--control", action="store_true", help="condition on a depth image with the ControlNet")
    parser.add_argument("--controlnet", type=Path, default=MODELS / "controlnet/qwen-image-controlnet-union-Q8_0.gguf")
    parser.add_argument("--flight", type=Path, default=ROOT / "build/TestingOutput/qwen_flyin",
                        help="fly-in whose depth G-buffer gives the control image")
    parser.add_argument("--out", type=Path, default=MODELS / "reference")
    args = parser.parse_args()
    torch.manual_seed(0)

    weights = Weights(args.gguf, args.lora_file if args.lora else None)
    text = np.fromfile(args.out / "text_hidden_f32.bin", dtype=np.float32).reshape(-1, TEXT_DIM)[TEMPLATE_TOKENS:]
    text = torch.from_numpy(np.ascontiguousarray(text)).unsqueeze(0)
    rows, columns = args.height // 2, args.width // 2
    latent = torch.randn(1, rows * columns, 64)
    suffix = ("_control" if args.control else "") + ("_lora" if args.lora else "")
    if not args.lora and not args.control:
        (args.out / "dit_config.txt").write_text(f"{args.height} {args.width} {text.shape[1]} {args.sigma}\n")
        latent[0].numpy().tofile(args.out / "dit_latent_f32.bin")
        text[0].numpy().tofile(args.out / "dit_text_f32.bin")

    with torch.no_grad():
        time_embed = QwenTimestepProjEmbeddings(DIM)
        weights.load(time_embed, "time_text_embed.")
        txt_norm = RMSNorm(TEXT_DIM, eps=1e-6)
        weights.load(txt_norm, "txt_norm.")
        img_in, txt_in = torch.nn.Linear(64, DIM), torch.nn.Linear(TEXT_DIM, DIM)
        weights.load(img_in, "img_in.")
        weights.load(txt_in, "txt_in.")
        rope = QwenEmbedRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)

        rotary = rope([[(1, rows, columns)]], max_txt_seq_len=text.shape[1], device=torch.device("cpu"))
        residuals = control_residuals(args, latent, text, rotary) if args.control else None
        image = img_in(latent)
        context = txt_in(txt_norm(text))
        temb = time_embed(torch.tensor([args.sigma]), image)
        for index in range(args.blocks):
            block = QwenImageTransformerBlock(DIM, HEADS, HEAD_DIM).eval()
            weights.load(block, f"transformer_blocks.{index}.")
            context, image = block(image, context, None, temb, rotary)
            if residuals is not None:
                # diffusers: block i gets residual i // ceil(blocks / residuals).
                image = image + residuals[index // math.ceil(60 / len(residuals))]
            if index == 0 and not args.lora and not args.control:
                image[0].numpy().tofile(args.out / "dit_block0_image_f32.bin")
                context[0].numpy().tofile(args.out / "dit_block0_text_f32.bin")
            print(f"  block {index}: |image| max {image.abs().max():.1f}, |text| max {context.abs().max():.1f}",
                  flush=True)
        norm_out = AdaLayerNormContinuous(DIM, DIM, elementwise_affine=False, eps=1e-6)
        weights.load(norm_out, "norm_out.")
        proj_out = torch.nn.Linear(DIM, 64)
        weights.load(proj_out, "proj_out.")
        output = proj_out(norm_out(image, temb))
    if args.blocks == 60:
        output[0].numpy().tofile(args.out / f"dit_output{suffix}_f32.bin")
    print(f"output: range [{output.min():.3f}, {output.max():.3f}]")


if __name__ == "__main__":
    main()
