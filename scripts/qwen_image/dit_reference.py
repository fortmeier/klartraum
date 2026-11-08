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

    uv run --project scripts/qwen_image python scripts/qwen_image/dit_reference.py [--lora]
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--out", type=Path, default=MODELS / "reference")
    args = parser.parse_args()
    torch.manual_seed(0)

    weights = Weights(args.gguf, args.lora_file if args.lora else None)
    text = np.fromfile(args.out / "text_hidden_f32.bin", dtype=np.float32).reshape(-1, TEXT_DIM)[TEMPLATE_TOKENS:]
    text = torch.from_numpy(np.ascontiguousarray(text)).unsqueeze(0)
    rows, columns = args.height // 2, args.width // 2
    latent = torch.randn(1, rows * columns, 64)
    suffix = "_lora" if args.lora else ""
    if not args.lora:
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

        image = img_in(latent)
        context = txt_in(txt_norm(text))
        temb = time_embed(torch.tensor([args.sigma]), image)
        rotary = rope([[(1, rows, columns)]], max_txt_seq_len=text.shape[1], device=torch.device("cpu"))
        for index in range(args.blocks):
            block = QwenImageTransformerBlock(DIM, HEADS, HEAD_DIM).eval()
            weights.load(block, f"transformer_blocks.{index}.")
            context, image = block(image, context, None, temb, rotary)
            if index == 0 and not args.lora:
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
