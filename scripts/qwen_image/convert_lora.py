# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Converts a Qwen-Image LoRA (safetensors with lora_down / lora_up / alpha per linear layer, e.g.
Qwen-Image-Lightning) into a GGUF for Klartraum's QwenImageTransformer.

The GGUF holds ``<layer>.lora_down.weight`` [rank][in] and ``<layer>.lora_up.weight`` [out][rank] in BF16 under the
transformer's tensor names, with alpha / rank folded into lora_up (exact when it is a power of two, as Lightning's
8 / 64; otherwise lora_up is stored in F32). The transformer adds up @ (down @ x) to each layer's output.

    uv run --project scripts/qwen_image python scripts/qwen_image/convert_lora.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
LIGHTNING = ROOT / "build/models/qwen-image/lightning"


def bf16_bits(tensor: torch.Tensor) -> np.ndarray:
    return tensor.to(torch.bfloat16).contiguous().view(torch.int16).numpy().view(np.uint16)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lora", type=Path, default=LIGHTNING / "Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors")
    parser.add_argument("--out", type=Path, default=LIGHTNING / "Qwen-Image-Lightning-8steps-V2.0.gguf")
    args = parser.parse_args()

    source = safe_open(args.lora, "pt")
    layers = sorted(key.removesuffix(".alpha") for key in source.keys() if key.endswith(".alpha"))
    writer = gguf.GGUFWriter(args.out, "qwen_image_lora")
    writer.add_string("general.name", args.lora.stem)
    writer.add_string("general.source.url", "https://huggingface.co/lightx2v/Qwen-Image-Lightning")
    for layer in layers:
        down = source.get_tensor(layer + ".lora_down.weight")
        up = source.get_tensor(layer + ".lora_up.weight").float()
        scale = float(source.get_tensor(layer + ".alpha")) / down.shape[0]
        writer.add_tensor(layer + ".lora_down.weight", bf16_bits(down), raw_shape=list(down.shape),
                          raw_dtype=gguf.GGMLQuantizationType.BF16)
        up = up * scale
        if math.log2(scale).is_integer():
            writer.add_tensor(layer + ".lora_up.weight", bf16_bits(up), raw_shape=list(up.shape),
                              raw_dtype=gguf.GGMLQuantizationType.BF16)
        else:
            writer.add_tensor(layer + ".lora_up.weight", up.numpy())
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"{len(layers)} layers -> {args.out}")


if __name__ == "__main__":
    main()
