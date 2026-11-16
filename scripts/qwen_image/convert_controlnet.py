# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Converts the Qwen-Image ControlNet Union (InstantX/Qwen-Image-ControlNet-Union, diffusers safetensors in BF16)
into a GGUF for Klartraum's QwenImageTransformer: matrices in Q8_0 (1.9 GB instead of 3.5 GB, so the ControlNet fits
on the GPU next to the transformer), vectors in F32, under diffusers' tensor names. Prints the worst relative error
of the quantized matrices.

    uv run --project scripts/qwen_image python scripts/qwen_image/convert_controlnet.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gguf
import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "build/models/qwen-image/controlnet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DIRECTORY / "diffusion_pytorch_model.safetensors")
    parser.add_argument("--out", type=Path, default=DIRECTORY / "qwen-image-controlnet-union-Q8_0.gguf")
    args = parser.parse_args()

    source = safe_open(args.source, "pt")
    writer = gguf.GGUFWriter(args.out, "qwen_image_controlnet")
    writer.add_string("general.name", "Qwen-Image-ControlNet-Union")
    writer.add_string("general.source.url", "https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union")
    worst = 0.0
    for name in sorted(source.keys()):
        values = source.get_tensor(name).float().numpy()
        if values.ndim == 2 and values.shape[1] % 32 == 0:
            quantized = gguf.quants.quantize(values, gguf.GGMLQuantizationType.Q8_0)
            restored = gguf.quants.dequantize(quantized, gguf.GGMLQuantizationType.Q8_0).reshape(values.shape)
            scale = np.abs(values).max(axis=1, keepdims=True) + 1e-12
            worst = max(worst, float((np.abs(restored - values) / scale).max()))
            writer.add_tensor(name, quantized, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        else:
            writer.add_tensor(name, np.ascontiguousarray(values, dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"{args.out}: worst Q8_0 error {worst:.4f} of a row's largest magnitude")


if __name__ == "__main__":
    main()
