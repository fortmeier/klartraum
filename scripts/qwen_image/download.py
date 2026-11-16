# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Downloads the Qwen-Image files Klartraum uses into build/models/qwen-image/ (ignored by git, ~18 GB).

* original/: configs, tokenizer and VAE of Qwen/Qwen-Image (not the 41 GB transformer and text encoder weights);
* transformer/qwen-image-Q4_K_S.gguf: city96/Qwen-Image-gguf;
* text/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf: unsloth/Qwen2.5-VL-7B-Instruct-GGUF;
* lightning/Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors: lightx2v/Qwen-Image-Lightning.

    uv run --project scripts/qwen_image python scripts/qwen_image/download.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[2]

NOTICE = """\
WARNING: Qwen-Image is not part of Klartraum.
  This script downloads third-party model weights: Qwen-Image and Qwen2.5-VL by Alibaba Cloud's Qwen team, GGUF
  conversions by city96 and Unsloth, the Lightning LoRA by lightx2v, all published under the Apache 2.0 license.
  Klartraum neither ships nor endorses these models. You are solely responsible for complying with their licenses
  and for how you use the models and anything they generate.
"""

FILES = [
    ("city96/Qwen-Image-gguf", "qwen-image-Q4_K_S.gguf", "transformer"),
    ("unsloth/Qwen2.5-VL-7B-Instruct-GGUF", "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf", "text"),
    ("lightx2v/Qwen-Image-Lightning", "Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors", "lightning"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=ROOT / "build/models/qwen-image")
    args = parser.parse_args()
    print(NOTICE, file=sys.stderr, flush=True)
    snapshot_download("Qwen/Qwen-Image", local_dir=args.output / "original",
                      allow_patterns=["model_index.json", "scheduler/*", "tokenizer/*", "vae/*", "*/config.json"])
    for repository, filename, directory in FILES:
        print(hf_hub_download(repository, filename, local_dir=args.output / directory), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
