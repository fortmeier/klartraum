# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Downloads a Qwen3.6-27B GGUF file for examples/qwen_chat_example.cpp.

The default quantization depends on the machine's memory: the whole model
stays resident in (unified) GPU memory, so 32 GB machines and smaller get
Q3_K_M (13.6 GB) and larger ones Q4_K_M (16.8 GB).

Usage (from this directory):
    uv run python download.py                  # memory-dependent default
    uv run python download.py --quant Q4_K_M
    uv run python download.py --list           # show the available files
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

from disclaimer import print_qwen_notice

REPOSITORY = "unsloth/Qwen3.6-27B-GGUF"
# Quantizations whose tensor types (Q3_K .. Q8_0, F16, BF16, F32) Klartraum's
# GGUF runtime dequantizes. IQ* and Q4_0/Q4_1 files are not supported.
SUPPORTED = ["Q3_K_S", "Q3_K_M", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0"]
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIRECTORY = REPO_ROOT / "data" / "gguf" / "qwen3.6-27b"


def physical_memory_bytes() -> int:
    if platform.system() == "Darwin":
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 0


def default_quant() -> str:
    return "Q3_K_M" if physical_memory_bytes() <= 32 * 1024**3 else "Q4_K_M"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quant", choices=SUPPORTED, default=None, help="quantization (default: by memory size)")
    parser.add_argument("--output", type=Path, default=DEFAULT_DIRECTORY, help="target directory")
    parser.add_argument("--list", action="store_true", help="list the repository's GGUF files and exit")
    args = parser.parse_args()
    print_qwen_notice()

    from huggingface_hub import HfApi, hf_hub_download

    if args.list:
        info = HfApi().model_info(REPOSITORY, files_metadata=True)
        for sibling in sorted(info.siblings, key=lambda s: s.rfilename):
            if sibling.rfilename.endswith(".gguf"):
                quant = sibling.rfilename.removeprefix("Qwen3.6-27B-").removesuffix(".gguf")
                mark = "supported" if quant in SUPPORTED else ""
                print(f"{sibling.rfilename:40s} {sibling.size / 1e9:6.1f} GB  {mark}")
        return 0

    quant = args.quant or default_quant()
    filename = f"Qwen3.6-27B-{quant}.gguf"
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPOSITORY}/{filename} to {args.output}", flush=True)
    path = hf_hub_download(REPOSITORY, filename, local_dir=args.output)
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
