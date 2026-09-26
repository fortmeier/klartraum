"""Smoke-test the pinned CUDA stack and gsplat extension."""

from __future__ import annotations

import numpy
import torch
from gsplat.cuda._backend import _C  # noqa: F401


print(f"NumPy: {numpy.__version__}")
print(f"PyTorch: {torch.__version__} (CUDA {torch.version.cuda})")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print("gsplat CUDA backend loaded")
