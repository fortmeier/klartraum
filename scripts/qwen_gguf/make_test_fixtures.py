# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Writes tests/data/gguf/quant_blocks.gguf for tests/test_gguf_quants.cpp.

For every weight type Klartraum decodes, the file holds a tensor `<TYPE>` of
random quantized blocks (4 rows of 512 values) and `<TYPE>.expected`, the
same values decoded by gguf-py's reference dequantization, as F32. The block
scales are drawn as finite half floats so every value is finite.

Usage (from this directory): uv run python make_test_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType as Q
from gguf import GGUFWriter
from gguf.constants import GGML_QUANT_SIZES
from gguf.quants import dequantize

ROWS, COLUMNS = 4, 512
OUTPUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "gguf" / "quant_blocks.gguf"

# Byte offsets of the half-float scales inside each block.
HALF_OFFSETS = {
    Q.Q8_0: [0],
    Q.Q3_K: [108],
    Q.Q4_K: [0, 2],
    Q.Q5_K: [0, 2],
    Q.Q6_K: [208],
}


def random_blocks(rng: np.random.Generator, qtype: Q) -> np.ndarray:
    block_size, block_bytes = GGML_QUANT_SIZES[qtype]
    blocks = ROWS * COLUMNS // block_size
    data = rng.integers(0, 256, size=(blocks, block_bytes), dtype=np.uint8)
    for offset in HALF_OFFSETS[qtype]:
        halves = rng.uniform(-0.05, 0.05, size=blocks).astype(np.float16)
        data[:, offset : offset + 2] = halves.view(np.uint8).reshape(blocks, 2)
    return data.reshape(ROWS, -1)


def main() -> None:
    rng = np.random.default_rng(1234)
    writer = GGUFWriter(str(OUTPUT), "klartraum-test")
    for qtype in (Q.Q8_0, Q.Q3_K, Q.Q4_K, Q.Q5_K, Q.Q6_K):
        raw = random_blocks(rng, qtype)
        writer.add_tensor(qtype.name, raw, raw_dtype=qtype)
        writer.add_tensor(qtype.name + ".expected", dequantize(raw, qtype).astype(np.float32))
    values = rng.standard_normal((ROWS, COLUMNS)).astype(np.float32)
    writer.add_tensor("F16", values.astype(np.float16))
    writer.add_tensor("F16.expected", values.astype(np.float16).astype(np.float32))
    bf16 = (values.view(np.uint32) >> 16).astype(np.uint16)
    writer.add_tensor("BF16", bf16.view(np.uint8), raw_dtype=Q.BF16)
    writer.add_tensor("BF16.expected", (bf16.astype(np.uint32) << 16).view(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(OUTPUT)


if __name__ == "__main__":
    main()
