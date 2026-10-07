# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Writes tests/data/gguf/qwen35_tokenizer_cases.gguf for tests/test_gguf_tokenizer.cpp.

Encodes a list of test strings with the official Qwen3.6 tokenizer
(Hugging Face `tokenizers`, Qwen/Qwen3.6-27B tokenizer.json) and stores the
strings (`texts`), the concatenated token ids (`ids`, I32) and the id count
per string (`counts`, I32). Special tokens in the strings are not parsed.

Usage (from this directory): uv run --group reference python make_tokenizer_cases.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from gguf import GGUFWriter
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

OUTPUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "gguf" / "qwen35_tokenizer_cases.gguf"

TEXTS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "I'm sure they'll say it's fine, but WE'RE not; YOU'VE seen it.",
    "Numbers: 1234567 and 3.14159, also 2026-10-06.",
    "   leading spaces and trailing   ",
    "tabs\tand\nnew\n\nlines\r\n  indented\n\n\n",
    "def f(x):\n    return x**2  # square\n",
    "Grüße aus Köln! Ça va? naïve café",
    "日本語のテキストと中文混合。",
    "Emoji: 🙂🚀 and symbols ©®™ ∑∫√",
    "Русский текст, ελληνικά, עברית, العربية",
    "हिन्दी में लिखा गया वाक्य",
    "<|im_start|> is written as text here",
    "a  b   c    d",
    "!!!??? ... --- ***",
    "\n",
    " ",
    "x",
]


def main() -> None:
    path = hf_hub_download("Qwen/Qwen3.6-27B", "tokenizer.json")
    tokenizer = Tokenizer.from_file(path)
    tokenizer.encode_special_tokens = True  # special-token names in the text are plain text
    ids, counts = [], []
    for text in TEXTS:
        encoded = tokenizer.encode(text, add_special_tokens=False).ids
        ids += encoded
        counts.append(len(encoded))
    writer = GGUFWriter(str(OUTPUT), "tokenizer-cases")
    writer.add_array("texts", TEXTS)
    writer.add_tensor("ids", np.array(ids, dtype=np.int32))
    writer.add_tensor("counts", np.array(counts, dtype=np.int32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(OUTPUT, len(ids), "ids")


if __name__ == "__main__":
    main()
