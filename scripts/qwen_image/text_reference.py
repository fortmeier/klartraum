# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Reference prompt embeddings of Qwen-Image's text encoder (Qwen2.5-VL-7B's language model) from a GGUF file.

Runs transformers' Qwen2_5_VLDecoderLayer in float32 one layer at a time on weights dequantized from the GGUF (the
whole model in float32 would need ~30 GB), for the prompt in Qwen-Image's template. Writes, for the Klartraum test
(tests/test_qwen_text_encoder.cpp), into build/models/qwen-image/reference/:

* ``text_prompt.txt``: the prompt;
* ``text_tokens_i32.bin``: the template's token ids (transformers' tokenizer of Qwen/Qwen-Image);
* ``text_layer0_f32.bin``: the hidden states after the first layer, [tokens][3584];
* ``text_hidden_f32.bin``: the final normed hidden states (diffusers' hidden_states[-1]), [tokens][3584]; the pipeline
  drops the first 34 (the template's system part).

    uv run --project scripts/qwen_image python scripts/qwen_image/text_reference.py --prompt "a bonsai tree"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gguf
import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as qwen

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "build/models/qwen-image"
TEMPLATE = ("<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
            "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
            "<|im_start|>assistant\n")

# GGUF tensor names of a layer -> transformers parameter names.
LAYER_TENSORS = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_q.bias": "self_attn.q_proj.bias",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_k.bias": "self_attn.k_proj.bias",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_v.bias": "self_attn.v_proj.bias",
    "attn_output.weight": "self_attn.o_proj.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}


def dequantize(tensor) -> torch.Tensor:
    """A GGUF tensor as float32 in PyTorch's [rows][columns] order."""
    values = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
    shape = [int(v) for v in reversed(tensor.shape)]
    return torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32).reshape(shape))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", default="a bonsai tree in a clay pot, watercolor painting")
    parser.add_argument("--gguf", type=Path, default=MODELS / "text/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf")
    parser.add_argument("--out", type=Path, default=MODELS / "reference")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODELS / "original/tokenizer")
    tokens = tokenizer(TEMPLATE.format(args.prompt))["input_ids"]
    print(f"{len(tokens)} tokens")
    np.asarray(tokens, dtype=np.int32).tofile(args.out / "text_tokens_i32.bin")
    (args.out / "text_prompt.txt").write_text(args.prompt)

    reader = gguf.GGUFReader(args.gguf)
    tensors = {t.name: t for t in reader.tensors}
    config = AutoConfig.from_pretrained(MODELS / "original/text_encoder").text_config
    config._attn_implementation = "eager"

    embedding = tensors["token_embd.weight"]
    rows = gguf.quants.dequantize(embedding.data, embedding.tensor_type).reshape(-1, config.hidden_size)
    hidden = torch.from_numpy(np.ascontiguousarray(rows[tokens], dtype=np.float32)).unsqueeze(0)
    count = len(tokens)
    # Text-only M-RoPE: all three position components are the token index.
    positions = torch.arange(count).view(1, 1, -1).expand(3, 1, -1)
    rotary = qwen.Qwen2_5_VLRotaryEmbedding(config)
    position_embeddings = rotary(hidden, positions)
    mask = torch.full((count, count), float("-inf")).triu(1).view(1, 1, count, count)

    with torch.no_grad():
        for index in range(config.num_hidden_layers):
            layer = qwen.Qwen2_5_VLDecoderLayer(config, index).eval()
            state = {target: dequantize(tensors[f"blk.{index}.{source}"]) for source, target in LAYER_TENSORS.items()}
            layer.load_state_dict(state, strict=True)
            hidden = layer(hidden, attention_mask=mask, position_embeddings=position_embeddings)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
            if index == 0:
                hidden[0].numpy().tofile(args.out / "text_layer0_f32.bin")
            print(f"  layer {index}: |h| max {hidden.abs().max():.1f}", flush=True)
        norm = qwen.Qwen2_5_VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        norm.weight.data.copy_(dequantize(tensors["output_norm.weight"]))
        hidden = norm(hidden)
    hidden[0].numpy().tofile(args.out / "text_hidden_f32.bin")
    print(f"final: range [{hidden.min():.3f}, {hidden.max():.3f}]")


if __name__ == "__main__":
    main()
