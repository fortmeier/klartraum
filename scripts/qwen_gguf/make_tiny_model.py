# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Writes a tiny random Qwen3.5 model as GGUF plus reference logits for tests/test_gguf_qwen35.cpp.

The model is a Hugging Face transformers `Qwen3_5ForCausalLM` with the
Qwen3.6-27B layout scaled down (three Gated-DeltaNet layers, one gated
full-attention layer; more linear value heads than key heads, partial
rotary embeddings). Its weights are randomized so every parameter matters,
then stored in the tensor layout of llama.cpp's `qwen35` GGUF files:

  - RMSNorm weights are stored as (1 + w), except the gated norm of the
    linear attention, which is stored as is;
  - `ssm_a` holds -exp(A_log);
  - linear-attention value heads are reordered from grouped
    (key head major) to tiled (value-per-key major) order, so value head j
    uses key head j % key_heads;
  - the conv1d kernel is stored as [channels][4].

Outputs (tests/data/gguf/):
  tiny_qwen35_f32.gguf      weights in F32
  tiny_qwen35_q8_0.gguf     2D projection weights in Q8_0 (norms etc. F32)
  tiny_qwen35_reference.gguf  `tokens` (I32) and `logits` (F32, [tokens][vocab])
                              from the transformers forward pass in float64

Usage (from this directory): uv run --group reference python make_tiny_model.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType, GGUFWriter
from gguf.quants import quantize
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

OUTPUT = Path(__file__).resolve().parents[2] / "tests" / "data" / "gguf"
TOKENS = [1, 17, 255, 3, 3, 120, 299, 42, 7]

CONFIG = dict(
    vocab_size=300,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_conv_kernel_dim=4,
    layer_types=["linear_attention"] * 3 + ["full_attention"],
    full_attention_interval=4,
    rms_norm_eps=1e-6,
    rope_parameters={
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.25,
        "mrope_section": [2, 1, 1],
        "mrope_interleaved": True,
    },
    tie_word_embeddings=False,
)


def randomize(model: torch.nn.Module, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)

    def normal(shape, scale):
        return torch.randn(shape, generator=generator, dtype=torch.float64) * scale

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("linear_attn.norm.weight"):
                value = 1.0 + normal(parameter.shape, 0.2)
            elif name.endswith("norm.weight") or name.endswith("layernorm.weight"):
                value = normal(parameter.shape, 0.2)  # applied as (1 + w)
            elif name.endswith("A_log"):
                value = torch.rand(parameter.shape, generator=generator, dtype=torch.float64) * 2.0 - 1.0
            elif name.endswith("dt_bias"):
                value = normal(parameter.shape, 0.5)
            elif name.endswith("embed_tokens.weight"):
                value = normal(parameter.shape, 1.0)
            else:
                fan_in = parameter.shape[-1]
                value = normal(parameter.shape, 1.0 / np.sqrt(fan_in))
            parameter.copy_(value.to(parameter.dtype))


def tiled_heads(tensor: np.ndarray, axis: int, key_heads: int, per_key: int, width: int) -> np.ndarray:
    """Reorders `key_heads * per_key` blocks of `width` along `axis` from grouped to tiled order."""
    moved = np.moveaxis(tensor, axis, 0)
    shape = moved.shape
    grouped = moved.reshape(key_heads, per_key, width, *shape[1:])
    tiled = np.swapaxes(grouped, 0, 1).reshape(shape)
    return np.ascontiguousarray(np.moveaxis(tiled, 0, axis))


def gguf_tensors(model: Qwen3_5ForCausalLM, config: Qwen3_5TextConfig) -> list[tuple[str, np.ndarray]]:
    state = {name: value.detach().to(torch.float64).numpy() for name, value in model.state_dict().items()}
    key_heads = config.linear_num_key_heads
    per_key = config.linear_num_value_heads // key_heads
    key_dim = config.linear_key_head_dim
    value_dim = config.linear_value_head_dim
    qk_channels = 2 * key_heads * key_dim

    out = [
        ("token_embd.weight", state["model.embed_tokens.weight"]),
        ("output_norm.weight", 1.0 + state["model.norm.weight"]),
        ("output.weight", state["lm_head.weight"]),
    ]
    for layer, kind in enumerate(config.layer_types):
        prefix = f"model.layers.{layer}."
        block = f"blk.{layer}."
        out += [
            (block + "attn_norm.weight", 1.0 + state[prefix + "input_layernorm.weight"]),
            (block + "post_attention_norm.weight", 1.0 + state[prefix + "post_attention_layernorm.weight"]),
            (block + "ffn_gate.weight", state[prefix + "mlp.gate_proj.weight"]),
            (block + "ffn_up.weight", state[prefix + "mlp.up_proj.weight"]),
            (block + "ffn_down.weight", state[prefix + "mlp.down_proj.weight"]),
        ]
        if kind == "linear_attention":
            p = prefix + "linear_attn."
            qkv = state[p + "in_proj_qkv.weight"]
            qkv = np.concatenate([qkv[:qk_channels], tiled_heads(qkv[qk_channels:], 0, key_heads, per_key, value_dim)])
            conv = state[p + "conv1d.weight"][:, 0, :]
            conv = np.concatenate([conv[:qk_channels], tiled_heads(conv[qk_channels:], 0, key_heads, per_key, value_dim)])
            out += [
                (block + "attn_qkv.weight", qkv),
                (block + "attn_gate.weight", tiled_heads(state[p + "in_proj_z.weight"], 0, key_heads, per_key, value_dim)),
                (block + "ssm_alpha.weight", tiled_heads(state[p + "in_proj_a.weight"], 0, key_heads, per_key, 1)),
                (block + "ssm_beta.weight", tiled_heads(state[p + "in_proj_b.weight"], 0, key_heads, per_key, 1)),
                (block + "ssm_a", tiled_heads(-np.exp(state[p + "A_log"]), 0, key_heads, per_key, 1)),
                (block + "ssm_dt.bias", tiled_heads(state[p + "dt_bias"], 0, key_heads, per_key, 1)),
                (block + "ssm_conv1d.weight", conv),
                (block + "ssm_norm.weight", state[p + "norm.weight"]),
                (block + "ssm_out.weight", tiled_heads(state[p + "out_proj.weight"], 1, key_heads, per_key, value_dim)),
            ]
        else:
            p = prefix + "self_attn."
            out += [
                (block + "attn_q.weight", state[p + "q_proj.weight"]),
                (block + "attn_k.weight", state[p + "k_proj.weight"]),
                (block + "attn_v.weight", state[p + "v_proj.weight"]),
                (block + "attn_output.weight", state[p + "o_proj.weight"]),
                (block + "attn_q_norm.weight", 1.0 + state[p + "q_norm.weight"]),
                (block + "attn_k_norm.weight", 1.0 + state[p + "k_norm.weight"]),
            ]
    return out


def write_model(path: Path, config: Qwen3_5TextConfig, tensors, quant: GGMLQuantizationType | None) -> None:
    writer = GGUFWriter(str(path), "qwen35")
    arch = "qwen35"
    writer.add_name("tiny-qwen35")
    writer.add_uint32(f"{arch}.block_count", config.num_hidden_layers)
    writer.add_uint32(f"{arch}.context_length", 4096)
    writer.add_uint32(f"{arch}.embedding_length", config.hidden_size)
    writer.add_uint32(f"{arch}.feed_forward_length", config.intermediate_size)
    writer.add_uint32(f"{arch}.attention.head_count", config.num_attention_heads)
    writer.add_uint32(f"{arch}.attention.head_count_kv", config.num_key_value_heads)
    writer.add_uint32(f"{arch}.attention.key_length", config.head_dim)
    writer.add_uint32(f"{arch}.attention.value_length", config.head_dim)
    writer.add_float32(f"{arch}.attention.layer_norm_rms_epsilon", config.rms_norm_eps)
    writer.add_float32(f"{arch}.rope.freq_base", config.rope_parameters["rope_theta"])
    writer.add_uint32(f"{arch}.rope.dimension_count", int(config.head_dim * config.rope_parameters["partial_rotary_factor"]))
    writer.add_array(f"{arch}.rope.dimension_sections", config.rope_parameters["mrope_section"] + [0])
    writer.add_uint32(f"{arch}.ssm.conv_kernel", config.linear_conv_kernel_dim)
    writer.add_uint32(f"{arch}.ssm.state_size", config.linear_key_head_dim)
    writer.add_uint32(f"{arch}.ssm.group_count", config.linear_num_key_heads)
    writer.add_uint32(f"{arch}.ssm.time_step_rank", config.linear_num_value_heads)
    writer.add_uint32(f"{arch}.ssm.inner_size", config.linear_value_head_dim * config.linear_num_value_heads)
    writer.add_uint32(f"{arch}.full_attention_interval", config.full_attention_interval)
    for name, value in tensors:
        value = value.astype(np.float32)
        if quant is not None and value.ndim == 2 and value.shape[1] % 32 == 0 and "ssm_conv1d" not in name:
            writer.add_tensor(name, quantize(value, quant), raw_dtype=quant)
        else:
            writer.add_tensor(name, value)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main() -> None:
    torch.manual_seed(0)
    config = Qwen3_5TextConfig(**CONFIG)
    model = Qwen3_5ForCausalLM(config).to(torch.float64).eval()
    randomize(model, seed=7)

    with torch.no_grad():
        logits = model(torch.tensor([TOKENS]), use_cache=False).logits[0].numpy()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    tensors = gguf_tensors(model, config)
    write_model(OUTPUT / "tiny_qwen35_f32.gguf", config, tensors, None)
    write_model(OUTPUT / "tiny_qwen35_q8_0.gguf", config, tensors, GGMLQuantizationType.Q8_0)

    writer = GGUFWriter(str(OUTPUT / "tiny_qwen35_reference.gguf"), "reference")
    writer.add_tensor("tokens", np.array(TOKENS, dtype=np.int32))
    writer.add_tensor("logits", logits.astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("logits", logits.shape, "argmax", logits.argmax(-1))


if __name__ == "__main__":
    main()
