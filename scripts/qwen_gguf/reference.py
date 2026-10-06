# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""NumPy reference forward pass of a `qwen35` GGUF model (Qwen3.5 / Qwen3.6).

Reads the GGUF tensors directly (dequantized with gguf-py), so it checks
Klartraum against the file it actually runs. All tokens of the prompt are
processed layer by layer (each weight is decoded once), in float32; the
linear-attention recurrence runs token by token.

Usage (from this directory):
    uv run python reference.py MODEL.gguf --tokens 1 17 255 [--check REFERENCE.gguf]
    uv run python reference.py MODEL.gguf --tokens ... --save logits.npy

With --check, the logits are compared to the `logits` tensor of a reference
file written by make_tiny_model.py. For the 27B model expect several minutes
and ~6 GB of memory (the output projection is decoded in slices).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize


class Model:
    def __init__(self, path: str):
        self.reader = GGUFReader(path)
        self.tensors = {t.name: t for t in self.reader.tensors}
        fields = self.reader.fields

        def value(key):
            field = fields[key]
            return field.parts[field.data[0]][0]

        arch = "qwen35"
        self.layers = int(value(f"{arch}.block_count"))
        self.hidden = int(value(f"{arch}.embedding_length"))
        self.heads = int(value(f"{arch}.attention.head_count"))
        self.kv_heads = int(value(f"{arch}.attention.head_count_kv"))
        self.head_dim = int(value(f"{arch}.attention.key_length"))
        self.eps = float(value(f"{arch}.attention.layer_norm_rms_epsilon"))
        self.rope_base = float(value(f"{arch}.rope.freq_base"))
        self.rope_dims = int(value(f"{arch}.rope.dimension_count"))
        self.key_heads = int(value(f"{arch}.ssm.group_count"))
        self.value_heads = int(value(f"{arch}.ssm.time_step_rank"))
        self.key_dim = int(value(f"{arch}.ssm.state_size"))
        self.value_dim = int(value(f"{arch}.ssm.inner_size")) // self.value_heads
        interval = int(value(f"{arch}.full_attention_interval"))
        self.recurrent = [(i + 1) % interval != 0 for i in range(self.layers)]

    def has(self, name: str) -> bool:
        return name in self.tensors

    def weight(self, name: str, rows: slice | None = None) -> np.ndarray:
        tensor = self.tensors[name]
        data = tensor.data if rows is None else tensor.data[rows]
        return dequantize(data, tensor.tensor_type).astype(np.float32)

    def linear(self, name: str, x: np.ndarray) -> np.ndarray:
        """x [T, in] times the transposed [out, in] weight."""
        tensor = self.tensors[name]
        rows = tensor.data.shape[0]
        if rows <= 65536:
            return x @ self.weight(name).T
        return np.concatenate(
            [x @ self.weight(name, slice(s, min(s + 32768, rows))).T for s in range(0, rows, 32768)], axis=-1
        )


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def rope(x: np.ndarray, positions: np.ndarray, dims: int, base: float) -> np.ndarray:
    """NEOX rotation of the first `dims` values of the last axis; x is [T, heads, D]."""
    half = dims // 2
    frequencies = base ** (-2.0 * np.arange(half) / dims)
    angles = positions[:, None] * frequencies[None, :]
    cos, sin = np.cos(angles)[:, None, :], np.sin(angles)[:, None, :]
    out = x.copy()
    first, second = x[..., :half], x[..., half:dims]
    out[..., :half] = first * cos - second * sin
    out[..., half:dims] = second * cos + first * sin
    return out


def full_attention(model: Model, block: str, x: np.ndarray, positions: np.ndarray) -> np.ndarray:
    tokens = x.shape[0]
    D, H, KV = model.head_dim, model.heads, model.kv_heads
    query_gate = model.linear(block + "attn_q.weight", x).reshape(tokens, H, 2, D)
    query, gate = query_gate[:, :, 0], query_gate[:, :, 1]
    key = model.linear(block + "attn_k.weight", x).reshape(tokens, KV, D)
    value = model.linear(block + "attn_v.weight", x).reshape(tokens, KV, D)
    query = rope(rms_norm(query, model.weight(block + "attn_q_norm.weight"), model.eps), positions, model.rope_dims,
                 model.rope_base)
    key = rope(rms_norm(key, model.weight(block + "attn_k_norm.weight"), model.eps), positions, model.rope_dims,
               model.rope_base)
    out = np.zeros((tokens, H, D), dtype=np.float32)
    group = H // KV
    for h in range(H):
        scores = query[:, h] @ key[:, h // group].T / np.sqrt(D)
        scores = np.where(np.tril(np.ones((tokens, tokens), dtype=bool)), scores, -np.inf)
        scores = np.exp(scores - scores.max(-1, keepdims=True))
        out[:, h] = (scores / scores.sum(-1, keepdims=True)) @ value[:, h // group]
    out = out * (1.0 / (1.0 + np.exp(-gate)))
    return model.linear(block + "attn_output.weight", out.reshape(tokens, H * D))


def linear_attention(model: Model, block: str, x: np.ndarray) -> np.ndarray:
    tokens = x.shape[0]
    Hk, Hv, Dk, Dv = model.key_heads, model.value_heads, model.key_dim, model.value_dim
    mixed = model.linear(block + "attn_qkv.weight", x)
    z = model.linear(block + "attn_gate.weight", x).reshape(tokens, Hv, Dv)
    alpha = model.linear(block + "ssm_alpha.weight", x)
    beta = 1.0 / (1.0 + np.exp(-model.linear(block + "ssm_beta.weight", x)))
    a = model.weight(block + "ssm_a")
    dt_bias = model.weight(block + "ssm_dt.bias")
    conv_weight = model.weight(block + "ssm_conv1d.weight")  # [channels, 4]
    padded = np.concatenate([np.zeros((3, mixed.shape[1]), dtype=np.float32), mixed])
    conv = np.stack([np.sum(padded[t : t + 4].T * conv_weight, axis=1) for t in range(tokens)])
    conv = silu(conv)
    q = conv[:, : Hk * Dk].reshape(tokens, Hk, Dk)
    k = conv[:, Hk * Dk : 2 * Hk * Dk].reshape(tokens, Hk, Dk)
    v = conv[:, 2 * Hk * Dk :].reshape(tokens, Hv, Dv)
    q = q / np.sqrt(np.sum(q * q, -1, keepdims=True) + 1e-6) / np.sqrt(Dk)
    k = k / np.sqrt(np.sum(k * k, -1, keepdims=True) + 1e-6)
    softplus = np.logaddexp(0.0, alpha + dt_bias)
    decay = np.exp(softplus * a)  # [T, Hv]
    state = np.zeros((Hv, Dk, Dv), dtype=np.float32)
    out = np.zeros((tokens, Hv, Dv), dtype=np.float32)
    for t in range(tokens):
        for h in range(Hv):
            kh = h % Hk
            S = state[h] * decay[t, h]
            delta = beta[t, h] * (v[t, h] - k[t, kh] @ S)
            S = S + np.outer(k[t, kh], delta)
            state[h] = S
            out[t, h] = q[t, kh] @ S
    out = rms_norm(out, model.weight(block + "ssm_norm.weight"), model.eps) * silu(z)
    return model.linear(block + "ssm_out.weight", out.reshape(tokens, Hv * Dv))


def forward(model: Model, tokens: list[int]) -> np.ndarray:
    embeddings = model.tensors["token_embd.weight"]
    x = np.stack([dequantize(embeddings.data[t : t + 1], embeddings.tensor_type)[0] for t in tokens]).astype(np.float32)
    positions = np.arange(len(tokens), dtype=np.float64)
    for layer in range(model.layers):
        block = f"blk.{layer}."
        normed = rms_norm(x, model.weight(block + "attn_norm.weight"), model.eps)
        if model.recurrent[layer]:
            x = x + linear_attention(model, block, normed)
        else:
            x = x + full_attention(model, block, normed, positions)
        normed = rms_norm(x, model.weight(block + "post_attention_norm.weight"), model.eps)
        hidden = silu(model.linear(block + "ffn_gate.weight", normed)) * model.linear(block + "ffn_up.weight", normed)
        x = x + model.linear(block + "ffn_down.weight", hidden)
        print(f"layer {layer} done", file=sys.stderr, flush=True)
    x = rms_norm(x, model.weight("output_norm.weight"), model.eps)
    return model.linear("output.weight", x)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("--tokens", type=int, nargs="+", required=True)
    parser.add_argument("--check", help="reference GGUF with a `logits` tensor")
    parser.add_argument("--save", help="write the logits as .npy")
    args = parser.parse_args()

    logits = forward(Model(args.model), args.tokens)
    print("argmax per position:", logits.argmax(-1).tolist())
    if args.save:
        np.save(args.save, logits)
    if args.check:
        reference = {t.name: t for t in GGUFReader(args.check).tensors}["logits"].data
        error = np.abs(logits - reference).max()
        print(f"max |logits - reference| = {error:.3e} (logit scale {np.abs(reference).max():.2f})")
        return 0 if error < 1e-3 * max(1.0, np.abs(reference).max()) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
