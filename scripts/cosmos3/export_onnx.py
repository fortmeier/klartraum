# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Export Cosmos3-Edge image-to-video as fixed-shape float32 ONNX graphs for Klartraum.

The diffusers transformer packs a causal text ("understanding") stream and a
video ("generation") stream into one sequence. The text stream never attends to
the video tokens, so its per-layer keys and values are independent of the
denoising step. The export therefore splits the transformer in two graphs:

* ``text_kv.onnx`` runs the text tower once for both guidance prompts and
  returns every layer's generation-side keys and values.
* ``denoiser.onnx`` runs the video tower for one scheduler step, for both
  guidance branches as a batch of two.

Both prompts are right-padded to one fixed text length. Padding is harmless in
the causal text tower and is excluded from the video attention by an additive
key bias. Rotary tables are inputs, computed on the host, because the video
positions depend on each prompt's length.

Stages run as separate invocations so that only one large model is resident:

    uv run python export_onnx.py prepare      # tokens, positions, latents, schedule
    uv run python export_onnx.py reference    # diffusers transformer, step 0
    uv run python export_onnx.py text         # text_kv.onnx
    uv run python export_onnx.py denoiser     # denoiser.onnx
    uv run python export_onnx.py vae          # vae_encoder.onnx / vae_decoder.onnx
    uv run python export_onnx.py pipeline     # float32 denoising + decode reference

``postprocess`` re-applies the graph clean-up to already exported graphs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLWan, Cosmos3OmniPipeline, Cosmos3OmniTransformer
from diffusers.models.transformers.transformer_cosmos3 import Cosmos3VLTextRotaryEmbedding
from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import (
    _preprocess_conditioning_image,
    get_3d_mrope_ids_text_tokens,
    get_3d_mrope_ids_vae_tokens,
)
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from huggingface_hub import hf_hub_download
from onnx import helper, numpy_helper, shape_inference
from onnx.external_data_helper import load_external_data_for_model
from onnx.reference import ReferenceEvaluator
from PIL import Image

from disclaimer import print_cosmos3_notice
from run_reference import DEFAULT_IMAGE, DEFAULT_PROMPT, MODEL_ID, structured_prompt


REPO_ROOT = Path(__file__).resolve().parents[2]
NEG_INF_BIAS = -1.0e9


# ---------------------------------------------------------------------------
# Configuration and fixture I/O
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["prepare", "reference", "text", "denoiser", "vae", "pipeline", "postprocess"])
    parser.add_argument("--size", type=int, default=256, help="square output size, multiple of 32")
    parser.add_argument("--num-frames", type=int, default=33)
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--flow-shift", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--prompt-file", type=Path, default=None, help="full JSON caption, used unchanged")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--onnx-dir", type=Path, default=None)
    parser.add_argument("--skip-ort", action="store_true", help="skip the ONNX Runtime comparison")
    parser.add_argument(
        "--fixtures-only",
        action="store_true",
        help="text/denoiser: write the PyTorch fixtures for the existing graphs without exporting them",
    )
    parser.add_argument(
        "--decoder-tile",
        type=int,
        default=0,
        help="prepare: decode in square latent tiles of this size (0: whole clip), e.g. 16 to reuse a 256 decoder",
    )
    parser.add_argument("--decoder-stride", type=int, default=8, help="prepare: latent stride between decoder tiles")
    parser.add_argument("--video-dir", type=Path, default=REPO_ROOT / "build" / "TestingOutput" / "cosmos3")
    args = parser.parse_args()
    if args.onnx_dir is None:
        args.onnx_dir = REPO_ROOT / "data" / "onnx" / f"cosmos3_{args.size}"
    return args


def save(directory: Path, name: str, array: np.ndarray) -> None:
    """Write a raw little-endian tensor fixture plus its shape in fixtures.json."""
    array = np.ascontiguousarray(array)
    directory.mkdir(parents=True, exist_ok=True)
    array.tofile(directory / name)
    index_path = directory / "fixtures.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    index[name] = {"dtype": str(array.dtype), "shape": list(array.shape)}
    index_path.write_text(json.dumps(index, indent=1, sort_keys=True))


def load(directory: Path, name: str) -> np.ndarray:
    entry = json.loads((directory / "fixtures.json").read_text())[name]
    return np.fromfile(directory / name, dtype=entry["dtype"]).reshape(entry["shape"])


def write_text_config(directory: Path, config: dict) -> None:
    """config.txt: one "key value..." line per numeric entry, read by the C++ example."""
    lines = []
    for key, value in config.items():
        values = value if isinstance(value, list) else [value]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            lines.append(" ".join([key] + [repr(v) for v in values]))
    (directory / "config.txt").write_text("\n".join(lines) + "\n")


def load_config(directory: Path) -> dict:
    return json.loads((directory / "config.json").read_text())


def report(name: str, actual: np.ndarray, expected: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    error = float(np.max(np.abs(actual - expected)))
    scale = float(np.max(np.abs(expected)))
    print(f"  {name}: max |error| {error:.3e} (reference max |value| {scale:.3e})", flush=True)
    if not np.isfinite(error):
        raise RuntimeError(f"{name} is not finite")
    return error


# ---------------------------------------------------------------------------
# Shared math (mirrors diffusers' transformer_cosmos3)
# ---------------------------------------------------------------------------


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Nemotron RMSNorm written with ops Klartraum provides (no Pow, no Reciprocal)."""
    variance = (x * x).mean(-1, keepdim=True)
    return x / torch.sqrt(variance + eps) * weight


def rope_tables(position_ids: torch.Tensor, rotary_emb) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sign-folded sin for ``x * cos + concat(x2, x1) * sin_rot``.

    ``rotate_half(x) * sin == concat(x2, x1) * concat(-sin1, sin2)``, so folding
    the sign into the table removes the negation from the exported graph.
    """
    cos, sin = rotary_emb(position_ids=position_ids.unsqueeze(1), device="cpu", dtype=torch.float32)
    cos, sin = cos.squeeze(0), sin.squeeze(0)
    half = sin.shape[-1] // 2
    sin_rot = torch.cat([-sin[..., :half], sin[..., half:]], dim=-1)
    return cos, sin_rot


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin_rot: torch.Tensor) -> torch.Tensor:
    """x: [B, T, H, D]; cos/sin_rot: [B, T, 1, D]."""
    half = x.shape[-1] // 2
    swapped = torch.cat([x[..., half:], x[..., :half]], dim=-1)
    return x * cos + swapped * sin_rot


def grouped_attention(q: torch.Tensor, k_t: torch.Tensor, v: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Grouped-query attention without repeating keys and values.

    q: [B, T, H, D] with H = groups * KV; k_t: [B, KV, D, S]; v: [B, KV, S, D];
    bias: broadcastable to [B, KV, groups*T, S]. Query head h uses key/value head
    h // groups, so the groups of one KV head become consecutive query rows.
    """
    b, t, h, d = q.shape
    kv = k_t.shape[1]
    q = q.permute(0, 2, 1, 3).reshape(b, kv, (h // kv) * t, d) * (1.0 / math.sqrt(d))
    scores = torch.matmul(q, k_t) + bias
    context = torch.matmul(torch.softmax(scores, dim=-1), v)
    return context.reshape(b, h, t, d).permute(0, 2, 1, 3).reshape(b, t, h * d)


def mlp_relu2(x: torch.Tensor, mlp) -> torch.Tensor:
    hidden = torch.relu(mlp.up_proj(x))
    return mlp.down_proj(hidden * hidden)


# ---------------------------------------------------------------------------
# Export wrappers
# ---------------------------------------------------------------------------


class TextKV(torch.nn.Module):
    """Text tower: token ids -> per-layer generation-side keys and values."""

    def __init__(self, transformer: Cosmos3OmniTransformer, text_length: int):
        super().__init__()
        self.embed_tokens = transformer.embed_tokens
        self.layers = transformer.layers
        self.eps = transformer.config.rms_norm_eps
        self.heads = transformer.config.num_attention_heads
        self.kv_heads = transformer.config.num_key_value_heads
        self.head_dim = transformer.config.head_dim
        groups = self.heads // self.kv_heads
        causal = torch.triu(torch.full((text_length, text_length), NEG_INF_BIAS), diagonal=1)
        self.register_buffer("causal_bias", causal.repeat(groups, 1), persistent=False)

    def forward(self, input_ids: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor):
        batch, length = input_ids.shape
        hidden = self.embed_tokens(input_ids)
        cos = rope_cos.unsqueeze(2)
        sin_rot = rope_sin.unsqueeze(2)
        outputs = []
        for index, layer in enumerate(self.layers):
            attn = layer.self_attn
            x = rms_norm(hidden, layer.input_layernorm.weight, self.eps)
            k = attn.to_k(x).reshape(batch, length, self.kv_heads, self.head_dim)
            v = attn.to_v(x).reshape(batch, length, self.kv_heads, self.head_dim)
            k_gen = apply_rope(rms_norm(k, attn.k_norm_und_for_gen.weight, self.eps), cos, sin_rot)
            v_t = v.permute(0, 2, 1, 3)
            outputs += [k_gen.permute(0, 2, 3, 1), v_t]
            if index == len(self.layers) - 1:
                break  # The last layer's text output is never consumed.
            # qk_norm_for_text is disabled for Cosmos3-Edge: text q/k are not normalized.
            q = apply_rope(attn.to_q(x).reshape(batch, length, self.heads, self.head_dim), cos, sin_rot)
            k = apply_rope(k, cos, sin_rot)
            hidden = hidden + attn.to_out(grouped_attention(q, k.permute(0, 2, 3, 1), v_t, self.causal_bias))
            hidden = hidden + mlp_relu2(rms_norm(hidden, layer.post_attention_layernorm.weight, self.eps), layer.mlp)
        return tuple(outputs)


class Denoiser(torch.nn.Module):
    """Video tower for one step: patch tokens -> velocity, for both guidance branches."""

    def __init__(self, transformer: Cosmos3OmniTransformer):
        super().__init__()
        self.layers = transformer.layers
        self.norm_moe_gen = transformer.norm_moe_gen
        self.proj_in = transformer.proj_in
        self.proj_out = transformer.proj_out
        self.time_proj = transformer.time_proj
        self.time_embedder = transformer.time_embedder
        self.timestep_scale = transformer.config.timestep_scale
        self.eps = transformer.config.rms_norm_eps
        self.heads = transformer.config.num_attention_heads
        self.kv_heads = transformer.config.num_key_value_heads
        self.head_dim = transformer.config.head_dim

    def forward(self, tokens, timestep, noisy_mask, rope_cos, rope_sin, text_bias, *text_kv):
        hidden = self.proj_in(tokens)
        embedding = self.time_embedder(self.time_proj(timestep * self.timestep_scale))
        hidden = hidden + embedding * noisy_mask
        hidden = torch.cat([hidden.unsqueeze(0), hidden.unsqueeze(0)], dim=0)
        batch, length = hidden.shape[:2]
        cos = rope_cos.unsqueeze(2)
        sin_rot = rope_sin.unsqueeze(2)
        for index, layer in enumerate(self.layers):
            attn = layer.self_attn
            x = rms_norm(hidden, layer.input_layernorm_moe_gen.weight, self.eps)
            q = attn.add_q_proj(x).reshape(batch, length, self.heads, self.head_dim)
            k = attn.add_k_proj(x).reshape(batch, length, self.kv_heads, self.head_dim)
            v = attn.add_v_proj(x).reshape(batch, length, self.kv_heads, self.head_dim)
            q = apply_rope(rms_norm(q, attn.norm_added_q.weight, self.eps), cos, sin_rot)
            k = apply_rope(rms_norm(k, attn.norm_added_k.weight, self.eps), cos, sin_rot)
            keys = torch.cat([text_kv[2 * index], k.permute(0, 2, 3, 1)], dim=3)
            values = torch.cat([text_kv[2 * index + 1], v.permute(0, 2, 1, 3)], dim=2)
            hidden = hidden + attn.to_add_out(grouped_attention(q, keys, values, text_bias))
            hidden = hidden + mlp_relu2(
                rms_norm(hidden, layer.post_attention_layernorm_moe_gen.weight, self.eps), layer.mlp_moe_gen
            )
        return self.proj_out(rms_norm(hidden, self.norm_moe_gen.weight, self.eps))


# ---------------------------------------------------------------------------
# Host-side packing (ported to C++ in the Klartraum example)
# ---------------------------------------------------------------------------


def patchify(latents: torch.Tensor, patch: int) -> torch.Tensor:
    """[C, T, H, W] -> [T*(H/p)*(W/p), p*p*C] in the transformer's token order."""
    c, t, h, w = latents.shape
    x = latents.reshape(c, t, h // patch, patch, w // patch, patch)
    return torch.einsum("cthpwq->thwpqc", x).reshape(-1, patch * patch * c)


def unpatchify(tokens: torch.Tensor, shape: tuple[int, int, int, int], patch: int) -> torch.Tensor:
    c, t, h, w = shape
    x = tokens.reshape(t, h // patch, w // patch, patch, patch, c)
    return torch.einsum("thwpqc->cthpwq", x).reshape(c, t, h, w)


def load_transformer(keep: str | None) -> Cosmos3OmniTransformer:
    """Load the transformer in float32, dropping the tower that ``keep`` does not need."""
    transformer = Cosmos3OmniTransformer.from_pretrained(MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16)
    transformer.lm_head = None
    if keep == "text":
        transformer.proj_in = transformer.proj_out = transformer.time_embedder = None
        transformer.norm_moe_gen = None
        for layer in transformer.layers:
            attn = layer.self_attn
            attn.add_q_proj = attn.add_k_proj = attn.add_v_proj = attn.to_add_out = None
            layer.mlp_moe_gen = layer.input_layernorm_moe_gen = layer.post_attention_layernorm_moe_gen = None
    elif keep == "denoiser":
        transformer.embed_tokens = None
        transformer.norm = None
        for layer in transformer.layers:
            attn = layer.self_attn
            attn.to_q = attn.to_k = attn.to_v = attn.to_out = None
            layer.mlp = layer.input_layernorm = layer.post_attention_layernorm = None
    for name in ("action_proj_in", "action_proj_out"):
        if hasattr(transformer, name):
            setattr(transformer, name, None)
    gc.collect()
    return transformer.float().eval()


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def stage_prepare(args: argparse.Namespace) -> None:
    """Tokenize, encode the conditioning frame, sample noise, and build host-side tables."""
    out = args.onnx_dir
    size, frames, fps = args.size, args.num_frames, args.fps
    pipe = Cosmos3OmniPipeline.from_pretrained(
        MODEL_ID, transformer=None, torch_dtype=torch.float32, enable_safety_checker=False
    )
    tconfig = Cosmos3OmniTransformer.load_config(MODEL_ID, subfolder="transformer")
    negative = json.dumps(json.loads(Path(hf_hub_download(MODEL_ID, "assets/negative_prompt.json")).read_text()))
    prompt = (
        json.dumps(json.loads(args.prompt_file.read_text()))
        if args.prompt_file
        else structured_prompt(args.prompt, frames, fps, size, size)
    )
    cond_ids, uncond_ids = pipe.tokenize_prompt(
        prompt, negative, num_frames=frames, height=size, width=size, fps=fps,
        add_resolution_template=False, add_duration_template=False,
    )
    lengths = [len(cond_ids), len(uncond_ids)]
    text_length = ((max(lengths) + 63) // 64) * 64
    pad = pipe.llm_special_tokens["eos_token_id"]
    input_ids = np.full((2, text_length), pad, dtype=np.int64)
    input_ids[0, : lengths[0]] = cond_ids
    input_ids[1, : lengths[1]] = uncond_ids

    # Conditioning frame: the VAE is temporally causal, so latent frame 0 only
    # depends on pixel frame 0 and a one-frame encode reproduces the pipeline.
    image = _preprocess_conditioning_image(Image.open(args.image).convert("RGB"), height=size, width=size)
    with torch.no_grad():
        clip = image.unsqueeze(2).expand(-1, -1, frames, -1, -1).contiguous()
        full = pipe._encode_video(clip)
        single = pipe._encode_video(image.unsqueeze(2).contiguous())
    report("one-frame encode vs full-clip latent 0", single[:, :, 0].numpy(), full[:, :, 0].numpy())
    latent_t = full.shape[2]
    latent_hw = size // 16
    patch = tconfig["latent_patch_size"]
    grid = latent_hw // patch
    tokens_per_frame = grid * grid
    num_tokens = latent_t * tokens_per_frame

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    noise = torch.randn(full.shape, generator=generator, dtype=torch.float32)
    mask = torch.zeros((latent_t, 1, 1))
    mask[0] = 1.0
    latents = (mask * full + (1.0 - mask) * noise)[0]

    # Rotary tables for both prompts (position ids depend on the prompt length).
    rotary = Cosmos3VLTextRotaryEmbedding(
        head_dim=tconfig["head_dim"], rope_theta=tconfig["rope_theta"], rope_axes_dim=tconfig["rope_axes_dim"]
    )
    text_cos, text_sin, vision_cos, vision_sin, bias = [], [], [], [], []
    for length in lengths:
        text_ids, next_offset = get_3d_mrope_ids_text_tokens(length, 0, use_float_positions=True)
        padded = torch.cat([text_ids, text_ids[:, -1:].expand(3, text_length - length)], dim=1)
        cos, sin_rot = rope_tables(padded, rotary)
        text_cos.append(cos)
        text_sin.append(sin_rot)
        vision_ids, _ = get_3d_mrope_ids_vae_tokens(
            grid_t=latent_t, grid_h=grid, grid_w=grid,
            temporal_offset=next_offset + tconfig["unified_3d_mrope_temporal_modality_margin"],
            reset_spatial_indices=tconfig["unified_3d_mrope_reset_spatial_ids"],
            fps=fps, base_fps=float(tconfig["base_fps"]), temporal_compression_factor=4,
        )
        cos, sin_rot = rope_tables(vision_ids, rotary)
        vision_cos.append(cos)
        vision_sin.append(sin_rot)
        row = np.zeros(text_length + num_tokens, dtype=np.float32)
        row[length:text_length] = NEG_INF_BIAS
        bias.append(row)
        save(out, f"vision_position_ids_{len(vision_cos) - 1}_f32.bin", vision_ids.numpy().astype(np.float32))
    noisy_mask = np.ones((num_tokens, 1), dtype=np.float32)
    noisy_mask[:tokens_per_frame] = 0.0

    scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=args.flow_shift, use_karras_sigmas=False
    )
    sigmas = np.linspace(1.0 - 1.0 / scheduler.config.num_train_timesteps, 0.0, args.steps + 1)[:-1]
    scheduler.set_timesteps(args.steps, sigmas=sigmas)

    config = {
        "size": size, "num_frames": frames, "fps": fps, "steps": args.steps,
        "guidance_scale": args.guidance_scale, "flow_shift": args.flow_shift, "seed": args.seed,
        "prompt": prompt if args.prompt_file else args.prompt, "text_length": text_length, "prompt_lengths": lengths,
        "latent_shape": list(latents.shape), "patch": patch, "num_tokens": num_tokens,
        "tokens_per_frame": tokens_per_frame, "num_layers": tconfig["num_hidden_layers"],
        "kv_heads": tconfig["num_key_value_heads"], "head_dim": tconfig["head_dim"],
        "rope_theta": tconfig["rope_theta"], "rope_axes_dim": tconfig["rope_axes_dim"],
        "modality_margin": tconfig["unified_3d_mrope_temporal_modality_margin"],
        "base_fps": tconfig["base_fps"],
        "vae_latents_mean": pipe.vae.config.latents_mean, "vae_latents_std": pipe.vae.config.latents_std,
        "decoder_tile": args.decoder_tile, "decoder_stride": args.decoder_stride,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(config, indent=1))
    write_text_config(out, config)
    (out / "prompt.json").write_text(prompt)
    save(out, "input_ids_i64.bin", input_ids)
    save(out, "text_rope_cos_f32.bin", torch.stack(text_cos).numpy())
    save(out, "text_rope_sin_f32.bin", torch.stack(text_sin).numpy())
    save(out, "vision_rope_cos_f32.bin", torch.stack(vision_cos).numpy())
    save(out, "vision_rope_sin_f32.bin", torch.stack(vision_sin).numpy())
    save(out, "text_bias_f32.bin", np.stack(bias).reshape(2, 1, 1, -1))
    save(out, "noisy_mask_f32.bin", noisy_mask)
    save(out, "image_f32.bin", image.unsqueeze(2).numpy())
    save(out, "condition_latent_f32.bin", single[0].numpy())
    save(out, "initial_latents_f32.bin", latents.numpy())
    save(out, "sigmas_f32.bin", scheduler.sigmas.numpy().astype(np.float32))
    save(out, "timesteps_f32.bin", scheduler.timesteps.numpy().astype(np.float32))
    print(f"prompt lengths {lengths} -> text length {text_length}, {num_tokens} video tokens", flush=True)
    print(f"wrote fixtures to {out}", flush=True)


def stage_reference(args: argparse.Namespace) -> None:
    """Run the unmodified diffusers transformer (float32, CPU) for step 0 of both branches."""
    out = args.onnx_dir
    cfg = load_config(out)
    transformer = load_transformer(keep=None)
    input_ids = torch.from_numpy(load(out, "input_ids_i64.bin"))
    latents = torch.from_numpy(load(out, "initial_latents_f32.bin"))
    timestep = float(load(out, "timesteps_f32.bin")[0])
    vision_ids = [torch.from_numpy(load(out, f"vision_position_ids_{i}_f32.bin")) for i in range(2)]
    lt, grid, per_frame = cfg["latent_shape"][1], cfg["latent_shape"][2] // cfg["patch"], cfg["tokens_per_frame"]
    velocities = []
    for branch in range(2):
        length = cfg["prompt_lengths"][branch]
        text_ids, _ = get_3d_mrope_ids_text_tokens(length, 0, use_float_positions=True)
        noisy = torch.arange(1, lt)
        started = time.perf_counter()
        with torch.no_grad():
            preds, _, _ = transformer(
                input_ids=input_ids[branch, :length],
                text_indexes=torch.arange(length),
                position_ids=torch.cat([text_ids, vision_ids[branch]], dim=1),
                und_len=length,
                sequence_length=length + cfg["num_tokens"],
                vision_tokens=[latents.unsqueeze(0)],
                vision_token_shapes=[(lt, grid, grid)],
                vision_sequence_indexes=torch.arange(length, length + cfg["num_tokens"]),
                vision_mse_loss_indexes=torch.arange(length + per_frame, length + cfg["num_tokens"]),
                vision_timesteps=torch.full(((lt - 1) * per_frame,), timestep),
                vision_noisy_frame_indexes=[noisy],
                return_dict=False,
            )
        print(f"branch {branch}: {time.perf_counter() - started:.1f}s", flush=True)
        velocities.append(preds[0][0].numpy())
    save(out, "reference_step0_velocity_f32.bin", np.stack(velocities))


def hoist_constants(model: onnx.ModelProto, threshold: int = 0) -> int:
    """Move Constant nodes into initializers, sharing storage between identical values.

    The tracer emits one Constant per use of a module buffer (the causal bias once
    per layer); initializers can be stored as external data and deduplicated, and
    get value_info like every other initializer.
    """
    by_digest: dict[bytes, str] = {}
    renamed: dict[str, str] = {}
    retained = []
    hoisted = 0
    for node in model.graph.node:
        value = next((a.t for a in node.attribute if a.name == "value" and a.HasField("t")), None)
        if node.op_type != "Constant" or value is None or np.prod(value.dims, dtype=np.int64) <= threshold:
            retained.append(node)
            continue
        array = numpy_helper.to_array(value)
        digest = hashlib.sha1(array.tobytes() + str(array.shape).encode() + str(array.dtype).encode()).digest()
        if digest in by_digest:
            renamed[node.output[0]] = by_digest[digest]
        else:
            by_digest[digest] = node.output[0]
            model.graph.initializer.append(numpy_helper.from_array(array, name=node.output[0]))
        hoisted += 1
    del model.graph.node[:]
    model.graph.node.extend(retained)
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            if name in renamed:
                node.input[index] = renamed[name]
    kept = [v for v in model.graph.value_info if v.name not in renamed]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept)
    return hoisted


def merge_pads_into_convs(model: onnx.ModelProto) -> int:
    """Fold constant zero Pad nodes into the asymmetric pads of the Conv they feed."""
    initializers = {t.name: t for t in model.graph.initializer}
    constants = {
        n.output[0]: numpy_helper.to_array(n.attribute[0].t) for n in model.graph.node if n.op_type == "Constant"
    }
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    def value(name: str):
        if name in constants:
            return constants[name]
        if name in initializers:
            return numpy_helper.to_array(initializers[name])
        return None

    removed = set()
    for node in model.graph.node:
        if node.op_type != "Pad" or len(consumers.get(node.output[0], [])) != 1:
            continue
        conv = consumers[node.output[0]][0]
        mode = next((a.s.decode() for a in node.attribute if a.name == "mode"), "constant")
        pads = value(node.input[1])
        fill = value(node.input[2]) if len(node.input) > 2 and node.input[2] else None
        if conv.op_type != "Conv" or conv.input[0] != node.output[0] or mode != "constant" or pads is None:
            continue
        if fill is not None and np.any(fill != 0):
            continue
        rank = len(pads) // 2
        begin, end = pads[:rank], pads[rank:]
        if np.any(begin[:2] != 0) or np.any(end[:2] != 0):
            continue
        spatial = rank - 2
        attribute = next((a for a in conv.attribute if a.name == "pads"), None)
        existing = list(attribute.ints) if attribute is not None else [0] * (2 * spatial)
        merged = [int(existing[i] + begin[2 + i]) for i in range(spatial)]
        merged += [int(existing[spatial + i] + end[2 + i]) for i in range(spatial)]
        if attribute is None:
            conv.attribute.append(helper.make_attribute("pads", merged))
        else:
            del attribute.ints[:]
            attribute.ints.extend(merged)
        conv.input[0] = node.input[0]
        removed.add(id(node))
    kept = [n for n in model.graph.node if id(n) not in removed]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    return len(removed)


def add_initializer_value_info(model: onnx.ModelProto) -> None:
    """Klartraum reads every tensor's shape from value_info, including initializers."""
    known = {v.name for v in (*model.graph.input, *model.graph.output, *model.graph.value_info)}
    for tensor in model.graph.initializer:
        if tensor.name not in known:
            model.graph.value_info.append(helper.make_tensor_value_info(tensor.name, tensor.data_type, tensor.dims))


def require_concrete_shapes(model: onnx.ModelProto) -> None:
    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        dims = value.type.tensor_type.shape.dim
        if not all(d.HasField("dim_value") and d.dim_value > 0 for d in dims):
            raise RuntimeError(f"tensor {value.name} has no concrete shape")


def fold_small_constants(model: onnx.ModelProto, limit: int = 65536) -> int:
    """Evaluate nodes whose inputs are all small constants and store the results as initializers.

    The tracer computes F.pad amounts and similar values with small runtime
    subgraphs; folding them gives the following nodes static shapes.
    """
    known: dict[str, np.ndarray] = {}
    for tensor in model.graph.initializer:
        if not tensor.external_data and np.prod(tensor.dims, dtype=np.int64) <= limit:
            known[tensor.name] = numpy_helper.to_array(tensor)
    retained, folded = [], 0
    for node in model.graph.node:
        inputs = [name for name in node.input if name]
        evaluable = node.op_type == "Constant" or (inputs and all(name in known for name in inputs))
        if evaluable and node.op_type not in {"RandomNormal", "RandomUniform"}:
            if node.op_type == "Constant":
                tensor = next((a.t for a in node.attribute if a.name == "value" and a.HasField("t")), None)
                if tensor is not None and not tensor.external_data and np.prod(tensor.dims, dtype=np.int64) <= limit:
                    known[node.output[0]] = numpy_helper.to_array(tensor)
                retained.append(node)
                continue
            evaluator = ReferenceEvaluator(node)
            results = evaluator.run(None, {name: known[name] for name in inputs})
            if all(np.asarray(r).size <= limit for r in results):
                for name, result in zip(node.output, results):
                    known[name] = np.asarray(result)
                    model.graph.initializer.append(numpy_helper.from_array(np.asarray(result), name=name))
                folded += 1
                continue
        retained.append(node)
    del model.graph.node[:]
    model.graph.node.extend(retained)
    return folded


def remove_dead_nodes(model: onnx.ModelProto) -> int:
    needed = {output.name for output in model.graph.output}
    kept = []
    for node in reversed(model.graph.node):
        if any(name in needed for name in node.output):
            kept.append(node)
            needed.update(name for name in node.input if name)
    removed = len(model.graph.node) - len(kept)
    del model.graph.node[:]
    model.graph.node.extend(reversed(kept))
    initializers = [t for t in model.graph.initializer if t.name in needed]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    values = [v for v in model.graph.value_info if v.name in needed]
    del model.graph.value_info[:]
    model.graph.value_info.extend(values)
    return removed


def export_graph(module, inputs, input_names, output_names, path: Path) -> None:
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=".export_") as scratch:
        raw = Path(scratch) / "raw.onnx"
        with torch.no_grad():
            torch.onnx.export(
                module, inputs, str(raw), input_names=input_names, output_names=output_names,
                opset_version=17, dynamo=False, do_constant_folding=True,
            )
        print(f"  traced in {time.perf_counter() - started:.1f}s", flush=True)
        # Fold and infer on the graph structure only; large weights stay on disk until saving.
        model = onnx.load(raw, load_external_data=False)
        folded = fold_small_constants(model)
        model = shape_inference.infer_shapes(model, strict_mode=True)
        load_external_data_for_model(model, scratch)
    print(f"  folded {folded} constant nodes, hoisted {hoist_constants(model)} constants", flush=True)
    merged = merge_pads_into_convs(model)
    print(f"  merged {merged} zero Pad nodes into Conv pads, removed {remove_dead_nodes(model)} dead nodes", flush=True)
    add_initializer_value_info(model)
    require_concrete_shapes(model)
    data_name = path.name + ".data"
    # ONNX appends to an existing external-data file.
    (path.parent / data_name).unlink(missing_ok=True)
    onnx.save_model(model, str(path), save_as_external_data=True, all_tensors_to_one_file=True,
                    location=data_name, size_threshold=1024)
    del model
    gc.collect()
    print(f"  saved {path.name} in {time.perf_counter() - started:.1f}s", flush=True)
    write_operator_report(path)


def write_operator_report(path: Path) -> None:
    model = onnx.load(path, load_external_data=False)
    counts = Counter(node.op_type for node in model.graph.node)
    lines = [f"{op}: {count}" for op, count in sorted(counts.items())]
    path.with_suffix(".operators.txt").write_text("\n".join(lines) + "\n")
    print(f"  operators: {dict(sorted(counts.items()))}", flush=True)


def run_ort(path: Path, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    started = time.perf_counter()
    result = session.run(None, feeds)
    print(f"  ONNX Runtime in {time.perf_counter() - started:.1f}s", flush=True)
    return result


def text_feeds(out: Path) -> dict[str, np.ndarray]:
    return {
        "input_ids": load(out, "input_ids_i64.bin"),
        "rope_cos": load(out, "text_rope_cos_f32.bin"),
        "rope_sin": load(out, "text_rope_sin_f32.bin"),
    }


def kv_names(layers: int) -> list[str]:
    names = []
    for layer in range(layers):
        names += [f"text_k_{layer}", f"text_v_{layer}"]
    return names


def stage_text(args: argparse.Namespace) -> None:
    out = args.onnx_dir
    cfg = load_config(out)
    transformer = load_transformer(keep="text")
    module = TextKV(transformer, cfg["text_length"]).eval()
    feeds = text_feeds(out)
    torch_inputs = tuple(torch.from_numpy(feeds[name]) for name in ("input_ids", "rope_cos", "rope_sin"))
    started = time.perf_counter()
    with torch.no_grad():
        kv = [t.numpy() for t in module(*torch_inputs)]
    print(f"  PyTorch text tower in {time.perf_counter() - started:.1f}s", flush=True)
    names = kv_names(cfg["num_layers"])
    for name, value in zip(names, kv):
        save(out, f"{name}_f32.bin", value)
    if args.fixtures_only:
        return
    path = out / "text_kv.onnx"
    export_graph(module, torch_inputs, list(feeds), names, path)
    del module, transformer
    gc.collect()
    if not args.skip_ort:
        result = run_ort(path, feeds)
        worst = max(report(name, value, expected) for name, value, expected in zip(names, result, kv))
        print(f"  worst ORT text K/V error {worst:.3e}", flush=True)


def denoiser_feeds(out: Path, cfg: dict, step: int = 0) -> dict[str, np.ndarray]:
    latents = torch.from_numpy(load(out, "initial_latents_f32.bin"))
    feeds = {
        "tokens": patchify(latents, cfg["patch"]).numpy(),
        "timestep": load(out, "timesteps_f32.bin")[step : step + 1].copy(),
        "noisy_mask": load(out, "noisy_mask_f32.bin"),
        "rope_cos": load(out, "vision_rope_cos_f32.bin"),
        "rope_sin": load(out, "vision_rope_sin_f32.bin"),
        "text_bias": load(out, "text_bias_f32.bin"),
    }
    for name in kv_names(cfg["num_layers"]):
        feeds[name] = load(out, f"{name}_f32.bin")
    return feeds


def stage_denoiser(args: argparse.Namespace) -> None:
    out = args.onnx_dir
    cfg = load_config(out)
    transformer = load_transformer(keep="denoiser")
    module = Denoiser(transformer).eval()
    feeds = denoiser_feeds(out, cfg)
    torch_inputs = tuple(torch.from_numpy(value) for value in feeds.values())
    started = time.perf_counter()
    with torch.no_grad():
        velocity = module(*torch_inputs).numpy()
    print(f"  PyTorch denoiser in {time.perf_counter() - started:.1f}s", flush=True)
    save(out, "denoiser_step0_output_f32.bin", velocity)
    if args.fixtures_only:
        return

    reference = load(out, "reference_step0_velocity_f32.bin")
    shape = tuple(cfg["latent_shape"])
    for branch in range(2):
        unpacked = unpatchify(torch.from_numpy(velocity[branch]), shape, cfg["patch"]).numpy()
        report(f"wrapper vs diffusers velocity, branch {branch}", unpacked[:, 1:], reference[branch][:, 1:])

    path = out / "denoiser.onnx"
    export_graph(module, torch_inputs, list(feeds), ["velocity"], path)
    del module, transformer
    gc.collect()
    if not args.skip_ort:
        (result,) = run_ort(path, feeds)
        report("ORT vs PyTorch denoiser", result, velocity)


# ---------------------------------------------------------------------------
# Wan2.2 VAE, fixed-shape whole-clip formulation
# ---------------------------------------------------------------------------
#
# diffusers runs the Wan VAE chunk by chunk with feature caches. For a fixed
# clip this equals one pass over the whole clip in which every causal 3D
# convolution zero-pads two frames in front, except for two first-chunk rules
# that are reproduced explicitly:
# * a temporal upsampler passes frame 0 through and runs its time convolution
#   on frames 1.. as a separate zero-padded sequence;
# * the duplicate-upsample shortcut of the first chunk keeps only its last
#   duplicated frame.
# Zero padding is written as F.pad and merged into the following Conv's
# asymmetric pads attribute after export (merge_pads_into_convs).


def dims(x: torch.Tensor) -> tuple[int, ...]:
    """Static Python sizes, so tracing records constants instead of Shape operations."""
    return tuple(int(v) for v in x.shape)


def causal_conv(conv: torch.nn.Conv3d, x: torch.Tensor, stride=(1, 1, 1)) -> torch.Tensor:
    kt, kh, kw = dims(conv.weight)[2:]
    x = F.pad(x, (kw // 2, kw // 2, kh // 2, kh // 2, kt - 1, 0))
    return F.conv3d(x, conv.weight, conv.bias, stride=stride)


def channel_rms(x: torch.Tensor, norm) -> torch.Tensor:
    """WanRMS_norm: F.normalize over channels * sqrt(C) * gamma == x / rms(x) * gamma."""
    gamma = norm.gamma.reshape(1, -1, *([1] * (x.dim() - 2)))
    return x / torch.sqrt((x * x).mean(1, keepdim=True) + 1e-24) * gamma


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def residual_block(block, x: torch.Tensor) -> torch.Tensor:
    shortcut = x if isinstance(block.conv_shortcut, torch.nn.Identity) else causal_conv(block.conv_shortcut, x)
    x = causal_conv(block.conv1, silu(channel_rms(x, block.norm1)))
    x = causal_conv(block.conv2, silu(channel_rms(x, block.norm2)))
    return x + shortcut


def attention_block(block, x: torch.Tensor) -> torch.Tensor:
    """Single-head spatial attention per frame."""
    b, c, t, h, w = dims(x)
    frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    qkv = F.conv2d(channel_rms(frames, block.norm), block.to_qkv.weight, block.to_qkv.bias)
    qkv = qkv.reshape(b * t, 3 * c, h * w).permute(0, 2, 1)
    q, k, v = qkv.split(c, dim=-1)
    q = q * (1.0 / math.sqrt(c))
    context = torch.matmul(torch.softmax(torch.matmul(q, k.transpose(1, 2)), dim=-1), v)
    out = F.conv2d(context.permute(0, 2, 1).reshape(b * t, c, h, w), block.proj.weight, block.proj.bias)
    return out.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4) + x


def mid_block(block, x: torch.Tensor) -> torch.Tensor:
    x = residual_block(block.resnets[0], x)
    for attention, resnet in zip(block.attentions, block.resnets[1:]):
        x = residual_block(resnet, attention_block(attention, x))
    return x


def spatial_conv(conv: torch.nn.Conv2d, x: torch.Tensor, stride=1, pads=(1, 1, 1, 1)) -> torch.Tensor:
    """A per-frame 2D convolution written as a 3D convolution with a temporal kernel of one."""
    x = F.pad(x, (pads[0], pads[1], pads[2], pads[3], 0, 0))
    return F.conv3d(x, conv.weight.unsqueeze(2), conv.bias, stride=(1, stride, stride))


def upsample(resample, x: torch.Tensor) -> torch.Tensor:
    b, c, t, h, w = dims(x)
    if resample.mode == "upsample3d":
        first, rest = x[:, :, :1], x[:, :, 1:]
        rest = causal_conv(resample.time_conv, rest)
        rest = rest.reshape(b, 2, c, t - 1, h, w).permute(0, 2, 3, 1, 4, 5).reshape(b, c, 2 * (t - 1), h, w)
        x = torch.cat([first, rest], dim=2)
        t = dims(x)[2]
    # Nearest 2x upsampling of every frame: fold channels and time into one axis.
    x = F.interpolate(x.reshape(b, c * t, h, w), scale_factor=2.0, mode="nearest").reshape(b, c, t, 2 * h, 2 * w)
    return spatial_conv(resample.resample[1], x)


def dup_up(shortcut, x: torch.Tensor, first_chunk_trim: bool) -> torch.Tensor:
    b, c, t, h, w = dims(x)
    ft, fs = shortcut.factor_t, shortcut.factor_s
    index = torch.arange(c * shortcut.repeats, device=x.device) // shortcut.repeats
    x = torch.index_select(x, 1, index)
    x = x.reshape(b, shortcut.out_channels, ft, fs, fs, t, h, w).permute(0, 1, 5, 2, 6, 3, 7, 4)
    x = x.reshape(b, shortcut.out_channels, t * ft, h * fs, w * fs)
    return x[:, :, ft - 1 :] if first_chunk_trim and ft > 1 else x


def avg_down(shortcut, x: torch.Tensor) -> torch.Tensor:
    b, c, t, h, w = dims(x)
    ft, fs = shortcut.factor_t, shortcut.factor_s
    pad_t = (ft - t % ft) % ft
    if pad_t:
        x = torch.cat([x[:, :, :pad_t] * 0.0, x], dim=2)
        t += pad_t
    x = x.reshape(b, c, t // ft, ft, h // fs, fs, w // fs, fs).permute(0, 1, 3, 5, 7, 2, 4, 6)
    x = x.reshape(b, shortcut.out_channels, shortcut.group_size, t // ft, h // fs, w // fs)
    return x.mean(dim=2)


def patchify_video(x: torch.Tensor, p: int) -> torch.Tensor:
    b, c, f, h, w = dims(x)
    x = x.reshape(b, c, f, h // p, p, w // p, p).permute(0, 1, 6, 4, 2, 3, 5)
    return x.reshape(b, c * p * p, f, h // p, w // p)


def unpatchify_video(x: torch.Tensor, p: int) -> torch.Tensor:
    b, cp, f, h, w = dims(x)
    x = x.reshape(b, cp // (p * p), p, p, f, h, w).permute(0, 1, 4, 5, 3, 6, 2)
    return x.reshape(b, cp // (p * p), f, h * p, w * p)


class VaeEncoder(torch.nn.Module):
    """Single-frame encoder: [1, 3, 1, H, W] in [-1, 1] -> normalized latent [1, 48, 1, H/16, W/16]."""

    def __init__(self, vae: AutoencoderKLWan, mean: torch.Tensor, inv_std: torch.Tensor):
        super().__init__()
        self.encoder = vae.encoder
        self.patch = vae.config.patch_size
        z = vae.config.z_dim
        # Fold the posterior mode and the latent normalization into quant_conv.
        weight = vae.quant_conv.weight[:z] * inv_std.view(-1, 1, 1, 1, 1)
        bias = (vae.quant_conv.bias[:z] - mean) * inv_std
        self.quant = torch.nn.Conv3d(weight.shape[1], z, 1)
        self.quant.weight.data.copy_(weight)
        self.quant.bias.data.copy_(bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        encoder = self.encoder
        x = causal_conv(encoder.conv_in, patchify_video(image, self.patch))
        for block in encoder.down_blocks:
            shortcut = avg_down(block.avg_shortcut, x)
            for resnet in block.resnets:
                x = residual_block(resnet, x)
            if block.downsampler is not None:
                # One frame: the temporal downsampler's first chunk only resamples spatially.
                x = spatial_conv(block.downsampler.resample[1], x, stride=2, pads=(0, 1, 0, 1))
            x = x + shortcut
        x = mid_block(encoder.mid_block, x)
        x = causal_conv(encoder.conv_out, silu(channel_rms(x, encoder.norm_out)))
        return F.conv3d(x, self.quant.weight, self.quant.bias)


class VaeDecoder(torch.nn.Module):
    """Whole-clip decoder: normalized latents [1, 48, T, h, w] -> video [1, 3, 4T-3, 16h, 16w] (unclamped)."""

    def __init__(self, vae: AutoencoderKLWan, mean: torch.Tensor, inv_std: torch.Tensor):
        super().__init__()
        self.decoder = vae.decoder
        self.patch = vae.config.patch_size
        # Fold the latent de-normalization z = latents / inv_std + mean into post_quant_conv.
        conv = vae.post_quant_conv
        weight = conv.weight / inv_std.view(1, -1, 1, 1, 1)
        bias = conv.bias + (conv.weight[:, :, 0, 0, 0] * mean.view(1, -1)).sum(1)
        self.post_quant = torch.nn.Conv3d(weight.shape[1], weight.shape[0], 1)
        self.post_quant.weight.data.copy_(weight)
        self.post_quant.bias.data.copy_(bias)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        decoder = self.decoder
        x = F.conv3d(latents, self.post_quant.weight, self.post_quant.bias)
        x = causal_conv(decoder.conv_in, x)
        x = mid_block(decoder.mid_block, x)
        for block in decoder.up_blocks:
            block_input = x
            for resnet in block.resnets:
                x = residual_block(resnet, x)
            if block.upsampler is not None:
                x = upsample(block.upsampler, x)
            if block.avg_shortcut is not None:
                x = x + dup_up(block.avg_shortcut, block_input, first_chunk_trim=True)
        x = causal_conv(decoder.conv_out, silu(channel_rms(x, decoder.norm_out)))
        return unpatchify_video(x, self.patch)


def stage_vae(args: argparse.Namespace) -> None:
    out = args.onnx_dir
    cfg = load_config(out)
    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32).eval()
    mean = torch.tensor(cfg["vae_latents_mean"], dtype=torch.float32)
    inv_std = 1.0 / torch.tensor(cfg["vae_latents_std"], dtype=torch.float32)

    encoder = VaeEncoder(vae, mean, inv_std).eval()
    image = torch.from_numpy(load(out, "image_f32.bin"))
    with torch.no_grad():
        latent = encoder(image)
    report("encoder wrapper vs prepared latent", latent[0].numpy(), load(out, "condition_latent_f32.bin"))
    path = out / "vae_encoder.onnx"
    export_graph(encoder, (image,), ["image"], ["latent"], path)
    if not args.skip_ort:
        (result,) = run_ort(path, {"image": image.numpy()})
        report("ORT vs PyTorch encoder", result, latent.numpy())

    if cfg.get("decoder_tile", 0):
        print("  decoder_tile is set: the decoder is not exported; link a decoder of the tile size", flush=True)
        return
    decoder = VaeDecoder(vae, mean, inv_std).eval()
    latents = torch.from_numpy(load(out, "initial_latents_f32.bin")).unsqueeze(0)
    started = time.perf_counter()
    with torch.no_grad():
        video = decoder(latents)
        print(f"  PyTorch decoder in {time.perf_counter() - started:.1f}s, output {tuple(video.shape)}", flush=True)
        reference = vae.decode(latents / inv_std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)).sample
    report("decoder wrapper vs diffusers (clamped)", video.clamp(-1, 1).numpy(), reference.numpy())
    save(out, "decoder_reference_input_f32.bin", latents[0].numpy())
    save(out, "decoder_reference_output_f32.bin", video[0].numpy())
    path = out / "vae_decoder.onnx"
    export_graph(decoder, (latents,), ["latents"], ["video"], path)
    if not args.skip_ort:
        (result,) = run_ort(path, {"latents": latents.numpy()})
        report("ORT vs PyTorch decoder", result, video.numpy())


def stage_postprocess(args: argparse.Namespace) -> None:
    """Re-apply the graph clean-up to already exported graphs, keeping their weight files."""
    for name in ("text_kv", "denoiser", "vae_encoder", "vae_decoder"):
        path = args.onnx_dir / f"{name}.onnx"
        if not path.exists():
            continue
        model = onnx.load(path, load_external_data=False)
        hoisted = hoist_constants(model)
        add_initializer_value_info(model)
        require_concrete_shapes(model)
        onnx.save_model(model, str(path))
        write_operator_report(path)
        print(f"  {name}: hoisted {hoisted} constants", flush=True)


def tile_offsets(size: int, tile: int, stride: int) -> list[int]:
    """Tile starts advancing by ``stride``, the last one aligned to the end (klartraum::tileOffsets)."""
    if tile <= 0 or tile > size or stride <= 0 or stride > tile:
        raise ValueError(f"invalid tiling: size {size}, tile {tile}, stride {stride}")
    offsets = list(range(0, size - tile, stride))
    return offsets + [size - tile]


def blend_ramp(length: int, ramp: int, ramp_start: bool, ramp_end: bool) -> torch.Tensor:
    """Linear blend weights towards neighbouring tiles, capped at one (klartraum::blendRamp)."""
    i = torch.arange(length, dtype=torch.float32)
    weights = torch.ones(length)
    if ramp_start:
        weights = torch.minimum(weights, (i + 0.5) / ramp)
    if ramp_end:
        weights = torch.minimum(weights, (length - i - 0.5) / ramp)
    return weights


def tiled_decode(decoder: torch.nn.Module, latents: torch.Tensor, tile: int, stride: int,
                 device: torch.device) -> torch.Tensor:
    """Decode [1, C, T, h, w] latents in overlapping square tiles and blend them (klartraum::TileBlender).

    Each tile runs the fixed-size decoder; overlaps are blended with linear
    ramps over the overlap width, so the decoder's memory stays that of one tile.
    """
    _, _, _, height, width = latents.shape
    ys, xs = tile_offsets(height, tile, stride), tile_offsets(width, tile, stride)
    total, weight = None, None
    for top in ys:
        for left in xs:
            started = time.perf_counter()
            with torch.no_grad():
                part = decoder(latents[..., top : top + tile, left : left + tile].to(device)).cpu()
            scale = part.shape[-1] // tile
            ramp = (tile - stride) * scale
            size = tile * scale
            rows = blend_ramp(size, ramp, top > 0, top + tile < height)
            columns = blend_ramp(size, ramp, left > 0, left + tile < width)
            w = rows[:, None] * columns[None, :]
            if total is None:
                total = torch.zeros(part.shape[:3] + (height * scale, width * scale))
                weight = torch.zeros(height * scale, width * scale)
            y0, x0 = top * scale, left * scale
            total[..., y0 : y0 + size, x0 : x0 + size] += part * w
            weight[y0 : y0 + size, x0 : x0 + size] += w
            print(f"  tile ({top}, {left}) decoded in {time.perf_counter() - started:.1f}s", flush=True)
    return total / weight


def make_scheduler(cfg: dict) -> UniPCMultistepScheduler:
    scheduler = UniPCMultistepScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")
    scheduler = UniPCMultistepScheduler.from_config(
        scheduler.config, flow_shift=cfg["flow_shift"], use_karras_sigmas=False
    )
    sigmas = np.linspace(1.0 - 1.0 / scheduler.config.num_train_timesteps, 0.0, cfg["steps"] + 1)[:-1]
    scheduler.set_timesteps(cfg["steps"], sigmas=sigmas)
    return scheduler


def write_video(video: np.ndarray, path: Path, fps: float) -> None:
    """[3, T, H, W] in [-1, 1] -> mp4. export_to_video expects float frames in [0, 1]."""
    from diffusers.utils import export_to_video

    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.clip((np.transpose(video, (1, 2, 3, 0)) + 1.0) / 2.0, 0.0, 1.0)
    export_to_video(list(frames), str(path), fps=int(fps), macro_block_size=1)


def stage_pipeline(args: argparse.Namespace) -> None:
    """Float32 reference of the whole denoising loop and decode, using the exported graphs' math.

    Saves the guided velocity and the latents after every step so the host
    scheduler can be checked independently of GPU rounding.
    """
    out = args.onnx_dir
    cfg = load_config(out)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    transformer = load_transformer(keep="denoiser")
    module = Denoiser(transformer).eval().to(device)
    feeds = denoiser_feeds(out, cfg)
    static = {k: torch.from_numpy(v).to(device) for k, v in feeds.items() if k not in ("tokens", "timestep")}
    kv = [static.pop(name) for name in kv_names(cfg["num_layers"])]
    scheduler = make_scheduler(cfg)
    shape = tuple(cfg["latent_shape"])
    latents = torch.from_numpy(load(out, "initial_latents_f32.bin"))
    condition = latents[:, :1].clone()
    guided, history = [], []
    for step, t in enumerate(scheduler.timesteps):
        started = time.perf_counter()
        tokens = patchify(latents, cfg["patch"]).to(device)
        timestep = torch.tensor([float(t)], dtype=torch.float32, device=device)
        with torch.no_grad():
            velocity = module(tokens, timestep, static["noisy_mask"], static["rope_cos"], static["rope_sin"],
                              static["text_bias"], *kv).cpu()
        cond, uncond = (unpatchify(velocity[i], shape, cfg["patch"]) for i in range(2))
        # The conditioning frame is kept clean: its velocity is masked to zero.
        cond[:, :1] = 0.0
        uncond[:, :1] = 0.0
        velocity = uncond + cfg["guidance_scale"] * (cond - uncond)
        latents = scheduler.step(velocity.unsqueeze(0), t, latents.unsqueeze(0), return_dict=False)[0].squeeze(0)
        guided.append(velocity.numpy())
        history.append(latents.numpy())
        print(f"  step {step + 1}/{cfg['steps']} t={float(t):.1f} in {time.perf_counter() - started:.1f}s", flush=True)
    report("conditioning frame preserved", latents[:, :1].numpy(), condition.numpy())
    save(out, "reference_guided_velocity_f32.bin", np.stack(guided))
    save(out, "reference_step_latents_f32.bin", np.stack(history))
    save(out, "reference_final_latents_f32.bin", latents.numpy())
    del module, transformer, kv, static
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32).eval()
    mean = torch.tensor(cfg["vae_latents_mean"], dtype=torch.float32)
    inv_std = 1.0 / torch.tensor(cfg["vae_latents_std"], dtype=torch.float32)
    decoder = VaeDecoder(vae, mean, inv_std).eval().to(device)
    started = time.perf_counter()
    if cfg.get("decoder_tile", 0):
        video = tiled_decode(decoder, latents.unsqueeze(0), cfg["decoder_tile"], cfg["decoder_stride"], device)
        video = video.clamp(-1.0, 1.0)[0]
    else:
        with torch.no_grad():
            video = decoder(latents.unsqueeze(0).to(device)).clamp(-1.0, 1.0).cpu()[0]
    print(f"  decoded in {time.perf_counter() - started:.1f}s", flush=True)
    save(out, "reference_video_f32.bin", video.numpy())
    path = args.video_dir / "reference_fp32_wrappers.mp4"
    write_video(video.numpy(), path, cfg["fps"])
    print(f"  wrote {path}", flush=True)


def main() -> None:
    print_cosmos3_notice()
    args = parse_args()
    torch.manual_seed(0)
    {
        "prepare": stage_prepare,
        "reference": stage_reference,
        "text": stage_text,
        "denoiser": stage_denoiser,
        "vae": stage_vae,
        "pipeline": stage_pipeline,
        "postprocess": stage_postprocess,
    }[args.stage](args)


if __name__ == "__main__":
    main()
