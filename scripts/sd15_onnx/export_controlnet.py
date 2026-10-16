# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Export SD1.5 ControlNet-OpenPose and an LCM UNet that takes its residuals.

Writes to --onnx-dir (default data/onnx/sd15_controlnet_<size>):

    sd15_controlnet.onnx         (sample, timestep, encoder_hidden_states, controlnet_cond)
                                 -> down_residual_0..11, mid_residual
    sd15_unet_control.onnx       (sample, timestep, encoder_hidden_states, down_residual_0..11, mid_residual)
                                 -> noise_prediction; LCM-LoRA is fused into its weights
    lcm_schedule_f32.bin         per step: timestep, alpha_t, alpha_prev, c_skip, c_out
    lcm_timesteps_i64.bin
    lcm_noise_f32.bin            [steps, 4, h, w] noise of the initial latent and of each LCM step
    pose_reference_f32.bin       [1, 3, size, size] pose map the reference was made from
    prompt_embeddings_f32.bin    [1, 77, 768] embedding of the prompt (classifier-free guidance is not used)
    controlnet_residuals_f32.bin the ControlNet outputs of the first step, concatenated
    unet_reference_f32.bin       the UNet output of the first step
    final_latents_f32.bin        latents after the last step
    pipeline_reference_f32.bin   decoded image [1, 3, size, size] in [-1, 1]

The pose map is read from --pose (a PPM drawn by Klartraum's OpenPoseSkeletonRenderer, e.g. with
openpose_skeleton_example). The CLIP text encoder and VAE decoder of export_denoiser.py are reused by the
Klartraum pipeline; pass the same --size.

LCM-LoRA (latent-consistency/lcm-lora-sdv1-5) turns SD1.5 into a latent consistency model that needs 4
steps without classifier-free guidance, instead of about 30 guided steps.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from diffusers import AutoencoderKL, ControlNetModel, LCMScheduler, UNet2DConditionModel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import CLIPTextModel, CLIPTokenizer

from disclaimer import print_stable_diffusion_notice
import onnx
from onnx import shape_inference

from export_denoiser import (
    add_initializer_value_info,
    concretize_fixed_shapes,
    export_graph,
    fold_fixed_shape_expressions,
    write_operator_report,
)
from export_onnx import Decoder as VaeDecoder
from run_reference import MODEL_ID, save_image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
CONTROLNET_ID = "lllyasviel/control_v11p_sd15_openpose"
LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"
DEFAULT_PROMPT = (
    "a photograph of a man walking on a city sidewalk, full body, jeans and a dark jacket, "
    "natural daylight, realistic, sharp focus"
)
RESIDUAL_COUNT = 12


def fuse_lora(unet: UNet2DConditionModel, path: str, scale: float = 1.0) -> int:
    """Adds scale * alpha / rank * up @ down to every UNet weight the LoRA file adapts."""
    state = load_file(path)
    modules = {name.replace(".", "_"): module for name, module in unet.named_modules()}
    fused = 0
    for key in [key for key in state if key.endswith("lora_down.weight")]:
        prefix = key[: -len(".lora_down.weight")]
        name = prefix
        for strip in ("lora_unet_", "unet."):
            if name.startswith(strip):
                name = name[len(strip) :]
        module = modules.get(name.replace(".", "_"))
        if module is None:
            raise KeyError(f"LoRA key {key} does not match a UNet module")
        down = state[key].float()
        up = state[prefix + ".lora_up.weight"].float()
        rank = down.shape[0]
        alpha = float(state.get(prefix + ".alpha", torch.tensor(rank)).item())
        weight = module.weight.data
        if weight.dim() == 4:
            delta = (up.flatten(1) @ down.flatten(1)).reshape(weight.shape)
        else:
            delta = up @ down
        weight += scale * alpha / rank * delta
        fused += 1
    return fused


class ControlNet(nn.Module):
    def __init__(self, controlnet: ControlNetModel):
        super().__init__()
        self.controlnet = controlnet

    def forward(self, sample, timestep, encoder_hidden_states, controlnet_cond):
        down, mid = self.controlnet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
            conditioning_scale=1.0,
            return_dict=False,
        )
        return (*down, mid)


class ControlledUnet(nn.Module):
    def __init__(self, unet: UNet2DConditionModel):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, *residuals):
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=list(residuals[:RESIDUAL_COUNT]),
            mid_block_additional_residual=residuals[RESIDUAL_COUNT],
            return_dict=False,
        )[0]


def finalize_static_shapes(path: Path, rounds: int = 12) -> None:
    """Folds the Shape nodes the export leaves behind and removes Identity nodes, without loading the weights.

    Tensors without value_info stay unknown to fold_fixed_shape_expressions, so
    a Shape reading them survives; repeating shape inference, folding and
    concretization resolves them one level at a time, until a round changes
    nothing. Every tensor must end up with a fixed shape: Klartraum sizes its
    buffers from value_info.
    """
    model = onnx.load(path, load_external_data=False)
    for _ in range(rounds):
        model = shape_inference.infer_shapes(model)
        add_initializer_value_info(model)
        changes = fold_fixed_shape_expressions(model) + concretize_fixed_shapes(model)
        if changes == 0:
            break
    unknown = [
        value.name
        for value in model.graph.value_info
        if any(not dimension.HasField("dim_value") for dimension in value.type.tensor_type.shape.dim)
    ]
    if unknown:
        raise RuntimeError(f"{path.name}: {len(unknown)} tensors keep unknown dimensions, e.g. {unknown[:3]}")
    # Identity only forwards its input: let its consumers read the input directly.
    graph_outputs = {output.name for output in model.graph.output}
    renamed = {}
    kept = []
    for node in model.graph.node:
        if node.op_type == "Identity" and node.output[0] not in graph_outputs:
            renamed[node.output[0]] = renamed.get(node.input[0], node.input[0])
        else:
            kept.append(node)
    for node in kept:
        for index, name in enumerate(node.input):
            if name in renamed:
                node.input[index] = renamed[name]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    # Folding leaves small inline constants that no node reads any more.
    used = {name for node in model.graph.node for name in node.input}
    unused = {tensor.name for tensor in model.graph.initializer if tensor.name not in used and not tensor.external_data}
    initializers = [tensor for tensor in model.graph.initializer if tensor.name not in unused]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    value_info = [value for value in model.graph.value_info if value.name not in unused]
    del model.graph.value_info[:]
    model.graph.value_info.extend(value_info)
    add_initializer_value_info(model)
    onnx.save(model, path)
    onnx.checker.check_model(str(path))


def read_ppm(path: Path, size: int) -> torch.Tensor:
    data = path.read_bytes()
    header = data.split(b"\n", 3)
    width, height = (int(value) for value in header[1].split())
    if (width, height) != (size, size):
        raise ValueError(f"{path} is {width}x{height}, expected {size}x{size}")
    pixels = np.frombuffer(header[3], dtype=np.uint8).reshape(height, width, 3)
    return torch.from_numpy(pixels.astype(np.float32).transpose(2, 0, 1)[None] / 255.0)


def lcm_schedule(scheduler: LCMScheduler) -> np.ndarray:
    rows = []
    timesteps = scheduler.timesteps.tolist()
    for index, timestep in enumerate(timesteps):
        previous = timesteps[index + 1] if index + 1 < len(timesteps) else -1
        alpha = float(scheduler.alphas_cumprod[timestep])
        alpha_previous = float(scheduler.alphas_cumprod[previous]) if previous >= 0 else float(
            scheduler.final_alpha_cumprod
        )
        c_skip, c_out = scheduler.get_scalings_for_boundary_condition_discrete(timestep)
        rows.append([timestep, alpha, alpha_previous, float(c_skip), float(c_out)])
    return np.asarray(rows, dtype=np.float32)


def lcm_step(latents, noise_prediction, row, noise, last):
    """One LCM step (diffusers LCMScheduler.step) with an explicit noise tensor; returns (latents, denoised)."""
    _, alpha, alpha_previous, c_skip, c_out = (float(value) for value in row)
    predicted_original = (latents - (1.0 - alpha) ** 0.5 * noise_prediction) / alpha**0.5
    denoised = c_out * predicted_original + c_skip * latents
    if last:
        return denoised, denoised
    return alpha_previous**0.5 * denoised + (1.0 - alpha_previous) ** 0.5 * noise, denoised


def run_ort(path: Path, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return session.run(None, feeds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--onnx-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pose", type=Path, help="pose map PPM of --size pixels")
    parser.add_argument("--controlnet", default=CONTROLNET_ID,
                        help="ControlNet model id or a local directory with config.json and the weights")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--reuse-models", action="store_true", help="only regenerate the fixtures")
    return parser.parse_args()


def main() -> None:
    print_stable_diffusion_notice()
    args = parse_args()
    if args.size <= 0 or args.size % 64:
        raise ValueError("--size must be a positive multiple of 64 (the ControlNet downsamples by 64)")
    onnx_dir = args.onnx_dir or REPO_ROOT / "data" / "onnx" / f"sd15_controlnet_{args.size}"
    output_dir = args.output_dir or REPO_ROOT / "build" / "TestingOutput" / f"sd15_controlnet_{args.size}"
    pose_path = args.pose or REPO_ROOT / "build" / "TestingOutput" / "openpose_skeleton" / "frame_0000.ppm"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder").eval()
    tokens = tokenizer([args.prompt], padding="max_length", max_length=77, truncation=True, return_tensors="pt")
    with torch.inference_mode():
        embeddings = text_encoder(tokens.input_ids, attention_mask=tokens.attention_mask)[0].clone()
    embeddings = torch.from_numpy(embeddings.numpy().copy())
    del text_encoder
    gc.collect()

    unet = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet").eval()
    fused = fuse_lora(unet, hf_hub_download(LCM_LORA_ID, "pytorch_lora_weights.safetensors"))
    print(f"Fused LCM-LoRA into {fused} UNet weights")
    controlnet = ControlNetModel.from_pretrained(args.controlnet).eval()
    scheduler = LCMScheduler.from_pretrained(args.model, subfolder="scheduler")
    scheduler.set_timesteps(args.steps)
    schedule = lcm_schedule(scheduler)
    timesteps = scheduler.timesteps.numpy().astype(np.int64)
    print(f"LCM timesteps: {timesteps.tolist()}")

    latent_size = args.size // 8
    generator = torch.Generator().manual_seed(args.seed)
    noise = torch.randn((args.steps, 4, latent_size, latent_size), generator=generator)
    pose = read_ppm(pose_path, args.size)

    control_model, unet_model = ControlNet(controlnet).eval(), ControlledUnet(unet).eval()
    latents = noise[:1] * scheduler.init_noise_sigma
    first = None
    started = time.perf_counter()
    with torch.inference_mode():
        for index, timestep in enumerate(timesteps):
            timestep_input = torch.tensor([timestep], dtype=torch.int64)
            residuals = control_model(latents, timestep_input, embeddings, pose)
            noise_prediction = unet_model(latents, timestep_input, embeddings, *residuals)
            if first is None:
                first = (latents.clone(), timestep_input.clone(), [r.clone() for r in residuals],
                         noise_prediction.clone())
            last = index == len(timesteps) - 1
            step_noise = noise[index + 1 : index + 2] if not last else None
            latents, _ = lcm_step(latents, noise_prediction, schedule[index], step_noise, last)
    print(f"PyTorch ControlNet + LCM: {time.perf_counter() - started:.1f} s for {args.steps} steps")

    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae").eval()
    with torch.inference_mode():
        decoded = VaeDecoder(vae)(latents / vae.config.scaling_factor)
    save_image(decoded, output_dir / "sd15_controlnet_reference.png")

    sample, timestep, residuals, unet_output = (
        torch.from_numpy(first[0].numpy().copy()),
        torch.from_numpy(first[1].numpy().copy()),
        [torch.from_numpy(r.numpy().copy()) for r in first[2]],
        first[3].numpy(),
    )
    controlnet_path = onnx_dir / "sd15_controlnet.onnx"
    unet_path = onnx_dir / "sd15_unet_control.onnx"
    residual_names = [f"down_residual_{index}" for index in range(RESIDUAL_COUNT)] + ["mid_residual"]
    if not args.reuse_models:
        export_graph(control_model, (sample, timestep, embeddings, pose), controlnet_path,
               ["sample", "timestep", "encoder_hidden_states", "controlnet_cond"], residual_names)
        export_graph(unet_model, (sample, timestep, embeddings, *residuals), unet_path,
               ["sample", "timestep", "encoder_hidden_states", *residual_names], ["noise_prediction"])
    for name, path in (("ControlNet", controlnet_path), ("UNet", unet_path)):
        finalize_static_shapes(path)
        unsupported = write_operator_report(output_dir / f"{path.stem}_operator_report.txt", path)
        print(f"Unsupported Klartraum {name} operations ({len(unsupported)}): {', '.join(sorted(unsupported))}")

    feeds = {"sample": sample.numpy(), "timestep": timestep.numpy(), "encoder_hidden_states": embeddings.numpy()}
    ort_residuals = run_ort(controlnet_path, {**feeds, "controlnet_cond": pose.numpy()})
    residual_error = max(float(np.abs(a - b.numpy()).max()) for a, b in zip(ort_residuals, residuals))
    ort_unet = run_ort(unet_path, {**feeds, **dict(zip(residual_names, ort_residuals))})[0]
    unet_error = float(np.abs(ort_unet - unet_output).max())
    print(f"ONNX vs PyTorch maximum error: ControlNet {residual_error:.3g}, UNet {unet_error:.3g}")
    if residual_error > 2e-3 or unet_error > 2e-3:
        raise RuntimeError("ONNX export does not match PyTorch")

    schedule.tofile(onnx_dir / "lcm_schedule_f32.bin")
    timesteps.tofile(onnx_dir / "lcm_timesteps_i64.bin")
    noise.numpy().astype(np.float32).tofile(onnx_dir / "lcm_noise_f32.bin")
    pose.numpy().astype(np.float32).tofile(onnx_dir / "pose_reference_f32.bin")
    embeddings.numpy().astype(np.float32).tofile(onnx_dir / "prompt_embeddings_f32.bin")
    tokens.input_ids.numpy().astype(np.int64).tofile(onnx_dir / "prompt_input_ids_i64.bin")
    np.concatenate([r.reshape(-1) for r in ort_residuals]).astype(np.float32).tofile(
        onnx_dir / "controlnet_residuals_f32.bin")
    ort_unet.astype(np.float32).tofile(onnx_dir / "unet_reference_f32.bin")
    latents.numpy().astype(np.float32).tofile(onnx_dir / "final_latents_f32.bin")
    decoded.numpy().astype(np.float32).tofile(onnx_dir / "pipeline_reference_f32.bin")
    (onnx_dir / "prompt.txt").write_text(args.prompt + "\n", encoding="utf-8")
    print(f"Wrote ControlNet fixtures to {onnx_dir}")


if __name__ == "__main__":
    main()
