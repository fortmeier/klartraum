# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Export and verify the Stable Diffusion 1.5 VAE encoder and decoder."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from onnx import helper, shape_inference
from onnxsim import simplify

from run_reference import MODEL_ID, DEFAULT_INPUT, load_image, save_image


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_ONNX_DIR = REPO_ROOT / "data" / "onnx" / "sd15"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "build" / "TestingOutput" / "sd15_onnx"


class Encoder(nn.Module):
    def __init__(self, vae: AutoencoderKL):
        super().__init__()
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        moments = self.quant_conv(self.encoder(image))
        return torch.chunk(moments, 2, dim=1)[0]


class Decoder(nn.Module):
    def __init__(self, vae: AutoencoderKL):
        super().__init__()
        self.post_quant_conv = vae.post_quant_conv
        self.decoder = vae.decoder

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.post_quant_conv(latent))


def add_initializer_value_info(model: onnx.ModelProto) -> onnx.ModelProto:
    known = {value.name for value in model.graph.input}
    known.update(value.name for value in model.graph.output)
    known.update(value.name for value in model.graph.value_info)
    for tensor in model.graph.initializer:
        if tensor.name not in known:
            model.graph.value_info.append(
                helper.make_tensor_value_info(tensor.name, tensor.data_type, tensor.dims)
            )
    return model


def export(model: nn.Module, sample: torch.Tensor, path: Path) -> None:
    raw_path = path.with_suffix(".raw.onnx")
    torch.onnx.export(
        model,
        sample,
        raw_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )
    inferred = shape_inference.infer_shapes(onnx.load(raw_path))
    simplified, valid = simplify(inferred)
    if not valid:
        raise RuntimeError(f"onnxsim rejected {raw_path.name}")
    onnx.save(add_initializer_value_info(simplified), path)
    raw_path.unlink()
    onnx.checker.check_model(onnx.load(path))


def run_ort(path: Path, value: np.ndarray) -> np.ndarray:
    # CPU is used as the portable reference backend. It also avoids tying the
    # generated fixtures to the CUDA/cuDNN version installed on the host.
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return session.run(["output"], {"input": value})[0]


def summarize(path: Path) -> None:
    model = onnx.load(path, load_external_data=False)
    operations = Counter(node.op_type for node in model.graph.node)
    size_mib = path.stat().st_size / (1024 * 1024)
    print(f"{path.name}: {len(model.graph.node)} nodes, {size_mib:.1f} MiB")
    print("  " + ", ".join(f"{name}={count}" for name, count in sorted(operations.items())))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    # Klartraum currently keeps every intermediate tensor alive for the whole
    # graph. Keep the checked-in smoke model small enough to execute on an 8 GB
    # GPU; pass --size 512 to produce the canonical SD 1.5 VAE shape.
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--model", default=MODEL_ID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.size <= 0 or args.size % 8:
        raise ValueError("--size must be a positive multiple of 8")
    args.onnx_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    vae = AutoencoderKL.from_pretrained(
        args.model, subfolder="vae", torch_dtype=torch.float32
    ).eval().cpu()
    encoder = Encoder(vae).eval()
    decoder = Decoder(vae).eval()
    image = load_image(args.input, args.size)
    with torch.inference_mode():
        latent = encoder(image)
        decoded = decoder(latent)
    # torch.onnx traces with autograd enabled and cannot retain an inference-mode
    # tensor as an intermediate. A regular tensor with identical values keeps the
    # exported decoder input tied to the lantern reference run.
    latent_for_export = torch.from_numpy(latent.numpy().copy())

    encoder_path = args.onnx_dir / "sd15_vae_encoder.onnx"
    decoder_path = args.onnx_dir / "sd15_vae_decoder.onnx"
    export(encoder, image, encoder_path)
    export(decoder, latent_for_export, decoder_path)
    summarize(encoder_path)
    summarize(decoder_path)

    image_array = image.numpy()
    latent_array = latent.numpy()
    encoder_output = run_ort(encoder_path, image_array)
    decoder_output = run_ort(decoder_path, encoder_output)
    encoder_error = float(np.max(np.abs(encoder_output - latent_array)))
    decoder_error = float(np.max(np.abs(decoder_output - decoded.numpy())))
    print(f"ONNX encoder maximum absolute error: {encoder_error:.7g}")
    print(f"ONNX decoder maximum absolute error: {decoder_error:.7g}")
    if not np.allclose(encoder_output, latent_array, atol=2e-3, rtol=2e-3):
        raise RuntimeError("ONNX encoder does not match PyTorch")
    if not np.allclose(decoder_output, decoded.numpy(), atol=2e-3, rtol=2e-3):
        raise RuntimeError("ONNX decoder does not match PyTorch")

    np.save(args.output_dir / "lantern_input.npy", image_array)
    np.save(args.output_dir / "lantern_sd15_onnx_latent.npy", encoder_output)
    np.save(args.output_dir / "lantern_sd15_onnx_decoded.npy", decoder_output)
    save_image(torch.from_numpy(decoder_output), args.output_dir / "lantern_sd15_onnx.png")
    image_array.astype(np.float32).tofile(args.onnx_dir / "lantern_input_f32.bin")
    decoder_output.astype(np.float32).tofile(args.onnx_dir / "lantern_reference_f32.bin")
    print(f"Wrote ONNX models to {args.onnx_dir}")


if __name__ == "__main__":
    main()
