# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Export the Qwen-Image VAE (the Wan 2.1 VAE) for single images as fixed-shape float32 ONNX graphs for Klartraum.

The VAE is a causal 3D convolutional autoencoder. For one frame, diffusers' first-chunk path reduces it to 2D: every
causal convolution zero-pads its kt - 1 earlier frames, so only the last time slice of its kernel sees the frame, and
the temporal resamplers skip their time convolutions on the first chunk. The graphs are written with 2D convolutions:

* ``vae_encoder_<H>x<W>.onnx``: image [1, 3, H, W] in [-1, 1] -> normalized latent [1, 16, H/8, W/8], the posterior
  mode with the pipeline's (z - mean) / std folded into quant_conv;
* ``vae_decoder_<H>x<W>.onnx``: normalized latent -> image [1, 3, H, W], unclamped (diffusers clamps to [-1, 1]).

Both are checked against diffusers' AutoencoderKLQwenImage and ONNX Runtime; a reference input and output pair is
stored next to each graph for the Klartraum test (tests/test_qwen_image_vae.cpp):

    uv run --project scripts/qwen_image python scripts/qwen_image/export_vae.py --size 256 256
"""

from __future__ import annotations

import argparse
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLQwenImage
from onnx import helper, numpy_helper, shape_inference
from onnx.external_data_helper import load_external_data_for_model
from onnx.reference import ReferenceEvaluator

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "build/models/qwen-image/original"
OUT = ROOT / "build/models/qwen-image/onnx"


def dims(x: torch.Tensor) -> tuple[int, ...]:
    """Static Python sizes, so tracing records constants instead of Shape operations."""
    return tuple(int(v) for v in x.shape)


def frame_conv(conv: torch.nn.Conv3d, x: torch.Tensor, stride: int = 1) -> torch.Tensor:
    """A causal 3D convolution applied to a single frame: the 2D convolution with its kernel's last time slice."""
    kh, kw = dims(conv.weight)[3:]
    return F.conv2d(x, conv.weight[:, :, -1], conv.bias, stride=stride, padding=(kh // 2, kw // 2))


def channel_rms(x: torch.Tensor, norm) -> torch.Tensor:
    """QwenImageRMS_norm: F.normalize over channels * sqrt(C) * gamma == x / rms(x) * gamma.

    Written in the form Klartraum's importer fuses into one ChannelRmsNormalization.
    """
    gamma = norm.gamma.reshape(1, -1, 1, 1)
    return x / torch.sqrt((x * x).mean(1, keepdim=True) + 1e-24) * gamma


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


def residual_block(block, x: torch.Tensor) -> torch.Tensor:
    shortcut = x if isinstance(block.conv_shortcut, torch.nn.Identity) else frame_conv(block.conv_shortcut, x)
    x = frame_conv(block.conv1, silu(channel_rms(x, block.norm1)))
    x = frame_conv(block.conv2, silu(channel_rms(x, block.norm2)))
    return x + shortcut


def attention_block(block, x: torch.Tensor) -> torch.Tensor:
    """Single-head attention over the pixels of the frame."""
    b, c, h, w = dims(x)
    qkv = F.conv2d(channel_rms(x, block.norm), block.to_qkv.weight, block.to_qkv.bias)
    qkv = qkv.reshape(b, 3 * c, h * w).permute(0, 2, 1)
    q, k, v = qkv.split(c, dim=-1)
    q = q * (1.0 / c**0.5)
    context = torch.matmul(torch.softmax(torch.matmul(q, k.transpose(1, 2)), dim=-1), v)
    out = F.conv2d(context.permute(0, 2, 1).reshape(b, c, h, w), block.proj.weight, block.proj.bias)
    return out + x


def mid_block(block, x: torch.Tensor) -> torch.Tensor:
    x = residual_block(block.resnets[0], x)
    for attention, resnet in zip(block.attentions, block.resnets[1:]):
        x = residual_block(resnet, attention_block(attention, x))
    return x


def resample(module, x: torch.Tensor) -> torch.Tensor:
    """QwenImageResample on the first chunk: spatial resampling only."""
    conv = module.resample[1]
    if module.mode.startswith("upsample"):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return F.conv2d(x, conv.weight, conv.bias, padding=1)
    # ZeroPad2d((0, 1, 0, 1)) followed by a stride-2 convolution.
    return F.conv2d(F.pad(x, (0, 1, 0, 1)), conv.weight, conv.bias, stride=2)


class VaeEncoder(torch.nn.Module):
    """Image [1, 3, H, W] in [-1, 1] -> normalized latent [1, 16, H/8, W/8]."""

    def __init__(self, vae: AutoencoderKLQwenImage, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.encoder = vae.encoder
        z = vae.config.z_dim
        # The posterior mode (the first z channels) and (z - mean) / std, folded into quant_conv.
        weight = vae.quant_conv.weight[:z, :, 0] / std.view(-1, 1, 1, 1)
        self.quant_weight = torch.nn.Parameter(weight.detach().clone(), requires_grad=False)
        self.quant_bias = torch.nn.Parameter(((vae.quant_conv.bias[:z] - mean) / std).detach().clone(),
                                             requires_grad=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        encoder = self.encoder
        x = frame_conv(encoder.conv_in, image)
        for layer in encoder.down_blocks:
            if hasattr(layer, "mode"):
                x = resample(layer, x)
            elif hasattr(layer, "to_qkv"):
                x = attention_block(layer, x)
            else:
                x = residual_block(layer, x)
        x = mid_block(encoder.mid_block, x)
        x = frame_conv(encoder.conv_out, silu(channel_rms(x, encoder.norm_out)))
        return F.conv2d(x, self.quant_weight, self.quant_bias)


class VaeDecoder(torch.nn.Module):
    """Normalized latent [1, 16, h, w] -> image [1, 3, 8h, 8w] (unclamped)."""

    def __init__(self, vae: AutoencoderKLQwenImage, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.decoder = vae.decoder
        # z = latent * std + mean, folded into post_quant_conv.
        conv = vae.post_quant_conv
        weight = conv.weight[:, :, 0]
        self.post_weight = torch.nn.Parameter((weight * std.view(1, -1, 1, 1)).detach().clone(), requires_grad=False)
        self.post_bias = torch.nn.Parameter((conv.bias + (weight[:, :, 0, 0] * mean.view(1, -1)).sum(1)).detach().clone(),
                                            requires_grad=False)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        decoder = self.decoder
        x = F.conv2d(latent, self.post_weight, self.post_bias)
        x = frame_conv(decoder.conv_in, x)
        x = mid_block(decoder.mid_block, x)
        for block in decoder.up_blocks:
            for resnet in block.resnets:
                x = residual_block(resnet, x)
            if block.upsamplers is not None:
                x = resample(block.upsamplers[0], x)
        return frame_conv(decoder.conv_out, silu(channel_rms(x, decoder.norm_out)))


# ---------------------------------------------------------------------------
# Graph clean-up for Klartraum's importer (after scripts/cosmos3/export_onnx.py of this repository)
# ---------------------------------------------------------------------------


def fold_small_constants(model: onnx.ModelProto, limit: int = 65536) -> int:
    """Evaluate nodes whose inputs are all small constants and store the results as initializers.

    The tracer computes F.pad amounts with small runtime subgraphs; folding them gives the Pads constant amounts.
    """
    known: dict[str, np.ndarray] = {}
    for tensor in model.graph.initializer:
        if not tensor.external_data and np.prod(tensor.dims, dtype=np.int64) <= limit:
            known[tensor.name] = numpy_helper.to_array(tensor)
    retained, folded = [], 0
    for node in model.graph.node:
        inputs = [name for name in node.input if name]
        if node.op_type == "Constant":
            tensor = next((a.t for a in node.attribute if a.name == "value" and a.HasField("t")), None)
            if tensor is not None and not tensor.external_data and np.prod(tensor.dims, dtype=np.int64) <= limit:
                known[node.output[0]] = numpy_helper.to_array(tensor)
            retained.append(node)
            continue
        if inputs and all(name in known for name in inputs):
            results = ReferenceEvaluator(node).run(None, {name: known[name] for name in inputs})
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


def merge_pads_into_convs(model: onnx.ModelProto) -> int:
    """Fold constant zero Pad nodes into the asymmetric pads of the Conv they feed (Klartraum has no Pad)."""
    initializers = {t.name: t for t in model.graph.initializer}
    constants = {n.output[0]: numpy_helper.to_array(n.attribute[0].t) for n in model.graph.node if n.op_type == "Constant"}
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


def add_initializer_value_info(model: onnx.ModelProto) -> None:
    """Klartraum reads every tensor's shape from value_info, including initializers."""
    known = {v.name for v in (*model.graph.input, *model.graph.output, *model.graph.value_info)}
    for tensor in model.graph.initializer:
        if tensor.name not in known:
            model.graph.value_info.append(helper.make_tensor_value_info(tensor.name, tensor.data_type, tensor.dims))


def require_concrete_shapes(model: onnx.ModelProto) -> None:
    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        shape = value.type.tensor_type.shape.dim
        if not all(d.HasField("dim_value") and d.dim_value > 0 for d in shape):
            raise RuntimeError(f"tensor {value.name} has no concrete shape")


def export_graph(module, example: torch.Tensor, input_name: str, output_name: str, path: Path) -> None:
    with torch.no_grad():
        output_shape = dims(module(example))
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=".export_") as scratch:
        raw = Path(scratch) / "raw.onnx"
        with torch.no_grad():
            torch.onnx.export(module, (example,), str(raw), input_names=[input_name], output_names=[output_name],
                              opset_version=17, dynamo=False, do_constant_folding=True)
        model = onnx.load(raw, load_external_data=False)
        output = model.graph.output[0].type.tensor_type.shape
        del output.dim[:]
        for size in output_shape:
            output.dim.add().dim_value = size
        fold_small_constants(model)
        model = shape_inference.infer_shapes(model, strict_mode=True)
        load_external_data_for_model(model, scratch)
    merged = merge_pads_into_convs(model)
    removed = remove_dead_nodes(model)
    add_initializer_value_info(model)
    require_concrete_shapes(model)
    data_name = path.name + ".data"
    (path.parent / data_name).unlink(missing_ok=True)
    onnx.save_model(model, str(path), save_as_external_data=True, all_tensors_to_one_file=True, location=data_name,
                    size_threshold=1024)
    counts = Counter(node.op_type for node in model.graph.node)
    print(f"  {path.name}: merged {merged} Pads, removed {removed} dead nodes; operators {dict(sorted(counts.items()))}")


def run_ort(path: Path, name: str, x: np.ndarray) -> np.ndarray:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    return session.run(None, {name: x})[0]


def report(label: str, actual: np.ndarray, expected: np.ndarray) -> None:
    error = np.abs(actual - expected)
    print(f"  {label}: max |diff| {error.max():.3e}, mean {error.mean():.3e}, expected range "
          f"[{expected.min():.3f}, {expected.max():.3f}]")


def test_image(height: int, width: int) -> torch.Tensor:
    """A smooth colourful pattern with an edge, in [-1, 1], so the round trip is meaningful without an image file."""
    y, x = torch.meshgrid(torch.linspace(-1, 1, height), torch.linspace(-1, 1, width), indexing="ij")
    image = torch.stack([torch.sin(3 * x + y), torch.cos(2 * y - x), torch.where(x * x + y * y < 0.4, 0.8, -0.6)])
    return image.unsqueeze(0).float()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, nargs=2, default=(512, 512), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    height, width = args.size
    if height % 16 or width % 16:
        raise SystemExit("the size must be a multiple of 16 (the transformer's 2x2 patches of the 8x latent)")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    vae = AutoencoderKLQwenImage.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32).eval()
    mean = torch.tensor(vae.config.latents_mean, dtype=torch.float32)
    std = torch.tensor(vae.config.latents_std, dtype=torch.float32)
    suffix = f"{height}x{width}"

    image = test_image(height, width)
    encoder = VaeEncoder(vae, mean, std).eval()
    with torch.no_grad():
        latent = encoder(image)
        posterior = vae.encode(image.unsqueeze(2)).latent_dist.mode()[:, :, 0]
        expected = (posterior - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
    report("encoder vs diffusers", latent.numpy(), expected.numpy())
    path = args.out / f"vae_encoder_{suffix}.onnx"
    export_graph(encoder, image, "image", "latent", path)
    report("encoder ONNX Runtime vs PyTorch", run_ort(path, "image", image.numpy()), latent.numpy())
    image.numpy().tofile(args.out / f"vae_encoder_{suffix}_input_f32.bin")
    latent.numpy().tofile(args.out / f"vae_encoder_{suffix}_output_f32.bin")

    decoder = VaeDecoder(vae, mean, std).eval()
    with torch.no_grad():
        decoded = decoder(latent)
        z = latent * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
        expected = vae.decode(z.unsqueeze(2)).sample[:, :, 0]
    report("decoder vs diffusers (clamped)", decoded.clamp(-1, 1).numpy(), expected.numpy())
    report("round trip vs input", decoded.clamp(-1, 1).numpy(), image.numpy())
    path = args.out / f"vae_decoder_{suffix}.onnx"
    export_graph(decoder, latent, "latent", "image", path)
    report("decoder ONNX Runtime vs PyTorch", run_ort(path, "latent", latent.numpy()), decoded.numpy())
    latent.numpy().tofile(args.out / f"vae_decoder_{suffix}_input_f32.bin")
    decoded.numpy().tofile(args.out / f"vae_decoder_{suffix}_output_f32.bin")


if __name__ == "__main__":
    main()
