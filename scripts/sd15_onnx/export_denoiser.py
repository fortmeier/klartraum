"""Export and verify a small end-to-end SD1.5 DDIM inference pipeline."""

from __future__ import annotations

import argparse
import gc
import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from diffusers import AutoencoderKL, DDIMScheduler, PNDMScheduler, UNet2DConditionModel
from onnx import helper, shape_inference
from onnx import numpy_helper
from transformers import CLIPTextModel, CLIPTokenizer

from export_onnx import Decoder as VaeDecoder
from export_onnx import export as export_vae_component
from export_onnx import run_ort as run_vae_ort
from disclaimer import print_stable_diffusion_notice
from run_reference import MODEL_ID, save_image


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_ONNX_DIR = REPO_ROOT / "data" / "onnx" / "sd15_denoiser_256"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "build" / "TestingOutput" / "sd15_denoiser_256"
DEFAULT_PROMPT = (
    "a realistic photograph of a traditional Japanese stone lantern in a green garden, "
    "single gray granite garden lantern, centered, moss, natural daylight"
)
DEFAULT_NEGATIVE_PROMPT = (
    "person, building, house, flower pot, collage, multiple images, metal, painting, "
    "illustration, abstract, blurry, distorted, oversaturated, text"
)
KLARTRAUM_OPERATIONS = {
    "Add",
    "Cast",
    "Concat",
    "Constant",
    "Conv",
    "ConvTranspose",
    "Cos",
    "Div",
    "Erf",
    "Expand",
    "Gemm",
    "Gather",
    "InstanceNormalization",
    "LayerNormalization",
    "MatMul",
    "Mul",
    "Relu",
    "Reshape",
    "Resize",
    "Sigmoid",
    "Sin",
    "Slice",
    "Softmax",
    "Sqrt",
    "Sub",
    "Split",
    "Transpose",
    "Unsqueeze",
}


class Denoiser(nn.Module):
    def __init__(self, unet: UNet2DConditionModel):
        super().__init__()
        self.unet = unet

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]


class TextEncoder(nn.Module):
    def __init__(self, encoder: CLIPTextModel):
        super().__init__()
        self.encoder = encoder
        causal_attention_mask = torch.full(
            (77, 77), torch.finfo(torch.float32).min, dtype=torch.float32
        ).triu(diagonal=1)
        self.register_buffer(
            "causal_attention_mask", causal_attention_mask[None, None, :, :]
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        text_model = self.encoder.text_model
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        hidden_states = text_model.embeddings(input_ids=input_ids)

        sequence_length = input_shape[-1]
        causal_attention_mask = self.causal_attention_mask.expand(
            input_shape[0], 1, sequence_length, sequence_length
        )

        expanded_attention_mask = attention_mask[:, None, None, :].expand(
            input_shape[0], 1, sequence_length, sequence_length
        ).to(hidden_states.dtype)
        expanded_attention_mask = (
            1.0 - expanded_attention_mask
        ) * -10000.0

        encoder_outputs = text_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=expanded_attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=False,
            output_hidden_states=False,
        )
        return text_model.final_layer_norm(encoder_outputs[0])


def add_initializer_value_info(model: onnx.ModelProto) -> onnx.ModelProto:
    values = {value.name: value for value in model.graph.value_info}
    graph_values = {value.name for value in model.graph.input}
    graph_values.update(value.name for value in model.graph.output)
    for tensor in model.graph.initializer:
        if tensor.name in graph_values:
            continue
        tensor_info = helper.make_tensor_value_info(tensor.name, tensor.data_type, tensor.dims)
        if tensor.name in values:
            values[tensor.name].type.CopyFrom(tensor_info.type)
        else:
            model.graph.value_info.append(tensor_info)
            values[tensor.name] = model.graph.value_info[-1]
    return model


def fold_fixed_shape_expressions(model: onnx.ModelProto) -> int:
    """Evaluate small shape-only subgraphs for this fixed-size experiment."""
    shapes: dict[str, tuple[int, ...]] = {}
    dtypes: dict[str, np.dtype] = {}
    for value in (*model.graph.input, *model.graph.output, *model.graph.value_info):
        tensor_type = value.type.tensor_type
        if tensor_type.elem_type:
            dtypes[value.name] = np.dtype(helper.tensor_dtype_to_np_dtype(tensor_type.elem_type))
        if tensor_type.HasField("shape") and all(d.HasField("dim_value") for d in tensor_type.shape.dim):
            shapes[value.name] = tuple(d.dim_value for d in tensor_type.shape.dim)
    for tensor in model.graph.initializer:
        shapes[tensor.name] = tuple(tensor.dims)

    constants: dict[str, np.ndarray] = {
        tensor.name: numpy_helper.to_array(tensor)
        for tensor in model.graph.initializer
        if not tensor.external_data and np.prod(tensor.dims, dtype=np.int64) <= 1024
    }
    folded: dict[str, np.ndarray] = {}
    retained_nodes = []
    foldable = {"Add", "Cast", "Concat", "ConstantOfShape", "Div", "Equal", "Expand", "Gather", "Mul", "Unsqueeze", "Where"}

    def attribute(node: onnx.NodeProto, name: str) -> object:
        for item in node.attribute:
            if item.name == name:
                return helper.get_attribute_value(item)
        raise KeyError(name)

    for node in model.graph.node:
        result: np.ndarray | None = None
        if node.op_type == "Constant":
            try:
                result = numpy_helper.to_array(attribute(node, "value"))
            except KeyError:
                pass
        elif node.op_type == "Shape" and node.input[0] in shapes:
            source_shape = shapes[node.input[0]]
            start = int(attribute(node, "start")) if any(a.name == "start" for a in node.attribute) else 0
            end = int(attribute(node, "end")) if any(a.name == "end" for a in node.attribute) else len(source_shape)
            result = np.asarray(source_shape[start:end], dtype=np.int64)
        elif node.op_type in foldable and all(name in constants for name in node.input):
            inputs = [constants[name] for name in node.input]
            if sum(value.size for value in inputs) <= 4096:
                if node.op_type == "Add": result = inputs[0] + inputs[1]
                elif node.op_type == "Mul": result = inputs[0] * inputs[1]
                elif node.op_type == "Div": result = inputs[0] / inputs[1]
                elif node.op_type == "Cast": result = inputs[0].astype(helper.tensor_dtype_to_np_dtype(attribute(node, "to")))
                elif node.op_type == "Concat": result = np.concatenate(inputs, axis=int(attribute(node, "axis")))
                elif node.op_type == "ConstantOfShape":
                    value = numpy_helper.to_array(attribute(node, "value")).reshape(-1)[0] if node.attribute else np.float32(0)
                    result = np.full(tuple(int(v) for v in inputs[0]), value)
                elif node.op_type == "Equal": result = np.equal(inputs[0], inputs[1])
                elif node.op_type == "Expand": result = np.broadcast_to(inputs[0], tuple(int(v) for v in inputs[1])).copy()
                elif node.op_type == "Gather": result = np.take(inputs[0], inputs[1], axis=int(attribute(node, "axis")) if node.attribute else 0)
                elif node.op_type == "Unsqueeze": result = np.expand_dims(inputs[0], tuple(int(v) for v in inputs[1].reshape(-1)))
                elif node.op_type == "Where": result = np.where(inputs[0], inputs[1], inputs[2])

        if result is not None and result.size <= 65536 and len(node.output) == 1:
            if node.output[0] in dtypes:
                result = result.astype(dtypes[node.output[0]], copy=False)
            result = np.ascontiguousarray(result)
            constants[node.output[0]] = result
            folded[node.output[0]] = result
        else:
            retained_nodes.append(node)

    del model.graph.node[:]
    model.graph.node.extend(retained_nodes)
    existing_initializers = {tensor.name for tensor in model.graph.initializer}
    for name, value in folded.items():
        if name not in existing_initializers:
            model.graph.initializer.append(numpy_helper.from_array(value, name=name))
    return len(folded)


def concretize_fixed_shapes(model: onnx.ModelProto) -> int:
    """Propagate concrete dimensions from the fixed graph inputs."""
    values = {value.name: value for value in (*model.graph.input, *model.graph.output, *model.graph.value_info)}
    shapes: dict[str, tuple[int, ...]] = {
        tensor.name: tuple(int(d) for d in tensor.dims) for tensor in model.graph.initializer
    }
    for name, value in values.items():
        dimensions = value.type.tensor_type.shape.dim
        if dimensions and all(d.HasField("dim_value") and d.dim_value > 0 for d in dimensions):
            shapes[name] = tuple(int(d.dim_value) for d in dimensions)

    constants: dict[str, np.ndarray] = {}
    for tensor in model.graph.initializer:
        if np.prod(tensor.dims, dtype=np.int64) <= 4096:
            try:
                constants[tensor.name] = numpy_helper.to_array(tensor)
            except ValueError:
                pass

    def attrs(node: onnx.NodeProto) -> dict[str, object]:
        return {item.name: helper.get_attribute_value(item) for item in node.attribute}

    def broadcast(lhs: tuple[int, ...], rhs: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(np.broadcast_shapes(lhs, rhs))

    changed = 0
    passthrough = {"Cast", "Cos", "Div", "Erf", "InstanceNormalization", "LayerNormalization", "Mul", "Add", "Sigmoid", "Sin", "Softmax", "Sqrt"}
    for node in model.graph.node:
        inputs = [shapes.get(name) for name in node.input]
        output_shape: tuple[int, ...] | None = None
        properties = attrs(node)
        if node.op_type in passthrough and inputs and inputs[0]:
            output_shape = inputs[0]
            if node.op_type in {"Add", "Div", "Mul"} and len(inputs) > 1 and inputs[1]:
                output_shape = broadcast(inputs[0], inputs[1])
        elif node.op_type == "Conv" and inputs[0] and inputs[1]:
            pads = properties.get("pads", [0, 0, 0, 0])
            strides = properties.get("strides", [1, 1])
            dilations = properties.get("dilations", [1, 1])
            x, weight = inputs[0], inputs[1]
            height = (x[2] + pads[0] + pads[2] - dilations[0] * (weight[2] - 1) - 1) // strides[0] + 1
            width = (x[3] + pads[1] + pads[3] - dilations[1] * (weight[3] - 1) - 1) // strides[1] + 1
            output_shape = (x[0], weight[0], height, width)
        elif node.op_type == "Gemm" and inputs[0] and inputs[1]:
            output_shape = (inputs[0][1] if properties.get("transA", 0) else inputs[0][0],
                            inputs[1][0] if properties.get("transB", 0) else inputs[1][1])
        elif node.op_type == "MatMul" and inputs[0] and inputs[1]:
            output_shape = broadcast(inputs[0][:-2], inputs[1][:-2]) + (inputs[0][-2], inputs[1][-1])
        elif node.op_type == "Reshape" and len(node.input) > 1 and inputs[0] and node.input[1] in constants:
            target = [int(v) for v in constants[node.input[1]].reshape(-1)]
            for index, dimension in enumerate(target):
                if dimension == 0: target[index] = inputs[0][index]
            if -1 in target:
                known = int(np.prod([v for v in target if v != -1], dtype=np.int64))
                target[target.index(-1)] = int(np.prod(inputs[0], dtype=np.int64)) // known
            output_shape = tuple(target)
        elif node.op_type == "Transpose" and inputs[0]:
            permutation = properties.get("perm", list(reversed(range(len(inputs[0])))))
            output_shape = tuple(inputs[0][int(axis)] for axis in permutation)
        elif node.op_type == "Concat" and all(inputs):
            axis = int(properties["axis"]); axis %= len(inputs[0])
            result = list(inputs[0]); result[axis] = sum(shape[axis] for shape in inputs)
            output_shape = tuple(result)
        elif node.op_type == "Unsqueeze" and inputs[0] and node.input[1] in constants:
            result = list(inputs[0])
            for axis in sorted(int(v) % (len(result) + 1) for v in constants[node.input[1]].reshape(-1)):
                result.insert(axis, 1)
            output_shape = tuple(result)
        elif node.op_type == "Expand" and len(node.input) > 1 and node.input[1] in constants:
            output_shape = tuple(int(v) for v in constants[node.input[1]].reshape(-1))
        elif node.op_type == "Shape" and inputs[0]:
            output_shape = (len(inputs[0]),)
        elif node.op_type == "Resize" and len(node.input) > 3 and node.input[3] in constants:
            output_shape = tuple(int(v) for v in constants[node.input[3]].reshape(-1))
        elif node.op_type == "Slice" and inputs[0] and all(name in constants for name in node.input[1:]):
            result = list(inputs[0])
            starts = constants[node.input[1]].reshape(-1)
            ends = constants[node.input[2]].reshape(-1)
            axes = constants[node.input[3]].reshape(-1) if len(node.input) > 3 else np.arange(len(starts))
            steps = constants[node.input[4]].reshape(-1) if len(node.input) > 4 else np.ones(len(starts), dtype=np.int64)
            for start, end, axis, step in zip(starts, ends, axes, steps):
                axis = int(axis) % len(result)
                begin, stop, stride = slice(int(start), int(end), int(step)).indices(result[axis])
                result[axis] = max(0, (stop - begin + (stride - (1 if stride > 0 else -1))) // stride)
            output_shape = tuple(result)

        if output_shape is not None and len(node.output) == 1:
            shapes[node.output[0]] = output_shape
            if node.output[0] in values:
                dimensions = values[node.output[0]].type.tensor_type.shape.dim
                if len(dimensions) == len(output_shape):
                    for dimension, size in zip(dimensions, output_shape):
                        if not dimension.HasField("dim_value") or dimension.dim_value != size:
                            dimension.ClearField("dim_param")
                            dimension.dim_value = size
                            changed += 1
    return changed


def encode_prompt(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    prompt: str,
    negative_prompt: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = tokenizer(
        [negative_prompt, prompt],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        embeddings = text_encoder(
            tokens.input_ids, attention_mask=tokens.attention_mask
        )[0]
    # The legacy ONNX tracer temporarily enables autograd while tracing Linear
    # modules.  Clone outside inference mode so this fixture is a regular tensor
    # that the tracer is allowed to save for backward.
    return (
        tokens.input_ids.contiguous(),
        tokens.attention_mask.contiguous(),
        embeddings.clone().detach().contiguous(),
    )


def export_text_encoder(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    path: Path,
) -> None:
    raw_path = path.with_suffix(".raw.onnx")
    inferred_path = path.with_suffix(".inferred.onnx")
    weights_name = path.name + ".data"
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        str(raw_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )
    shape_inference.infer_shapes_path(str(raw_path), str(inferred_path), strict_mode=True)
    inferred = add_initializer_value_info(onnx.load(inferred_path))
    folded_count = fold_fixed_shape_expressions(inferred)
    concrete_count = concretize_fixed_shapes(inferred)
    folded_count += fold_fixed_shape_expressions(inferred)
    concrete_count += concretize_fixed_shapes(inferred)
    add_initializer_value_info(inferred)
    print(
        f"Folded {folded_count} fixed-shape CLIP ONNX nodes and "
        f"concretized {concrete_count} dimensions"
    )
    (path.parent / weights_name).unlink(missing_ok=True)
    onnx.save_model(
        inferred,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=weights_name,
        size_threshold=1024,
    )
    onnx.checker.check_model(str(path))
    raw_path.unlink()
    inferred_path.unlink()


def run_text_encoder_ort(
    path: Path, input_ids: np.ndarray, attention_mask: np.ndarray
) -> np.ndarray:
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return session.run(
        ["last_hidden_state"],
        {"input_ids": input_ids, "attention_mask": attention_mask},
    )[0]


def export(
    model: nn.Module,
    sample: torch.Tensor,
    timestep: torch.Tensor,
    embeddings: torch.Tensor,
    path: Path,
) -> None:
    raw_path = path.with_suffix(".raw.onnx")
    inferred_path = path.with_suffix(".inferred.onnx")
    weights_name = path.name + ".data"
    torch.onnx.export(
        model,
        (sample, timestep, embeddings),
        str(raw_path),
        input_names=["sample", "timestep", "encoder_hidden_states"],
        output_names=["noise_prediction"],
        opset_version=17,
        dynamo=False,
        do_constant_folding=True,
    )
    shape_inference.infer_shapes_path(str(raw_path), str(inferred_path), strict_mode=True)
    raw_graph = onnx.load(raw_path, load_external_data=False)
    external_files = {
        entry.value
        for tensor in raw_graph.graph.initializer
        for entry in tensor.external_data
        if entry.key == "location"
    }
    inferred = add_initializer_value_info(onnx.load(inferred_path))
    folded_count = fold_fixed_shape_expressions(inferred)
    concrete_count = concretize_fixed_shapes(inferred)
    folded_count += fold_fixed_shape_expressions(inferred)
    concrete_count += concretize_fixed_shapes(inferred)
    add_initializer_value_info(inferred)
    print(f"Folded {folded_count} fixed-shape ONNX nodes and concretized {concrete_count} dimensions")
    # ONNX appends to an existing external-data file. Remove only this
    # export's generated weight file so repeated exports remain deterministic.
    (path.parent / weights_name).unlink(missing_ok=True)
    onnx.save_model(
        inferred,
        str(path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=weights_name,
        size_threshold=1024,
    )
    onnx.checker.check_model(str(path))
    raw_path.unlink()
    inferred_path.unlink()
    for name in external_files:
        if name != weights_name:
            (path.parent / name).unlink(missing_ok=True)


def run_ort(path: Path, sample: np.ndarray, timestep: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return session.run(
        ["noise_prediction"],
        {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": embeddings,
        },
    )[0]


def require_finite(name: str, value: np.ndarray) -> None:
    if not np.isfinite(value).all():
        invalid = int(value.size - np.isfinite(value).sum())
        raise RuntimeError(f"{name} contains {invalid} non-finite values")


def validate_decoded_image(name: str, value: np.ndarray) -> None:
    require_finite(name, value)
    pixels = np.clip((value + 1.0) * 127.5, 0.0, 255.0)
    dynamic_range = float(np.ptp(pixels))
    standard_deviation = float(np.std(pixels))
    print(
        f"{name} pixel sanity: min={pixels.min():.2f}, max={pixels.max():.2f}, "
        f"range={dynamic_range:.2f}, std={standard_deviation:.2f}"
    )
    if dynamic_range < 16.0 or standard_deviation < 2.0:
        raise RuntimeError(f"{name} is effectively blank or constant")


def write_operator_report(path: Path, model_path: Path) -> set[str]:
    model = onnx.load(model_path, load_external_data=False)
    operations = Counter(node.op_type for node in model.graph.node)
    unsupported = set(operations) - KLARTRAUM_OPERATIONS
    lines = [
        f"nodes={len(model.graph.node)}",
        f"size_mib={model_path.stat().st_size / (1024 * 1024):.1f}",
        "operations=" + ", ".join(f"{name}:{count}" for name, count in sorted(operations.items())),
        "unsupported=" + ", ".join(sorted(unsupported)),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return unsupported


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--reuse-models",
        action="store_true",
        help="reuse compatible fixed-size ONNX files and only regenerate reference fixtures",
    )
    parser.add_argument(
        "--text-encoder-only",
        action="store_true",
        help="export and validate CLIP/tokenizer assets without running the image pipeline",
    )
    return parser.parse_args()


def main() -> None:
    print_stable_diffusion_notice()
    args = parse_args()
    if args.size <= 0 or args.size % 8:
        raise ValueError("--size must be a positive multiple of 8")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    args.onnx_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    text_encoder_model = CLIPTextModel.from_pretrained(
        args.model, subfolder="text_encoder"
    ).eval().cpu()
    clip_started = time.perf_counter()
    input_ids, attention_mask, embeddings = encode_prompt(
        tokenizer, text_encoder_model, args.prompt, args.negative_prompt
    )
    clip_seconds = time.perf_counter() - clip_started
    print(f"PyTorch CLIP inference: {clip_seconds:.3f} s")
    tokenizer.save_pretrained(args.onnx_dir)
    text_encoder_path = args.onnx_dir / "sd15_text_encoder.onnx"
    if not args.reuse_models or not text_encoder_path.exists():
        export_text_encoder(
            TextEncoder(text_encoder_model).eval(),
            input_ids,
            attention_mask,
            text_encoder_path,
        )
    clip_output = run_text_encoder_ort(
        text_encoder_path, input_ids.numpy(), attention_mask.numpy()
    )
    clip_error = float(np.max(np.abs(clip_output - embeddings.numpy())))
    print(f"ONNX CLIP text encoder maximum absolute error: {clip_error:.7g}")
    if not np.allclose(clip_output, embeddings.numpy(), atol=2e-3, rtol=2e-3):
        raise RuntimeError("ONNX CLIP text encoder does not match PyTorch")
    input_ids.numpy().astype(np.int64).tofile(args.onnx_dir / "prompt_input_ids_i64.bin")
    attention_mask.numpy().astype(np.int64).tofile(
        args.onnx_dir / "prompt_attention_mask_i64.bin"
    )
    embeddings.numpy().astype(np.float32).tofile(args.onnx_dir / "prompt_embeddings_f32.bin")
    clip_output.astype(np.float32).tofile(args.onnx_dir / "text_encoder_reference_f32.bin")
    clip_unsupported = write_operator_report(
        args.output_dir / "text_encoder_operator_report.txt", text_encoder_path
    )
    if args.text_encoder_only:
        print(
            "Unsupported Klartraum CLIP operations "
            f"({len(clip_unsupported)}): {', '.join(sorted(clip_unsupported))}"
        )
        print(f"Python inference total: {clip_seconds:.3f} s")
        return
    del text_encoder_model, tokenizer
    gc.collect()

    unet = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet").eval().cpu()
    pndm_scheduler = PNDMScheduler.from_pretrained(args.model, subfolder="scheduler")
    scheduler = DDIMScheduler.from_config(pndm_scheduler.config)
    scheduler.set_timesteps(args.steps)

    latent_size = args.size // 8
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    latents = torch.randn((1, 4, latent_size, latent_size), generator=generator)
    latents *= scheduler.init_noise_sigma
    denoiser = Denoiser(unet).eval()
    initial_latents = latents.clone()
    first_model_sample = None
    first_timestep_input = None
    first_torch_noise = None
    first_guided_noise = None
    denoise_started = time.perf_counter()
    step_seconds = []
    with torch.inference_mode():
        for step_index, timestep in enumerate(scheduler.timesteps):
            step_started = time.perf_counter()
            model_sample = scheduler.scale_model_input(torch.cat([latents, latents]), timestep)
            timestep_input = timestep.reshape(1)
            torch_noise = denoiser(model_sample, timestep_input, embeddings)
            unconditional, conditional = torch_noise.chunk(2)
            guided_noise = unconditional + args.guidance_scale * (conditional - unconditional)
            latents = scheduler.step(guided_noise, timestep, latents).prev_sample
            if first_model_sample is None:
                first_model_sample = model_sample.clone()
                first_timestep_input = timestep_input.clone()
                first_torch_noise = torch_noise.clone()
                first_guided_noise = guided_noise.clone()
            elapsed = time.perf_counter() - step_started
            step_seconds.append(elapsed)
            print(
                f"PyTorch DDIM step {step_index + 1}/{args.steps}: "
                f"{elapsed:.3f} s (t={int(timestep)})"
            )
    denoise_seconds = time.perf_counter() - denoise_started

    assert first_model_sample is not None
    assert first_timestep_input is not None
    assert first_torch_noise is not None
    assert first_guided_noise is not None
    # The legacy ONNX tracer enables autograd while tracing modules. Convert
    # values produced under inference_mode back to ordinary tensors first.
    first_model_sample = torch.from_numpy(first_model_sample.numpy().copy())
    first_timestep_input = torch.from_numpy(first_timestep_input.numpy().copy())

    vae = AutoencoderKL.from_pretrained(
        args.model, subfolder="vae", torch_dtype=torch.float32
    ).eval().cpu()
    decoder = VaeDecoder(vae).eval()
    decoder_input = latents / vae.config.scaling_factor
    decode_started = time.perf_counter()
    with torch.inference_mode():
        decoded = decoder(decoder_input)
    decode_seconds = time.perf_counter() - decode_started
    inference_seconds = clip_seconds + denoise_seconds + decode_seconds
    print(
        "Python inference timing: "
        f"CLIP={clip_seconds:.3f} s, "
        f"DDIM UNet={denoise_seconds:.3f} s "
        f"({denoise_seconds / len(step_seconds):.3f} s/step), "
        f"VAE decode={decode_seconds:.3f} s, total={inference_seconds:.3f} s"
    )
    decoder_input_for_export = torch.from_numpy(decoder_input.numpy().copy())

    model_path = args.onnx_dir / "sd15_unet.onnx"
    decoder_path = args.onnx_dir / "sd15_vae_decoder.onnx"
    if args.reuse_models:
        for path in (model_path, decoder_path):
            if not path.exists():
                raise RuntimeError(f"Cannot reuse missing model: {path}")
        print("Reusing existing fixed-size ONNX models")
    else:
        export(
            denoiser,
            first_model_sample,
            first_timestep_input,
            embeddings,
            model_path,
        )
        export_vae_component(decoder, decoder_input_for_export, decoder_path)
    unsupported = write_operator_report(args.output_dir / "operator_report.txt", model_path)

    sample_array = first_model_sample.numpy()
    timestep_array = first_timestep_input.numpy()
    embedding_array = embeddings.numpy()
    ort_noise = run_ort(model_path, sample_array, timestep_array, embedding_array)
    require_finite("PyTorch UNet output", first_torch_noise.numpy())
    require_finite("ONNX UNet output", ort_noise)
    maximum_error = float(np.max(np.abs(ort_noise - first_torch_noise.numpy())))
    print(f"ONNX UNet maximum absolute error: {maximum_error:.7g}")
    if not np.allclose(ort_noise, first_torch_noise.numpy(), atol=2e-3, rtol=2e-3):
        raise RuntimeError("ONNX UNet does not match PyTorch")

    ort_decoded = run_vae_ort(decoder_path, decoder_input.numpy())
    validate_decoded_image("PyTorch decoded image", decoded.numpy())
    validate_decoded_image("ONNX decoded image", ort_decoded)
    decoder_error = float(np.max(np.abs(ort_decoded - decoded.numpy())))
    print(f"ONNX VAE decoder maximum absolute error: {decoder_error:.7g}")
    if not np.allclose(ort_decoded, decoded.numpy(), atol=2e-3, rtol=2e-3):
        raise RuntimeError("ONNX VAE decoder does not match PyTorch")

    timestep_values = scheduler.timesteps.cpu().numpy().astype(np.int64)
    alpha_pairs = []
    step_size = scheduler.config.num_train_timesteps // args.steps
    for timestep_value in timestep_values:
        previous = int(timestep_value) - step_size
        alpha_current = scheduler.alphas_cumprod[int(timestep_value)]
        alpha_previous = (
            scheduler.alphas_cumprod[previous]
            if previous >= 0
            else scheduler.final_alpha_cumprod
        )
        alpha_pairs.append([float(alpha_current), float(alpha_previous)])

    sample_array.astype(np.float32).tofile(args.onnx_dir / "unet_sample_f32.bin")
    timestep_array.astype(np.int64).tofile(args.onnx_dir / "unet_timestep_i64.bin")
    embedding_array.astype(np.float32).tofile(args.onnx_dir / "prompt_embeddings_f32.bin")
    input_ids.numpy().astype(np.int64).tofile(args.onnx_dir / "prompt_input_ids_i64.bin")
    clip_output.astype(np.float32).tofile(args.onnx_dir / "text_encoder_reference_f32.bin")
    ort_noise.astype(np.float32).tofile(args.onnx_dir / "unet_reference_f32.bin")
    first_guided_noise.numpy().astype(np.float32).tofile(args.onnx_dir / "guided_noise_f32.bin")
    initial_latents.numpy().astype(np.float32).tofile(args.onnx_dir / "initial_latents_f32.bin")
    timestep_values.tofile(args.onnx_dir / "scheduler_timesteps_i64.bin")
    np.asarray(alpha_pairs, dtype=np.float32).tofile(
        args.onnx_dir / "scheduler_alphas_f32.bin"
    )
    latents.numpy().astype(np.float32).tofile(args.onnx_dir / "final_latents_f32.bin")
    decoder_input.numpy().astype(np.float32).tofile(args.onnx_dir / "decoder_input_f32.bin")
    ort_decoded.astype(np.float32).tofile(args.onnx_dir / "pipeline_reference_f32.bin")
    save_image(torch.from_numpy(ort_decoded), args.output_dir / "sd15_pipeline_reference.png")
    (args.output_dir / "prompt.txt").write_text(
        f"prompt={args.prompt}\nnegative_prompt={args.negative_prompt}\n",
        encoding="utf-8",
    )
    print(f"Unsupported Klartraum operations ({len(unsupported)}): {', '.join(sorted(unsupported))}")
    print(f"Wrote denoiser fixtures to {args.onnx_dir}")


if __name__ == "__main__":
    main()
