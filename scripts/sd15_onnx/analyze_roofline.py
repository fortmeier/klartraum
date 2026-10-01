"""Estimate fixed-shape SD1.5 ONNX roofline lower bounds.

The FLOP count covers the dominant multiply-accumulate operators (Conv,
ConvTranspose, MatMul, and Gemm). The traffic count assumes every dispatched
operator reads each input and writes each output once. Klartraum's fused
attention patterns are counted without materializing their score matrices.
These are optimistic lower bounds, not predicted execution times.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import onnx
from onnx import TensorProto

from disclaimer import print_stable_diffusion_notice


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

TYPE_BYTES = {
    TensorProto.FLOAT: 4,
    TensorProto.UINT8: 1,
    TensorProto.INT8: 1,
    TensorProto.UINT16: 2,
    TensorProto.INT16: 2,
    TensorProto.INT32: 4,
    TensorProto.INT64: 8,
    TensorProto.BOOL: 1,
    TensorProto.FLOAT16: 2,
    TensorProto.DOUBLE: 8,
    TensorProto.UINT32: 4,
    TensorProto.UINT64: 8,
    TensorProto.BFLOAT16: 2,
}

METADATA_OPERATIONS = {"Constant", "Reshape", "Unsqueeze"}
CONDITIONAL_ALIASES = {"Cast", "Slice"}


@dataclass
class ModelCost:
    fma_flops: int
    operator_bytes: int
    compulsory_bytes: int
    fused_attention_count: int

    def scaled(self, multiplier: int) -> "ModelCost":
        return ModelCost(
            self.fma_flops * multiplier,
            self.operator_bytes * multiplier,
            self.compulsory_bytes * multiplier,
            self.fused_attention_count * multiplier,
        )


def value_shape(value: onnx.ValueInfoProto) -> tuple[int, ...] | None:
    tensor = value.type.tensor_type
    if not tensor.HasField("shape"):
        return None
    dimensions: list[int] = []
    for dimension in tensor.shape.dim:
        if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
            return None
        dimensions.append(dimension.dim_value)
    return tuple(dimensions)


def load_metadata(model_path: Path):
    model = onnx.load(model_path, load_external_data=False)
    graph = model.graph
    shapes: dict[str, tuple[int, ...]] = {}
    types: dict[str, int] = {}

    for value in (*graph.input, *graph.value_info, *graph.output):
        shape = value_shape(value)
        if shape is not None:
            shapes[value.name] = shape
        types[value.name] = value.type.tensor_type.elem_type
    for initializer in graph.initializer:
        shapes[initializer.name] = tuple(initializer.dims)
        types[initializer.name] = initializer.data_type
    return model, shapes, types


def tensor_bytes(name: str, shapes: dict[str, tuple[int, ...]], types: dict[str, int]) -> int:
    shape = shapes.get(name)
    data_type = types.get(name)
    if shape is None or data_type not in TYPE_BYTES:
        return 0
    return math.prod(shape) * TYPE_BYTES[data_type]


def dominant_flops(node: onnx.NodeProto, shapes: dict[str, tuple[int, ...]]) -> int:
    if not node.output or node.output[0] not in shapes:
        return 0
    output_elements = math.prod(shapes[node.output[0]])

    if node.op_type == "Conv" and len(node.input) >= 2:
        weight_shape = shapes.get(node.input[1])
        if weight_shape is not None and len(weight_shape) >= 3:
            return 2 * output_elements * math.prod(weight_shape[1:])

    if node.op_type == "ConvTranspose" and len(node.input) >= 2:
        input_shape = shapes.get(node.input[0])
        weight_shape = shapes.get(node.input[1])
        if input_shape is not None and weight_shape is not None and len(weight_shape) >= 3:
            # Each input value contributes to Cout/group times every kernel position.
            macs = math.prod(input_shape) * math.prod(weight_shape[1:])
            return 2 * macs

    if node.op_type in {"MatMul", "Gemm"} and len(node.input) >= 2:
        left_shape = shapes.get(node.input[0])
        if left_shape is not None and len(left_shape) >= 2:
            return 2 * output_elements * left_shape[-1]

    return 0


def find_fused_attention(graph: onnx.GraphProto) -> dict[int, tuple[int, int]]:
    producers: dict[str, int] = {}
    consumers: dict[str, list[int]] = {}
    for index, node in enumerate(graph.node):
        for output in node.output:
            producers[output] = index
        for input_name in node.input:
            consumers.setdefault(input_name, []).append(index)

    fused: dict[int, tuple[int, int]] = {}
    for softmax_index, softmax in enumerate(graph.node):
        if softmax.op_type != "Softmax" or not softmax.input or not softmax.output:
            continue
        first_index = producers.get(softmax.input[0])
        second_users = consumers.get(softmax.output[0], [])
        if first_index is None or len(second_users) != 1:
            continue
        second_index = second_users[0]
        if graph.node[first_index].op_type == "MatMul" and graph.node[second_index].op_type == "MatMul":
            fused[first_index] = (softmax_index, second_index)
    return fused


def analyze(model_path: Path) -> ModelCost:
    model, shapes, types = load_metadata(model_path)
    graph = model.graph
    fused = find_fused_attention(graph)
    fused_members = {member for pair in fused.values() for member in pair}

    fma_flops = sum(dominant_flops(node, shapes) for node in graph.node)
    operator_bytes = 0
    for index, node in enumerate(graph.node):
        if index in fused_members:
            continue
        if index in fused:
            _, second_index = fused[index]
            second = graph.node[second_index]
            tensor_names = [*node.input, second.input[1], *second.output]
            operator_bytes += sum(tensor_bytes(name, shapes, types) for name in tensor_names)
            continue
        if node.op_type in METADATA_OPERATIONS:
            continue
        if node.op_type in CONDITIONAL_ALIASES and node.input and node.output:
            input_bytes = tensor_bytes(node.input[0], shapes, types)
            output_bytes = tensor_bytes(node.output[0], shapes, types)
            if input_bytes == output_bytes and types.get(node.input[0]) == types.get(node.output[0]):
                continue
        operator_bytes += sum(tensor_bytes(name, shapes, types) for name in node.input)
        operator_bytes += sum(tensor_bytes(name, shapes, types) for name in node.output)

    initializer_names = {initializer.name for initializer in graph.initializer}
    compulsory_bytes = sum(
        tensor_bytes(initializer.name, shapes, types) for initializer in graph.initializer
    )
    compulsory_bytes += sum(
        tensor_bytes(value.name, shapes, types)
        for value in graph.input
        if value.name not in initializer_names
    )
    compulsory_bytes += sum(tensor_bytes(value.name, shapes, types) for value in graph.output)

    return ModelCost(fma_flops, operator_bytes, compulsory_bytes, len(fused))


def format_row(
    stage: str,
    cost: ModelCost,
    fp32_tflops: float,
    bandwidth_gbps: float,
    measured_seconds: float | None,
) -> str:
    compute_ms = cost.fma_flops / (fp32_tflops * 1e12) * 1e3
    bandwidth_ms = cost.operator_bytes / (bandwidth_gbps * 1e9) * 1e3
    lower_bound_ms = max(compute_ms, bandwidth_ms)
    efficiency = "-"
    headroom = "-"
    if measured_seconds is not None:
        efficiency = f"{lower_bound_ms / (measured_seconds * 1e3) * 100:.2f}%"
        headroom = f"{measured_seconds * 1e3 / lower_bound_ms:.2f}x"
    return (
        f"| {stage} | {cost.fma_flops / 1e9:.3f} GFLOP | "
        f"{cost.operator_bytes / 1e9:.3f} GB | {compute_ms:.3f} ms | "
        f"{bandwidth_ms:.3f} ms | {lower_bound_ms:.3f} ms | {efficiency} | {headroom} |"
    )


def main() -> None:
    print_stable_diffusion_notice()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REPO_ROOT / "data" / "onnx" / "sd15_denoiser_512",
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--fp32-tflops", type=float, default=9.0624)
    parser.add_argument("--memory-bandwidth-gbps", type=float, default=448.0)
    parser.add_argument("--clip-seconds", type=float, default=0.0503837)
    parser.add_argument("--unet-seconds", type=float, default=101.797)
    parser.add_argument("--vae-seconds", type=float, default=4.01977)
    args = parser.parse_args()

    clip = analyze(args.model_dir / "sd15_text_encoder.onnx")
    unet = analyze(args.model_dir / "sd15_unet.onnx")
    vae = analyze(args.model_dir / "sd15_vae_decoder.onnx")
    stages = [
        ("CLIP", clip, args.clip_seconds),
        ("UNet average step", unet, args.unet_seconds / args.steps),
        (f"UNet {args.steps} steps", unet.scaled(args.steps), args.unet_seconds),
        ("VAE decode", vae, args.vae_seconds),
    ]
    analyzed: list[tuple[str, ModelCost, float | None]] = stages

    total = ModelCost(0, 0, 0, 0)
    for stage, cost, _ in analyzed:
        if stage in {"CLIP", f"UNet {args.steps} steps", "VAE decode"}:
            total.fma_flops += cost.fma_flops
            total.operator_bytes += cost.operator_bytes
            total.compulsory_bytes += cost.compulsory_bytes
            total.fused_attention_count += cost.fused_attention_count
    total_seconds = args.clip_seconds + args.unet_seconds + args.vae_seconds
    analyzed.append(("Full pipeline", total, total_seconds))

    print(f"FP32 peak: {args.fp32_tflops:.4f} TFLOP/s")
    print(f"Memory bandwidth: {args.memory_bandwidth_gbps:.1f} GB/s")
    print("| Stage | Dominant FLOPs | Operator traffic | Compute floor | Bandwidth floor | Roofline floor | Measured efficiency | Remaining headroom |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for stage, cost, measured in analyzed:
        print(format_row(stage, cost, args.fp32_tflops, args.memory_bandwidth_gbps, measured))


if __name__ == "__main__":
    main()
