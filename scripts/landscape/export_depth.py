# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Exports Depth Anything V2 Metric Outdoor Small (Apache-2.0; metres, at most 80) as a fixed-shape float32 ONNX graph
for Klartraum: image [1, 3, H, W] in [0, 1] (the ImageNet normalisation is part of the graph) -> depth [1, 1, H, W]
in metres. H and W are multiples of 14 (default 518 x 924, 16:9). Checks the graph against transformers and ONNX
Runtime and stores a reference input and output next to it for tests/test_landscape_networks.cpp.

    uv run --project scripts/qwen_image python scripts/landscape/export_depth.py [--size 518 924] [--image photo.png]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image
from transformers import AutoModelForDepthEstimation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qwen_image"))
from export_vae import add_initializer_value_info, fold_small_constants, remove_dead_nodes, require_concrete_shapes  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "build/models/depth/da2-metric-outdoor-small"


def make_shapes_static(graph: onnx.ModelProto, rounds: int = 100) -> onnx.ModelProto:
    """Replaces Shape nodes whose input shape is known by constants and folds what depends on them, until no Shape
    node is left that could be resolved (the tracer computes reshapes from the input's shape at run time)."""
    for _ in range(rounds):
        graph = onnx.shape_inference.infer_shapes(graph)
        shapes = {v.name: [d.dim_value if d.HasField("dim_value") else None for d in v.type.tensor_type.shape.dim]
                  for v in (*graph.graph.input, *graph.graph.value_info, *graph.graph.output)}
        replaced, kept = 0, []
        for node in graph.graph.node:
            dims = shapes.get(node.input[0]) if node.op_type == "Shape" else None
            if dims is not None and all(d is not None for d in dims):
                start = next((a.i for a in node.attribute if a.name == "start"), 0)
                end = next((a.i for a in node.attribute if a.name == "end"), len(dims))
                value = np.array(dims, dtype=np.int64)[start:end]
                graph.graph.initializer.append(onnx.numpy_helper.from_array(value, name=node.output[0]))
                replaced += 1
            else:
                kept.append(node)
        del graph.graph.node[:]
        graph.graph.node.extend(kept)
        folded = fold_small_constants(graph)
        if replaced == 0 and folded == 0:
            break
    # Weight-only subgraphs (e.g. the bicubically resized position embedding) fold as well, whatever their size.
    fold_small_constants(graph, limit=1 << 26)
    # Shapes are inferred again from scratch; entries of tensors that became initializers would stay incomplete.
    del graph.graph.value_info[:]
    return graph


def constants_to_initializers(graph: onnx.ModelProto) -> int:
    """Moves Constant nodes into initializers, where Klartraum's importer reads operator parameters from."""
    kept, moved = [], 0
    for node in graph.graph.node:
        value = next((a.t for a in node.attribute if a.name == "value" and a.HasField("t")), None)
        if node.op_type == "Constant" and value is not None:
            tensor = onnx.numpy_helper.from_array(onnx.numpy_helper.to_array(value), name=node.output[0])
            graph.graph.initializer.append(tensor)
            moved += 1
        else:
            kept.append(node)
    del graph.graph.node[:]
    graph.graph.node.extend(kept)
    return moved


def squeezes_to_reshapes(graph: onnx.ModelProto) -> int:
    """Replaces Squeeze and Unsqueeze (which Klartraum lacks) by Reshape to their inferred static output shapes."""
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in (*graph.graph.value_info, *graph.graph.output)}
    replaced = 0
    for node in graph.graph.node:
        if node.op_type in ("Squeeze", "Unsqueeze") and node.output[0] in shapes:
            shape_name = node.output[0] + "_shape"
            graph.graph.initializer.append(
                onnx.numpy_helper.from_array(np.array(shapes[node.output[0]], dtype=np.int64), name=shape_name))
            node.op_type = "Reshape"
            del node.input[1:]
            node.input.append(shape_name)
            del node.attribute[:]
            replaced += 1
    return replaced


class Depth(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        depth = self.model(pixel_values=(image - self.mean) / self.std).predicted_depth
        return depth.unsqueeze(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, nargs=2, default=(518, 924), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--image", type=Path, help="reference image (default: a Qwen fly-in preview)")
    parser.add_argument("--out", type=Path, default=ROOT / "build/models/depth/onnx")
    args = parser.parse_args()
    height, width = args.size
    if height % 14 or width % 14:
        raise SystemExit("the size must be a multiple of 14")
    args.out.mkdir(parents=True, exist_ok=True)

    model = AutoModelForDepthEstimation.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    wrapper = Depth(model).eval()
    source = args.image or ROOT / "build/TestingOutput/qwen_flyin/preview/styled_0399.png"
    picture = Image.open(source).convert("RGB").resize((width, height), Image.BICUBIC)
    image = torch.from_numpy(np.asarray(picture, dtype=np.float32) / 255.0).permute(2, 0, 1)[None].contiguous()
    with torch.no_grad():
        depth = wrapper(image)
    print(f"  depth of {source.name}: {depth.min():.2f} .. {depth.max():.2f} m")

    name = f"depth_anything_v2_metric_outdoor_small_{height}x{width}"
    path = args.out / f"{name}.onnx"
    with torch.no_grad():
        torch.onnx.export(wrapper, (image,), str(path), input_names=["image"], output_names=["depth"],
                          opset_version=17, dynamo=False, do_constant_folding=True)
    graph = make_shapes_static(onnx.load(path))
    graph = onnx.shape_inference.infer_shapes(graph, strict_mode=True)
    squeezes_to_reshapes(graph)
    constants_to_initializers(graph)
    remove_dead_nodes(graph)
    add_initializer_value_info(graph)
    require_concrete_shapes(graph)
    onnx.save_model(graph, str(path))
    print(f"  operators: {dict(sorted(Counter(n.op_type for n in graph.graph.node).items()))}")

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    result = ort.InferenceSession(str(path), options).run(None, {"image": image.numpy()})[0]
    print(f"  ONNX Runtime vs PyTorch: max |diff| {np.abs(result - depth.numpy()).max():.3e} m")
    image.numpy().tofile(args.out / f"{name}_input_f32.bin")
    depth.numpy().tofile(args.out / f"{name}_output_f32.bin")


if __name__ == "__main__":
    main()
