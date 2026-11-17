# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Exports UperNet with a ConvNeXt-Base backbone trained on ADE20K (openmmlab/upernet-convnext-base, MIT; 150
classes) as a fixed-shape float32 ONNX graph for Klartraum: image [1, 3, H, W] in [0, 1] (the ImageNet normalisation
is part of the graph) -> class logits [1, 150, H/4, W/4]; the class of a pixel is the argmax of its bilinearly
upsampled logits. H and W are multiples of 32 (default 512 x 896, 16:9).

The pyramid pooling module's adaptive average pooling (which neither exports for sizes that do not divide nor
exists in Klartraum) is written as a product with a fixed pooling matrix, which computes the same averages. Checks
the graph against transformers and ONNX Runtime and stores a reference input and output for
tests/test_landscape_networks.cpp.

    uv run --project scripts/qwen_image python scripts/landscape/export_semseg.py [--size 512 896] [--image photo.png]
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
from transformers import UperNetForSemanticSegmentation

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qwen_image"))
from export_depth import constants_to_initializers, make_shapes_static, squeezes_to_reshapes  # noqa: E402
from export_vae import add_initializer_value_info, remove_dead_nodes, require_concrete_shapes  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "build/models/semseg/upernet-convnext-base"


def pooling_matrix(size: int, bins: int) -> torch.Tensor:
    """[bins][size] weights of adaptive average pooling along one axis (PyTorch's bin edges)."""
    matrix = torch.zeros(bins, size)
    for b in range(bins):
        start, end = (b * size) // bins, -((-(b + 1) * size) // bins)
        matrix[b, start:end] = 1.0 / (end - start)
    return matrix


class MatrixPooling(torch.nn.Module):
    """AdaptiveAvgPool2d(bins) of an [N, C, H, W] input of fixed H and W as one matrix product."""

    def __init__(self, bins: int, height: int, width: int):
        super().__init__()
        self.bins, self.height, self.width = bins, height, width
        rows, columns = pooling_matrix(height, bins), pooling_matrix(width, bins)
        # weights[i * bins + j][y * width + x] = rows[i][y] * columns[j][x]
        self.register_buffer("weights", torch.einsum("iy,jx->ijyx", rows, columns).reshape(bins * bins, -1).T.contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c = int(x.shape[0]), int(x.shape[1])
        pooled = torch.matmul(x.reshape(n, c, self.height * self.width), self.weights)
        return pooled.reshape(n, c, self.bins, self.bins)


class Segmentation(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        pixel_values = (image - self.mean) / self.std
        features = self.model.backbone.forward_with_filtered_kwargs(pixel_values).feature_maps
        return self.model.decode_head(features)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--size", type=int, nargs=2, default=(512, 896), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--image", type=Path, help="reference image (default: a Qwen fly-in preview)")
    parser.add_argument("--out", type=Path, default=ROOT / "build/models/semseg/onnx")
    args = parser.parse_args()
    height, width = args.size
    if height % 32 or width % 32:
        raise SystemExit("the size must be a multiple of 32")
    args.out.mkdir(parents=True, exist_ok=True)

    model = UperNetForSemanticSegmentation.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    wrapper = Segmentation(model).eval()
    source = args.image or ROOT / "build/TestingOutput/qwen_flyin/preview/styled_0399.png"
    picture = Image.open(source).convert("RGB").resize((width, height), Image.BICUBIC)
    image = torch.from_numpy(np.asarray(picture, dtype=np.float32) / 255.0).permute(2, 0, 1)[None].contiguous()
    with torch.no_grad():
        expected = model(pixel_values=(image - wrapper.mean) / wrapper.std).logits  # transformers, full size
        for block in model.decode_head.psp_modules.blocks:
            pool = block.layers[0]
            block.layers[0] = MatrixPooling(pool.output_size, height // 32, width // 32)
            block._modules["0"] = block.layers[0]
        logits = wrapper(image)
        upsampled = torch.nn.functional.interpolate(logits, size=(height, width), mode="bilinear",
                                                    align_corners=model.config.align_corners if hasattr(model.config, "align_corners") else False)
    agreement = (upsampled.argmax(1) == expected.argmax(1)).float().mean()
    classes = Counter(model.config.id2label[int(c)] for c in expected.argmax(1).flatten().tolist())
    print(f"  matrix pooling vs transformers: max |diff| {(upsampled - expected).abs().max():.3e}, "
          f"argmax agreement {agreement * 100:.2f} %")
    print(f"  classes of {source.name}: {dict(classes.most_common(6))}")

    name = f"upernet_convnext_base_ade20k_{height}x{width}"
    path = args.out / f"{name}.onnx"
    with torch.no_grad():
        torch.onnx.export(wrapper, (image,), str(path), input_names=["image"], output_names=["logits"],
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
    print(f"  ONNX Runtime vs PyTorch: max |diff| {np.abs(result - logits.numpy()).max():.3e}")
    image.numpy().tofile(args.out / f"{name}_input_f32.bin")
    logits.numpy().tofile(args.out / f"{name}_output_f32.bin")


if __name__ == "__main__":
    main()
