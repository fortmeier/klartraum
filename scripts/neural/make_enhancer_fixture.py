# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "onnx", "pillow"]
# ///

"""Writes the fixture of the tree enhancer test (tests/test_tree_enhancer.cpp) to data/onnx/:

- tree_enhancer_fixture.onnx: the Enhancer of train_tree_enhancer.py at 96 x 64 with fixed random weights (the
  output layer too, so that the network changes the image);
- tree_enhancer_fixture_images.bin: its input as three BGRA8 images of 96 x 64 (colour, feature A, feature B),
  random bytes;
- tree_enhancer_fixture_output.bin: PyTorch's output for them, float32 NCHW (1 x 3 x 64 x 96).

    uv run scripts/neural/make_enhancer_fixture.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_tree_enhancer import Enhancer, constants_as_initializers  # noqa: E402

WIDTH, HEIGHT = 96, 64


def main() -> None:
    out = Path(__file__).resolve().parents[2] / "data" / "onnx"
    torch.manual_seed(5)
    model = Enhancer().eval()
    with torch.no_grad():
        for p in model.out.parameters():
            p.normal_(0.0, 0.2)
    rng = np.random.default_rng(5)
    images = rng.integers(0, 256, size=(3, HEIGHT, WIDTH, 4), dtype=np.uint8)
    images[..., 3] = 255
    # BGRA bytes to the network's planes: colour RGB, feature A RGB, feature B RG.
    rgb = images[..., [2, 1, 0]].astype(np.float32) / 255.0
    planes = np.concatenate([rgb[0], rgb[1], rgb[2][..., :2]], axis=-1).transpose(2, 0, 1)[None]
    with torch.no_grad():
        output = model(torch.from_numpy(planes)).numpy().astype(np.float32)
    torch.onnx.export(model, torch.from_numpy(planes), out / "tree_enhancer_fixture.onnx", input_names=["input"],
                      output_names=["output"], opset_version=17, dynamo=False)
    # OnnxNetwork needs the shapes of the intermediate tensors.
    import onnx
    path = out / "tree_enhancer_fixture.onnx"
    onnx.save(onnx.shape_inference.infer_shapes(constants_as_initializers(onnx.load(path))), path)
    images.tofile(out / "tree_enhancer_fixture_images.bin")
    output.tofile(out / "tree_enhancer_fixture_output.bin")
    print(f"fixture in {out}; output range [{output.min():.3f}, {output.max():.3f}]")


if __name__ == "__main__":
    main()
