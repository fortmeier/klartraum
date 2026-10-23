# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "onnx", "pillow"]
# ///

"""Writes the fixtures of the tree enhancer test (tests/test_tree_enhancer.cpp) to data/onnx/, one per network
of train_tree_enhancer.py (v1: tree_enhancer_fixture*, v2: tree_enhancer_v2_fixture*, v3: tree_enhancer_v3_fixture*,
v6: tree_enhancer_v6_fixture*, which also reads feature B's third channel, the splatted noise):

- *.onnx: the network at 96 x 64 with fixed random weights (the output layer too, so that it changes the image);
- *_images.bin: its input as three BGRA8 images of 96 x 64 (colour, feature A, feature B), random bytes;
- *_output.bin: PyTorch's output for them, float32 NCHW (1 x 3 x 64 x 96).

    uv run scripts/neural/make_enhancer_fixture.py [v1 v2 v3]   # default all
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_tree_enhancer import ARCHITECTURES, constants_as_initializers  # noqa: E402

WIDTH, HEIGHT = 96, 64


def write(arch: str, stem: str) -> None:
    out = Path(__file__).resolve().parents[2] / "data" / "onnx"
    torch.manual_seed(5)
    model = ARCHITECTURES[arch]().eval()
    with torch.no_grad():
        for p in model.out.parameters():
            p.normal_(0.0, 0.2)
    rng = np.random.default_rng(5)
    images = rng.integers(0, 256, size=(3, HEIGHT, WIDTH, 4), dtype=np.uint8)
    images[..., 3] = 255
    # BGRA bytes to the network's planes: colour RGB, feature A RGB, feature B RG (and B for the splatted noise).
    rgb = images[..., [2, 1, 0]].astype(np.float32) / 255.0
    feature_b = rgb[2] if getattr(model, "splatted_noise", False) else rgb[2][..., :2]
    planes = np.concatenate([rgb[0], rgb[1], feature_b], axis=-1).transpose(2, 0, 1)[None]
    with torch.no_grad():
        output = model(torch.from_numpy(planes)).numpy().astype(np.float32)
    torch.onnx.export(model, torch.from_numpy(planes), out / f"{stem}.onnx", input_names=["input"],
                      output_names=["output"], opset_version=17, dynamo=False)
    # OnnxNetwork needs the shapes of the intermediate tensors.
    import onnx
    path = out / f"{stem}.onnx"
    onnx.save(onnx.shape_inference.infer_shapes(constants_as_initializers(onnx.load(path))), path)
    images.tofile(out / f"{stem}_images.bin")
    output.tofile(out / f"{stem}_output.bin")
    print(f"{stem} in {out}; output range [{output.min():.3f}, {output.max():.3f}]")


FIXTURES = {"v1": "tree_enhancer_fixture", "v2": "tree_enhancer_v2_fixture", "v3": "tree_enhancer_v3_fixture",
            "v6": "tree_enhancer_v6_fixture"}


def main() -> None:
    for arch in sys.argv[1:] or FIXTURES:
        write(arch, FIXTURES[arch])


if __name__ == "__main__":
    main()
