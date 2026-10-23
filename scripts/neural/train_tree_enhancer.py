# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "pillow", "onnx"]
# ///

"""Trains the small CNN that enhances the Gaussian renderer's distant trees.

Reads the view pairs of tree_neural_dataset_example (lr_*, fa_*, fb_*, hf_*, mask_*): the network sees the
Gaussian image and its features (8 channels) and predicts a correction added to the image. Its target is the
reference where the far trees' mask says so and the Gaussian image elsewhere, so it learns to change only the
distant trees. Every tenth view is held out for validation. Writes the weights, the ONNX model and the
validation numbers (the error of 4 px block means against the target on the far trees' pixels, of the input
and of the output) to the output directory.

    uv run scripts/neural/train_tree_enhancer.py <dataset-dir> <out-dir> [--steps 20000]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

INPUT_CHANNELS = 8


def load_view(directory: Path, name: str) -> dict[str, np.ndarray]:
    def rgb(prefix: str) -> np.ndarray:
        return np.array(Image.open(directory / f"{prefix}_{name}.ppm").convert("RGB"), dtype=np.uint8)

    return {
        "lr": rgb("lr"),
        "fa": rgb("fa"),
        "fb": rgb("fb"),
        "hf": rgb("hf"),
        "mask": np.array(Image.open(directory / f"mask_{name}.pgm"), dtype=np.uint8),
    }


def to_tensors(view: dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The network's input (8, H, W), its target (3, H, W) and the far trees' mask (1, H, W), all in [0, 1]."""
    lr = torch.from_numpy(view["lr"]).permute(2, 0, 1).float() / 255.0
    fa = torch.from_numpy(view["fa"]).permute(2, 0, 1).float() / 255.0
    fb = torch.from_numpy(view["fb"][..., :2]).permute(2, 0, 1).float() / 255.0
    hf = torch.from_numpy(view["hf"]).permute(2, 0, 1).float() / 255.0
    mask = torch.from_numpy(view["mask"]).unsqueeze(0).float() / 255.0
    inputs = torch.cat([lr, fa, fb], dim=0)
    target = mask * hf + (1.0 - mask) * lr
    return inputs, target, mask


class Enhancer(nn.Module):
    """A three-level U-Net: 16, 32 and 48 channels; the output is added to the input image."""

    def __init__(self, width: tuple[int, int, int] = (16, 32, 48)) -> None:
        super().__init__()
        a, b, c = width

        def block(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1), nn.ReLU(),
                                 nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU())

        self.enc0 = block(INPUT_CHANNELS, a)
        self.enc1 = block(a, b, stride=2)
        self.enc2 = block(b, c, stride=2)
        self.up1 = nn.ConvTranspose2d(c, b, 2, stride=2)
        self.dec1 = block(2 * b, b)
        self.up0 = nn.ConvTranspose2d(b, a, 2, stride=2)
        self.dec0 = block(2 * a, a)
        self.out = nn.Conv2d(a, 3, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        # The input image's RGB, taken by a fixed 1 x 1 convolution with a zero bias (exported as a Conv with bias,
        # as OnnxNetwork runs it).
        self.pick = nn.Conv2d(INPUT_CHANNELS, 3, 1)
        with torch.no_grad():
            self.pick.weight.zero_()
            self.pick.bias.zero_()
            for c in range(3):
                self.pick.weight[c, c] = 1.0
        self.pick.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.enc0(x)
        e1 = self.enc1(e0)
        e2 = self.enc2(e1)
        d1 = self.dec1(torch.cat([self.up1(e2), e1], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), e0], dim=1))
        return self.pick(x) + self.out(d0)


def constants_as_initializers(model):
    """Moves the values of Constant nodes into the graph's initializers and lists every initializer as a graph
    input, as OnnxNetwork reads them."""
    import onnx

    graph = model.graph
    keep = []
    for node in graph.node:
        if node.op_type == "Constant" and len(node.attribute) == 1 and node.attribute[0].name == "value":
            tensor = onnx.TensorProto()
            tensor.CopyFrom(node.attribute[0].t)
            tensor.name = node.output[0]
            graph.initializer.append(tensor)
        else:
            keep.append(node)
    del graph.node[:]
    graph.node.extend(keep)
    # OnnxNetwork makes tensors of the graph's inputs (and fills them from the initializers of the same name):
    # every initializer is listed as an input, too.
    listed = {i.name for i in graph.input}
    for tensor in graph.initializer:
        if tensor.name not in listed:
            graph.input.append(onnx.helper.make_tensor_value_info(tensor.name, tensor.data_type, list(tensor.dims)))
    return model


def block_error(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    """The mean absolute difference of 4 px block means (RGB averaged), over the blocks the mask touches."""
    da = F.avg_pool2d(a, 4) - F.avg_pool2d(b, 4)
    weight = F.max_pool2d(mask, 4) > 0.0
    per_block = da.abs().mean(dim=1, keepdim=True)
    return float(per_block[weight].mean()) if weight.any() else 0.0


def crops(inputs: list[torch.Tensor], targets: list[torch.Tensor], masks: list[torch.Tensor], size: int,
          count: int, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """`count` crops of `size` px, three quarters of them centred on a far-tree pixel."""
    xs, ys, ms = [], [], []
    for _ in range(count):
        k = int(rng.integers(len(inputs)))
        _, h, w = inputs[k].shape
        mask = masks[k][0]
        if rng.random() < 0.75 and bool((mask > 0.5).any()):
            idx = torch.nonzero(mask > 0.5)
            cy, cx = idx[int(rng.integers(len(idx)))].tolist()
            y = int(np.clip(cy - size // 2, 0, h - size))
            x = int(np.clip(cx - size // 2, 0, w - size))
        else:
            y = int(rng.integers(h - size + 1))
            x = int(rng.integers(w - size + 1))
        xs.append(inputs[k][:, y:y + size, x:x + size])
        ys.append(targets[k][:, y:y + size, x:x + size])
        ms.append(masks[k][:, y:y + size, x:x + size])
    return torch.stack(xs), torch.stack(ys), torch.stack(ms)


def gradient_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    dxa, dxb = a[..., :, 1:] - a[..., :, :-1], b[..., :, 1:] - b[..., :, :-1]
    dya, dyb = a[..., 1:, :] - a[..., :-1, :], b[..., 1:, :] - b[..., :-1, :]
    return (dxa - dxb).abs().mean() + (dya - dyb).abs().mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    names = sorted(p.stem[3:] for p in args.dataset.glob("hf_*.ppm"))
    train, val = [n for n in names if int(n) % 10 != 9], [n for n in names if int(n) % 10 == 9]
    print(f"{len(train)} training views, {len(val)} validation views on {device}")

    def load(group: list[str]):
        tensors = [to_tensors(load_view(args.dataset, n)) for n in group]
        return [t[0] for t in tensors], [t[1] for t in tensors], [t[2] for t in tensors]

    tx, ty, tm = load(train)
    vx, vy, vm = load(val)

    model = Enhancer().to(device)
    print(f"{sum(p.numel() for p in model.parameters())} parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, s / 500) * 0.5 * (1.0 + math.cos(math.pi * s / args.steps)))
    rng = np.random.default_rng(1)

    def validate() -> tuple[float, float]:
        model.eval()
        before = after = 0.0
        with torch.no_grad():
            for x, y, m in zip(vx, vy, vm):
                out = model(x.unsqueeze(0).to(device)).clamp(0.0, 1.0).cpu()
                before += block_error(x[:3].unsqueeze(0), y.unsqueeze(0), m.unsqueeze(0))
                after += block_error(out, y.unsqueeze(0), m.unsqueeze(0))
        model.train()
        return before / max(len(vx), 1), after / max(len(vx), 1)

    history = []
    start = time.time()
    for step in range(1, args.steps + 1):
        x, y, m = crops(tx, ty, tm, args.crop, args.batch, rng)
        x, y, m = x.to(device), y.to(device), m.to(device)
        out = model(x)
        # The far trees weigh most; elsewhere the network learns to leave the image alone.
        weight = 0.2 + m
        loss = ((out - y).abs() * weight).mean() + 0.5 * gradient_l1(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % 500 == 0 or step == args.steps:
            before, after = validate()
            history.append({"step": step, "loss": loss.item(), "val_input": before, "val_output": after})
            print(f"step {step}: loss {loss.item():.4f}; validation block error input {before:.4f}, "
                  f"output {after:.4f} ({100.0 * (1.0 - after / max(before, 1e-9)):.1f} % lower); "
                  f"{time.time() - start:.0f} s", flush=True)

    torch.save(model.state_dict(), args.out_dir / "enhancer.pt")
    (args.out_dir / "history.json").write_text(json.dumps(history, indent=1))
    model.eval().cpu()
    for height, width in ((720, 1280),):
        dummy = torch.zeros(1, INPUT_CHANNELS, height, width)
        path = args.out_dir / f"enhancer_{width}x{height}.onnx"
        torch.onnx.export(model, dummy, path, input_names=["input"], output_names=["output"], opset_version=17,
                          dynamo=False)
        # OnnxNetwork needs the shapes of the intermediate tensors.
        onnx.save(onnx.shape_inference.infer_shapes(constants_as_initializers(onnx.load(path))), path)
    print(f"model in {args.out_dir}")


if __name__ == "__main__":
    main()
