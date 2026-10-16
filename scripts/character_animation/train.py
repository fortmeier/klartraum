# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Train the mixture-of-experts character controller and export it to ONNX.

The architecture follows the mode-adaptive neural networks of Zhang, Starke
et al. (SIGGRAPH 2018), the basis of Starke's Neural State Machine and Local
Motion Phases: a small gating network computes blending weights for K expert
networks, and the motion network evaluates the blended expert. Blending the
expert outputs layer by layer is the same as blending their weights, because
every layer is linear before its activation:

    sum_k w_k (x W_k + b_k) = x (sum_k w_k W_k) + sum_k w_k b_k

The exported model takes the raw features of features.py and returns raw
outputs; normalization is part of the graph.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

import features

ROOT = Path(__file__).resolve().parents[2]


class Experts(nn.Module):
    """One layer of K experts, blended per sample."""

    def __init__(self, experts: int, inputs: int, outputs: int):
        super().__init__()
        bound = math.sqrt(6.0 / (inputs + outputs))
        self.weight = nn.Parameter(torch.empty(experts, inputs, outputs).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.zeros(experts, outputs))

    def forward(self, x: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        experts, inputs, outputs = self.weight.shape
        # x [B, in] -> per-expert outputs [B, K, out] -> blended [B, out]
        per_expert = (x @ self.weight.permute(1, 0, 2).reshape(inputs, experts * outputs)).reshape(-1, experts, outputs)
        return (blend.unsqueeze(1) @ per_expert).reshape(-1, outputs) + blend @ self.bias


class Controller(nn.Module):
    def __init__(self, experts: int = 8, hidden: int = 512, gating_hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.gating = nn.Sequential(
            nn.Linear(features.GATING_INPUT_SIZE, gating_hidden),
            nn.ELU(),
            nn.Linear(gating_hidden, gating_hidden),
            nn.ELU(),
            nn.Linear(gating_hidden, experts),
            nn.Softmax(dim=-1),
        )
        self.layers = nn.ModuleList(
            [
                Experts(experts, features.MOTION_INPUT_SIZE, hidden),
                Experts(experts, hidden, hidden),
                Experts(experts, hidden, features.OUTPUT_SIZE),
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()

    def forward(self, motion: torch.Tensor, gating: torch.Tensor) -> torch.Tensor:
        blend = self.gating(gating)
        x = motion
        for index, layer in enumerate(self.layers):
            x = layer(self.dropout(x), blend)
            if index < len(self.layers) - 1:
                x = self.activation(x)
        return x


class Normalized(nn.Module):
    """Raw features in, raw outputs out; the form that is exported."""

    def __init__(self, controller: Controller, statistics: dict[str, np.ndarray]):
        super().__init__()
        self.controller = controller
        for name, value in statistics.items():
            self.register_buffer(name, torch.from_numpy(value).float().unsqueeze(0))

    def forward(self, motion_input: torch.Tensor, gating_input: torch.Tensor) -> torch.Tensor:
        motion = (motion_input - self.motion_mean) / self.motion_std
        gating = (gating_input - self.gating_mean) / self.gating_std
        return self.controller(motion, gating) * self.output_std + self.output_mean


def statistics(database: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    def mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return values.mean(0), np.maximum(values.std(0), 1e-2)

    motion_mean, motion_std = mean_std(database["motion_inputs"])
    gating_mean, gating_std = mean_std(database["gating_inputs"])
    output_mean, output_std = mean_std(database["outputs"])
    return {
        "motion_mean": motion_mean,
        "motion_std": motion_std,
        "gating_mean": gating_mean,
        "gating_std": gating_std,
        "output_mean": output_mean,
        "output_std": output_std,
    }


def train(args: argparse.Namespace) -> None:
    database = dict(np.load(args.database))
    stats = statistics(database)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    motion = torch.from_numpy((database["motion_inputs"] - stats["motion_mean"]) / stats["motion_std"]).to(device)
    gating = torch.from_numpy((database["gating_inputs"] - stats["gating_mean"]) / stats["gating_std"]).to(device)
    outputs = torch.from_numpy((database["outputs"] - stats["output_mean"]) / stats["output_std"]).to(device)
    count = motion.shape[0]
    pose_slice = slice(features.TRAJECTORY_SIZE, features.TRAJECTORY_SIZE + features.POSE_SIZE)

    controller = Controller(args.experts, args.hidden).to(device)
    optimizer = torch.optim.AdamW(controller.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = count // args.batch_size
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * steps_per_epoch)
    print(f"{count} samples on {device}, {sum(p.numel() for p in controller.parameters())} parameters")

    for epoch in range(args.epochs):
        started = time.time()
        permutation = torch.randperm(count, device=device)
        total = 0.0
        controller.train()
        for step in range(steps_per_epoch):
            batch = permutation[step * args.batch_size : (step + 1) * args.batch_size]
            x = motion[batch]
            # Noise on the fed-back pose makes autoregressive rollouts recover from their own errors.
            if args.pose_noise > 0:
                x = x.clone()
                x[:, pose_slice] += args.pose_noise * torch.randn_like(x[:, pose_slice])
            prediction = controller(x, gating[batch])
            loss = torch.mean((prediction - outputs[batch]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item()
        print(f"epoch {epoch + 1}/{args.epochs}: loss {total / steps_per_epoch:.5f} ({time.time() - started:.0f} s)")

    controller.eval().cpu()
    torch.save({"state": controller.state_dict(), "statistics": stats, "args": vars(args)}, args.checkpoint)
    export(controller, stats, args.onnx)


def add_initializer_value_info(model: onnx.ModelProto) -> None:
    """Klartraum reads the shape of every tensor from value_info, initializers included."""
    known = {value.name for value in model.graph.value_info} | {value.name for value in model.graph.input}
    for initializer in model.graph.initializer:
        if initializer.name not in known:
            model.graph.value_info.append(
                onnx.helper.make_tensor_value_info(initializer.name, initializer.data_type, list(initializer.dims))
            )


def export(controller: Controller, stats: dict[str, np.ndarray], path: Path) -> None:
    model = Normalized(controller, stats).eval()
    motion = torch.zeros(1, features.MOTION_INPUT_SIZE)
    gating = torch.zeros(1, features.GATING_INPUT_SIZE)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (motion, gating),
        path,
        input_names=["motion_input", "gating_input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    exported = onnx.shape_inference.infer_shapes(onnx.load(path))
    add_initializer_value_info(exported)
    onnx.save(exported, path)
    print(f"wrote {path} ({path.stat().st_size / 2**20:.1f} MiB), operators: "
          f"{sorted({node.op_type for node in exported.graph.node})}")

    rng = np.random.default_rng(0)
    motion = rng.normal(size=(1, features.MOTION_INPUT_SIZE)).astype(np.float32) * stats["motion_std"] + stats[
        "motion_mean"
    ]
    gating = rng.normal(size=(1, features.GATING_INPUT_SIZE)).astype(np.float32) * stats["gating_std"] + stats[
        "gating_mean"
    ]
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    reference = model(torch.from_numpy(motion.astype(np.float32)), torch.from_numpy(gating.astype(np.float32)))
    result = session.run(None, {"motion_input": motion.astype(np.float32), "gating_input": gating.astype(np.float32)})
    error = np.abs(result[0] - reference.detach().numpy()).max()
    print(f"ONNX Runtime vs PyTorch maximum error: {error:.2e}")
    if error > 1e-3:
        raise RuntimeError("exported model does not match PyTorch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", type=Path, default=ROOT / "build" / "character_animation" / "database.npz")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "build" / "character_animation" / "controller.pt")
    parser.add_argument("--onnx", type=Path, default=ROOT / "data" / "character" / "controller.onnx")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--pose-noise", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--export-only", action="store_true", help="export the saved checkpoint again")
    args = parser.parse_args()
    if args.export_only:
        checkpoint = torch.load(args.checkpoint, weights_only=False)
        saved = checkpoint["args"]
        controller = Controller(saved["experts"], saved["hidden"])
        controller.load_state_dict(checkpoint["state"])
        export(controller.eval(), checkpoint["statistics"], args.onnx)
        return
    train(args)


if __name__ == "__main__":
    main()
