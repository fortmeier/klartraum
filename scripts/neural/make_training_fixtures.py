# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Reference gradients for the training elements of Klartraum (tests/test_photo_enhancer_training.cpp): for every
case, random inputs and an upstream gradient, and PyTorch autograd's gradients, as float32 files
`<out>/<case>_<tensor>.bin` with the shapes in `<out>/<case>.txt` (one line per tensor: name and dimensions).

    uv run --project scripts/qwen_image python scripts/neural/make_training_fixtures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

OUT = Path(__file__).resolve().parents[2] / "build/TestingOutput/training_fixtures"


def save(out: Path, case: str, tensors: dict[str, torch.Tensor]) -> None:
    lines = []
    for name, tensor in tensors.items():
        tensor.detach().float().contiguous().numpy().tofile(out / f"{case}_{name}.bin")
        lines.append(name + " " + " ".join(str(d) for d in tensor.shape))
    (out / f"{case}.txt").write_text("\n".join(lines) + "\n")


def conv_case(out: Path, case: str, n: int, cin: int, cout: int, size: tuple[int, int], kernel: int, stride: int,
              pad: int, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, cin, *size, generator=generator, requires_grad=True)
    w = (torch.randn(cout, cin, kernel, kernel, generator=generator) * 0.3).requires_grad_(True)
    b = torch.randn(cout, generator=generator, requires_grad=True)
    y = F.conv2d(x, w, b, stride=stride, padding=pad)
    dy = torch.randn(y.shape, generator=generator)
    y.backward(dy)
    save(out, case, {"x": x, "w": w, "b": b, "y": y, "dy": dy, "dx": x.grad, "dw": w.grad, "db": b.grad})


def activation_case(out: Path, case: str, function, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    x = (torch.randn(3, 4, 5, 6, generator=generator) * 2.0).requires_grad_(True)
    y = function(x)
    dy = torch.randn(y.shape, generator=generator)
    y.backward(dy)
    save(out, case, {"x": x, "y": y, "dy": dy, "dx": x.grad})


def channel_norm_case(out: Path, seed: int) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_photo_enhancer import ChannelNorm

    generator = torch.Generator().manual_seed(seed)
    norm = ChannelNorm(6)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(6, generator=generator))
        norm.bias.copy_(torch.randn(6, generator=generator))
    x = (torch.randn(2, 6, 5, 7, generator=generator) * 3.0 + 1.0).requires_grad_(True)
    y = norm(x)
    dy = torch.randn(y.shape, generator=generator)
    y.backward(dy)
    save(out, "channel_norm", {"x": x, "weight": norm.weight, "bias": norm.bias, "y": y, "dy": dy, "dx": x.grad,
                               "dweight": norm.weight.grad, "dbias": norm.bias.grad})


def upsample_case(out: Path, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 3, 5, 7, generator=generator, requires_grad=True)
    y = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
    dy = torch.randn(y.shape, generator=generator)
    y.backward(dy)
    save(out, "upsample2x", {"x": x, "y": y, "dy": dy, "dx": x.grad})


def film_case(out: Path, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 4, 5, 6, generator=generator, requires_grad=True)
    scale = torch.randn(2, 4, generator=generator, requires_grad=True)
    shift = torch.randn(2, 4, generator=generator, requires_grad=True)
    y = x * scale[:, :, None, None] + shift[:, :, None, None]
    dy = torch.randn(y.shape, generator=generator)
    y.backward(dy)
    save(out, "film", {"x": x, "scale": scale, "shift": shift, "y": y, "dy": dy, "dx": x.grad,
                       "dscale": scale.grad, "dshift": shift.grad})


def hash_texture_case(out: Path, seed: int) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hash_texture import SurfaceTexture

    generator = torch.Generator().manual_seed(seed)
    texture = SurfaceTexture(classes=(2, 3), levels=4, features=2, log2_table=10, base=4, finest=64)
    with torch.no_grad():
        for grid in texture.grids:
            grid.embeddings.copy_(torch.rand(grid.embeddings.shape, generator=generator) * 2.0 - 1.0)
    coordinates = torch.rand(2, 3, 5, 7, generator=generator)
    classes = torch.randint(0, 5, (2, 5, 7), generator=generator)
    features = texture(coordinates, classes)
    gradient = torch.randn(features.shape, generator=generator)
    features.backward(gradient)
    table = torch.stack([grid.embeddings for grid in texture.grids])
    table_gradient = torch.stack([grid.embeddings.grad for grid in texture.grids])
    grid = texture.grids[0]
    (out / "hash_texture_shape.txt").write_text(
        f"width 7\nheight 5\nlevels {grid.levels}\nfeatures {grid.features}\ntable {grid.table}\n"
        "resolutions " + " ".join(str(r) for r in grid.resolutions) + "\nclasses 2 3\n")
    save(out, "hash_texture", {"coordinates": coordinates, "classes": classes.float(), "table": table,
                               "features": features, "dfeatures": gradient, "dtable": table_gradient})


def photometric_case(out: Path, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    image = torch.rand(2, 3, 16, 20, generator=generator, requires_grad=True)
    target = torch.rand(2, 3, 16, 20, generator=generator)
    loss = 0.5 * F.l1_loss(image, target) + 5.0 * F.l1_loss(F.avg_pool2d(image, 8), F.avg_pool2d(target, 8))
    loss.backward()
    save(out, "photometric", {"image": image, "target": target, "dimage": image.grad, "loss": loss.reshape(1)})


def linear_case(out: Path, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    layer = torch.nn.Linear(5, 7)
    with torch.no_grad():
        layer.weight.copy_(torch.randn(7, 5, generator=generator))
        layer.bias.copy_(torch.randn(7, generator=generator))
    x = torch.randn(3, 5, generator=generator, requires_grad=True)
    y = layer(x)
    dy = torch.randn(3, 7, generator=generator)
    y.backward(dy)
    save(out, "linear", {"x": x, "w": layer.weight, "b": layer.bias, "y": y, "dy": dy, "dx": x.grad,
                         "dw": layer.weight.grad, "db": layer.bias.grad})


def clip_case(out: Path, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    a = torch.nn.Parameter(torch.zeros(300))
    b = torch.nn.Parameter(torch.zeros(70))
    a.grad = torch.randn(300, generator=generator) * 0.2
    b.grad = torch.randn(70, generator=generator) * 0.2
    before = {"ga": a.grad.clone(), "gb": b.grad.clone()}
    torch.nn.utils.clip_grad_norm_([a, b], 1.0)
    save(out, "clip", {**before, "clipped_a": a.grad, "clipped_b": b.grad})


def generator_step_case(out: Path, seed: int, case: str = "generator_step", perceptual: float = 0.0) -> None:
    """One photometric training step of the local photo enhancer (3 levels, ChannelNorm, FiLM by appearance codes,
    hash-grid texture) on a batch of two 32 x 32 crops, as train_photo_enhancer.py takes it: the parameters before
    (named as in the state dict), the batch, the image, the loss, every parameter's gradient and the parameters after
    clipping and the Adam step of the three parameter groups. The zero-initialised FiLM heads, output convolution and
    tiny texture are randomised so that every gradient is non-zero. With `perceptual`, the VGG loss joins with that
    weight."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hash_texture import SurfaceTexture
    from train_photo_enhancer import Generator

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    texture = SurfaceTexture(classes=(2, 3, 4), levels=4, features=2, log2_table=10, base=4, finest=64)
    network = Generator(widths=(32, 64, 128), extra=texture.channels, blur_depth=False, appearance=8, norm="channel")
    codes = torch.nn.Embedding(5, 8)
    with torch.no_grad():
        for grid in texture.grids:
            grid.embeddings.copy_(torch.rand(grid.embeddings.shape, generator=generator) * 0.2 - 0.1)
        for head in list(network.film.gammas) + list(network.film.betas):
            head.weight.copy_(torch.randn(head.weight.shape, generator=generator) * 0.1)
            head.bias.copy_(torch.randn(head.bias.shape, generator=generator) * 0.1)
        network.out.weight.copy_(torch.randn(network.out.weight.shape, generator=generator) * 0.2)
        codes.weight.copy_(torch.randn(codes.weight.shape, generator=generator) * 0.1)
    parameters = {"texture.table": None, **{f"generator.{k}": v for k, v in network.named_parameters()},
                  "codes.weight": codes.weight}
    b, size = 2, 32
    channels = torch.rand(b, 11, size, size, generator=generator)
    coordinates = torch.rand(b, 3, size, size, generator=generator)
    classes = torch.randint(0, 7, (b, size, size), generator=generator)
    target = torch.rand(b, 3, size, size, generator=generator)
    rows = torch.tensor([3, 1])
    x = torch.cat([channels, texture(coordinates, classes)], dim=1)
    image = network(x, None, codes(rows))
    loss = 0.5 * F.l1_loss(image, target) + 5.0 * F.l1_loss(F.avg_pool2d(image, 8), F.avg_pool2d(target, 8))
    if perceptual > 0:
        from train_photo_enhancer import Vgg

        loss = loss + perceptual * Vgg().eval().distance(image, target)
    loss = loss + 1e-2 * codes(rows).pow(2).sum(1).mean()
    loss.backward()
    table = torch.stack([grid.embeddings for grid in texture.grids])
    table_gradient = torch.stack([grid.embeddings.grad for grid in texture.grids])
    tensors = {"channels": channels, "coordinates": coordinates, "classes": classes.float(), "target": target,
               "rows": rows.float(), "image": image, "loss": loss.reshape(1)}
    names = []
    for name, value in parameters.items():
        value = table if name == "texture.table" else value
        gradient = table_gradient if name == "texture.table" else value.grad
        tensors["before." + name] = value.detach().clone()
        tensors["grad." + name] = gradient
        names.append(name)
    network_parameters = list(network.parameters())
    optimiser = torch.optim.Adam([{"params": network_parameters, "lr": 1e-4},
                                  {"params": [g.embeddings for g in texture.grids], "lr": 1e-2, "betas": (0.9, 0.99)},
                                  {"params": list(codes.parameters()), "lr": 5e-3, "betas": (0.9, 0.99)}],
                                 lr=1e-4, betas=(0.5, 0.999))
    torch.nn.utils.clip_grad_norm_(network_parameters, 1.0)
    optimiser.step()
    table = torch.stack([grid.embeddings for grid in texture.grids])
    for name, value in parameters.items():
        tensors["after." + name] = (table if name == "texture.table" else value).detach().clone()
    save(out, case, tensors)
    (out / f"{case}_parameters.txt").write_text("\n".join(names) + "\n")
    grid = texture.grids[0]
    (out / f"{case}_texture.txt").write_text(
        f"width {size}\nheight {size}\nlevels {grid.levels}\nfeatures {grid.features}\ntable {grid.table}\n"
        "resolutions " + " ".join(str(r) for r in grid.resolutions) + "\nclasses 2 3 4\n")


def vgg_case(out: Path, seed: int) -> None:
    """train_photo_enhancer.Vgg's distance of an image to a target and its gradient, and a 2 x 2 max-pool's."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_photo_enhancer import Vgg

    generator = torch.Generator().manual_seed(seed)
    vgg = Vgg().eval()
    image = torch.rand(2, 3, 32, 32, generator=generator, requires_grad=True)
    target = torch.rand(2, 3, 32, 32, generator=generator)
    distance = vgg.distance(image, target)
    distance.backward()
    save(out, "vgg", {"image": image, "target": target, "dimage": image.grad, "loss": distance.reshape(1)})
    x = torch.randn(2, 3, 6, 8, generator=generator, requires_grad=True)
    y = F.max_pool2d(x, 2)
    dy = torch.randn(y.shape, generator=generator)
    y.backward(dy)
    save(out, "max_pool", {"x": x, "y": y, "dy": dy, "dx": x.grad})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    conv_case(args.out, "conv3x3", 2, 5, 7, (9, 11), 3, 1, 1, 1)
    conv_case(args.out, "conv3x3_stride2", 2, 5, 7, (10, 12), 3, 2, 1, 2)
    conv_case(args.out, "conv1x1", 2, 6, 3, (8, 8), 1, 1, 0, 3)
    conv_case(args.out, "conv4x4_stride2", 2, 4, 6, (12, 10), 4, 2, 1, 4)
    activation_case(args.out, "elu", F.elu, 5)
    activation_case(args.out, "leaky_relu", lambda x: F.leaky_relu(x, 0.2), 6)
    channel_norm_case(args.out, 7)
    upsample_case(args.out, 8)
    film_case(args.out, 9)
    hash_texture_case(args.out, 10)
    photometric_case(args.out, 11)
    linear_case(args.out, 12)
    clip_case(args.out, 13)
    generator_step_case(args.out, 14)
    vgg_case(args.out, 15)
    generator_step_case(args.out, 14, "generator_step_vgg", 1.0)
    print(f"fixtures in {args.out}")


if __name__ == "__main__":
    main()
