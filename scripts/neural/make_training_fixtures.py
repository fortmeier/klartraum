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


def augment_with(pair: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    """train_photo_enhancer.augment_pair() with its random numbers given, per sample: brightness, contrast, the roll
    (y, x) and the cutout centre (y, x)."""
    b, c, h, w = pair.shape
    out = []
    for k in range(b):
        brightness, contrast, sy, sx, cy, cx = parameters[k].tolist()
        image = pair[k:k + 1, -3:] + brightness
        mean = image.mean(dim=1, keepdim=True)
        image = (image - mean) * contrast + mean
        sample = torch.cat([pair[k:k + 1, :-3], image], dim=1)
        sample = torch.roll(sample, shifts=(int(sy), int(sx)), dims=(2, 3))
        mask = torch.ones(1, 1, h, w)
        cy, cx = int(cy), int(cx)
        mask[:, :, max(0, cy - h // 4):cy + h // 4, max(0, cx - w // 4):cx + w // 4] = 0
        out.append(sample * mask)
    return torch.cat(out)


def gan_step_case(out: Path, seed: int) -> None:
    """One adversarial training step as train_photo_enhancer.py takes it, on 64 x 64 crops: the discriminator's step
    on the augmented real and fake pairs (non-saturating loss, Adam lr 2e-4), then the generator's with the
    photometric losses, the adversarial loss (0.3) and feature matching (2) against the updated discriminator (no R1).
    The augmentations' random numbers are given (augment_with)."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hash_texture import SurfaceTexture
    from train_photo_enhancer import Generator, MultiScaleDiscriminator

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    texture = SurfaceTexture(classes=(2, 3, 4), levels=4, features=2, log2_table=10, base=4, finest=64)
    network = Generator(widths=(32, 64, 128), extra=texture.channels, blur_depth=False, appearance=8, norm="channel")
    codes = torch.nn.Embedding(5, 8)
    discriminator = MultiScaleDiscriminator()
    with torch.no_grad():
        for grid in texture.grids:
            grid.embeddings.copy_(torch.rand(grid.embeddings.shape, generator=generator) * 0.2 - 0.1)
        for head in list(network.film.gammas) + list(network.film.betas):
            head.weight.copy_(torch.randn(head.weight.shape, generator=generator) * 0.1)
            head.bias.copy_(torch.randn(head.bias.shape, generator=generator) * 0.1)
        network.out.weight.copy_(torch.randn(network.out.weight.shape, generator=generator) * 0.2)
        codes.weight.copy_(torch.randn(codes.weight.shape, generator=generator) * 0.1)
    b, size = 2, 64
    channels = torch.rand(b, 11, size, size, generator=generator)
    coordinates = torch.rand(b, 3, size, size, generator=generator)
    classes = torch.randint(0, 7, (b, size, size), generator=generator)
    target = torch.rand(b, 3, size, size, generator=generator)
    rows = torch.tensor([3, 1])
    # Per sample of [real pairs; fake pairs]: brightness, contrast, roll y, roll x, cutout y, x (one roll and cutout per
    # call of augment_pair, so per half).
    augmentation = torch.tensor([[0.1, 1.2, 2, -5, 10, 40], [-0.15, 0.7, 2, -5, 10, 40],
                                 [0.05, 0.9, -3, 7, 50, 3], [0.18, 1.4, -3, 7, 50, 3]], dtype=torch.float32)
    generator_parameters = {"texture.table": None, **{f"generator.{k}": v for k, v in network.named_parameters()},
                            "codes.weight": codes.weight}
    discriminator_parameters = {f"discriminator.{k}": v for k, v in discriminator.named_parameters()}
    tensors = {"channels": channels, "coordinates": coordinates, "classes": classes.float(), "target": target,
               "rows": rows.float(), "augmentation": augmentation}
    table = torch.stack([grid.embeddings for grid in texture.grids])
    for name, value in {**generator_parameters, **discriminator_parameters}.items():
        tensors["before." + name] = (table if name == "texture.table" else value).detach().clone()

    x = torch.cat([channels, texture(coordinates, classes)], dim=1)
    fake = network(x, None, codes(rows))
    tensors["image"] = fake
    optimise_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    real_pair, fake_pair = torch.cat([channels, target], 1), torch.cat([channels, fake.detach()], 1)
    augmented = augment_with(torch.cat([real_pair, fake_pair]), augmentation)
    loss_d = 0.0
    for (_, real_logits), (_, fake_logits) in zip(discriminator(augmented[:b]), discriminator(augmented[b:])):
        loss_d = loss_d + F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()
    optimise_d.zero_grad()
    loss_d.backward()
    optimise_d.step()
    tensors["loss_d"] = loss_d.detach().reshape(1)

    loss = 0.5 * F.l1_loss(fake, target) + 5.0 * F.l1_loss(F.avg_pool2d(fake, 8), F.avg_pool2d(target, 8))
    fake_pair, real_pair = torch.cat([channels, fake], 1), torch.cat([channels, target], 1)
    adversarial = feature_matching = 0.0
    for (fake_features, fake_logits), (real_features, _) in zip(discriminator(fake_pair), discriminator(real_pair)):
        adversarial = adversarial + F.softplus(-fake_logits).mean()
        for f, r in zip(fake_features, real_features):
            feature_matching = feature_matching + F.l1_loss(f, r.detach()) / len(fake_features)
    loss = loss + 0.3 * adversarial + 2.0 * feature_matching + 1e-2 * codes(rows).pow(2).sum(1).mean()
    tensors["loss_adversarial"] = adversarial.detach().reshape(1)
    tensors["loss_feature_matching"] = feature_matching.detach().reshape(1)
    network_parameters = list(network.parameters())
    optimiser = torch.optim.Adam([{"params": network_parameters, "lr": 1e-4},
                                  {"params": [g.embeddings for g in texture.grids], "lr": 1e-2, "betas": (0.9, 0.99)},
                                  {"params": list(codes.parameters()), "lr": 5e-3, "betas": (0.9, 0.99)}],
                                 lr=1e-4, betas=(0.5, 0.999))
    optimiser.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network_parameters, 1.0)
    optimiser.step()
    table = torch.stack([grid.embeddings for grid in texture.grids])
    for name, value in {**generator_parameters, **discriminator_parameters}.items():
        tensors["after." + name] = (table if name == "texture.table" else value).detach().clone()
    save(out, "gan_step", tensors)
    (out / "gan_step_parameters.txt").write_text("\n".join([*generator_parameters, *discriminator_parameters]) + "\n")
    grid = texture.grids[0]
    (out / "gan_step_texture.txt").write_text(
        f"width {size}\nheight {size}\nlevels {grid.levels}\nfeatures {grid.features}\ntable {grid.table}\n"
        "resolutions " + " ".join(str(r) for r in grid.resolutions) + "\nclasses 2 3 4\n")


def latent_cases(out: Path, seed: int) -> None:
    """The latent renderer's elements (latent_renderer.py): the mean over each object's pixels and its gradient, the
    VAE's sample and KL with their gradients, and one training step of the whole renderer (texture, encoder, decoder;
    L1, block-mean L1, VGG and KL; clipping and Adam) on a batch of two 32 x 32 crops with given noise."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hash_texture import SurfaceTexture
    from latent_renderer import LatentRenderer, kl_divergence, object_mean, splat
    from train_photo_enhancer import Vgg

    generator = torch.Generator().manual_seed(seed)
    # Object means: 2 samples, 6 channels, 5 x 7 pixels, objects 0 (none), 1 and 2; object 2 missing in sample 1.
    values = torch.randn(2, 6, 5, 7, generator=generator, requires_grad=True)
    objects = torch.randint(0, 3, (2, 5, 7), generator=generator)
    objects[1][objects[1] == 2] = 1
    means, counts = object_mean(values, objects, 2)
    weights = torch.randn(means.shape, generator=generator)
    (means * weights).sum().backward()
    latents = torch.randn(2, 2, 3, generator=generator, requires_grad=True)
    splatted = splat(latents, objects)
    splat_weights = torch.randn(splatted.shape, generator=generator)
    (splatted * splat_weights).sum().backward()
    save(out, "object_mean", {"values": values, "objects": objects.float(), "means": means, "counts": counts,
                              "dmeans": weights, "dvalues": values.grad, "latents": latents, "splatted": splatted,
                              "dsplatted": splat_weights, "dlatents": latents.grad})
    # VAE: 2 samples, 2 objects (object 2 empty in sample 1), latent 3.
    mu = torch.randn(2, 2, 3, generator=generator, requires_grad=True)
    log_variance = (torch.randn(2, 2, 3, generator=generator) * 0.5).requires_grad_(True)
    epsilon = torch.randn(2, 2, 3, generator=generator)
    vae_counts = torch.tensor([[4.0, 3.0], [5.0, 0.0]])
    sample = mu + (0.5 * log_variance).exp() * epsilon
    dsample = torch.randn(sample.shape, generator=generator)
    kl = kl_divergence(mu, log_variance, vae_counts)
    ((sample * dsample).sum() + 0.3 * kl).backward()
    save(out, "vae", {"mu": mu, "log_variance": log_variance, "epsilon": epsilon, "counts": vae_counts,
                      "sample": sample, "kl": kl.reshape(1), "dsample": dsample, "dmu": mu.grad,
                      "dlog_variance": log_variance.grad})
    # The same VAE with 0.3 free bits: numbers below 0.3 nats get no KL gradient.
    mu.grad = None
    log_variance.grad = None
    sample = mu + (0.5 * log_variance).exp() * epsilon
    kl = kl_divergence(mu, log_variance, vae_counts, free_bits=0.3)
    ((sample * dsample).sum() + 0.3 * kl).backward()
    save(out, "vae_free_bits", {"kl": kl.reshape(1), "dmu": mu.grad, "dlog_variance": log_variance.grad})

    # One training step.
    torch.manual_seed(seed)
    texture = SurfaceTexture(classes=(3,), levels=4, features=2, log2_table=10, base=4, finest=64)
    model = LatentRenderer(texture, geometry=4, latent=4, objects=1)
    with torch.no_grad():
        for grid in texture.grids:
            grid.embeddings.copy_(torch.rand(grid.embeddings.shape, generator=generator) * 0.2 - 0.1)
        model.generator.out.weight.copy_(torch.randn(model.generator.out.weight.shape, generator=generator) * 0.2)
    b, size = 2, 32
    geometry = torch.rand(b, 4, size, size, generator=generator)
    coordinates = torch.rand(b, 3, size, size, generator=generator)
    classes = torch.zeros(b, size, size, dtype=torch.long)
    classes[:, 6:26, 8:28] = 3
    objects = (classes == 3).long()
    photos = torch.rand(b, 3, size, size, generator=generator)
    epsilon = torch.randn(b, 1, 4, generator=generator)
    parameters = {"texture.table": None, **{f"generator.{k}": v for k, v in model.generator.named_parameters()},
                  **{f"encoder.{k}": v for k, v in model.encoder.named_parameters()}}
    tensors = {"geometry": geometry, "coordinates": coordinates, "classes": classes.float(),
               "objects": objects.float(), "photos": photos, "epsilon": epsilon}
    table = torch.stack([grid.embeddings for grid in texture.grids])
    for name, value in parameters.items():
        tensors["before." + name] = (table if name == "texture.table" else value).detach().clone()
    mu, log_variance, latent_counts = model.encode(photos, objects)
    latents = mu + (0.5 * log_variance).exp() * epsilon
    image = model(geometry, coordinates, classes, objects, latents)
    photometric = 0.5 * F.l1_loss(image, photos) + 5.0 * F.l1_loss(F.avg_pool2d(image, 8), F.avg_pool2d(photos, 8))
    perceptual = Vgg().eval().distance(image, photos)
    kl = kl_divergence(mu, log_variance, latent_counts)
    loss = photometric + perceptual + 1e-2 * kl
    network = list(model.generator.parameters()) + list(model.encoder.parameters())
    optimiser = torch.optim.Adam([{"params": network, "lr": 1e-4},
                                  {"params": [g.embeddings for g in texture.grids], "lr": 1e-2, "betas": (0.9, 0.99)}],
                                 lr=1e-4, betas=(0.5, 0.999))
    optimiser.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network, 1.0)
    optimiser.step()
    table = torch.stack([grid.embeddings for grid in texture.grids])
    for name, value in parameters.items():
        tensors["after." + name] = (table if name == "texture.table" else value).detach().clone()
    tensors.update({"image": image, "photometric": photometric.reshape(1), "perceptual": perceptual.reshape(1),
                    "kl": kl.reshape(1), "mu": mu, "log_variance": log_variance})
    save(out, "latent_step", tensors)
    (out / "latent_step_parameters.txt").write_text("\n".join(parameters) + "\n")
    grid = texture.grids[0]
    (out / "latent_step_texture.txt").write_text(
        f"width {size}\nheight {size}\nlevels {grid.levels}\nfeatures {grid.features}\ntable {grid.table}\n"
        "resolutions " + " ".join(str(r) for r in grid.resolutions) + "\nclasses 3\n")


def latent_check_case(out: Path, seed: int) -> None:
    """One latent renderer step as latent_cases()' with the latent check (weight 0.5): each view rendered again with
    the other view's latents and encoded by the fixed encoder."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hash_texture import SurfaceTexture
    from latent_renderer import LatentRenderer, kl_divergence, latent_check
    from train_photo_enhancer import Vgg

    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    texture = SurfaceTexture(classes=(3,), levels=4, features=2, log2_table=10, base=4, finest=64)
    model = LatentRenderer(texture, geometry=4, latent=4, objects=1)
    with torch.no_grad():
        for grid in texture.grids:
            grid.embeddings.copy_(torch.rand(grid.embeddings.shape, generator=generator) * 0.2 - 0.1)
        model.generator.out.weight.copy_(torch.randn(model.generator.out.weight.shape, generator=generator) * 0.2)
    b, size = 2, 32
    geometry = torch.rand(b, 4, size, size, generator=generator)
    coordinates = torch.rand(b, 3, size, size, generator=generator)
    classes = torch.zeros(b, size, size, dtype=torch.long)
    classes[:, 6:26, 8:28] = 3
    objects = (classes == 3).long()
    photos = torch.rand(b, 3, size, size, generator=generator)
    epsilon = torch.randn(b, 1, 4, generator=generator)
    permutation = torch.tensor([1, 0])
    parameters = {"texture.table": None, **{f"generator.{k}": v for k, v in model.generator.named_parameters()},
                  **{f"encoder.{k}": v for k, v in model.encoder.named_parameters()}}
    tensors = {"geometry": geometry, "coordinates": coordinates, "classes": classes.float(),
               "objects": objects.float(), "photos": photos, "epsilon": epsilon}
    table = torch.stack([grid.embeddings for grid in texture.grids])
    for name, value in parameters.items():
        tensors["before." + name] = (table if name == "texture.table" else value).detach().clone()
    mu, log_variance, latent_counts = model.encode(photos, objects)
    latents = mu + (0.5 * log_variance).exp() * epsilon
    image = model(geometry, coordinates, classes, objects, latents)
    photometric = 0.5 * F.l1_loss(image, photos) + 5.0 * F.l1_loss(F.avg_pool2d(image, 8), F.avg_pool2d(photos, 8))
    perceptual = Vgg().eval().distance(image, photos)
    kl = kl_divergence(mu, log_variance, latent_counts)
    check, _ = latent_check(model, geometry, coordinates, classes, objects, latents, permutation)
    loss = photometric + perceptual + 1e-2 * kl + 0.5 * check
    network = list(model.generator.parameters()) + list(model.encoder.parameters())
    optimiser = torch.optim.Adam([{"params": network, "lr": 1e-4},
                                  {"params": [g.embeddings for g in texture.grids], "lr": 1e-2, "betas": (0.9, 0.99)}],
                                 lr=1e-4, betas=(0.5, 0.999))
    optimiser.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network, 1.0)
    optimiser.step()
    table = torch.stack([grid.embeddings for grid in texture.grids])
    for name, value in parameters.items():
        tensors["after." + name] = (table if name == "texture.table" else value).detach().clone()
    tensors.update({"image": image, "photometric": photometric.reshape(1), "perceptual": perceptual.reshape(1),
                    "kl": kl.reshape(1), "mu": mu, "log_variance": log_variance, "check": check.reshape(1),
                    "permutation": permutation.float()})
    save(out, "latent_check_step", tensors)
    (out / "latent_check_step_parameters.txt").write_text("\n".join(parameters) + "\n")
    grid = texture.grids[0]
    (out / "latent_check_step_texture.txt").write_text(
        f"width {size}\nheight {size}\nlevels {grid.levels}\nfeatures {grid.features}\ntable {grid.table}\n"
        "resolutions " + " ".join(str(r) for r in grid.resolutions) + "\nclasses 3\n")


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
    gan_step_case(args.out, 16)
    latent_cases(args.out, 17)
    latent_check_case(args.out, 18)
    print(f"fixtures in {args.out}")


if __name__ == "__main__":
    main()
