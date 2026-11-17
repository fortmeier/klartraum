# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Reference training steps of PhotoEnhancerTraining (tests/test_photo_enhancer_training.cpp): for every case, the
parameters before and after one step on a batch, the image, the losses and the clipped gradients, as float32 files
`<out>/<case>_<tensor>.bin` with the shapes in `<out>/<case>.txt` (one line per tensor: name and dimensions).

    uv run --project scripts/qwen_image python scripts/neural/make_training_fixtures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

OUT = Path(__file__).resolve().parents[2] / "build/TestingOutput/training_fixtures"


def save(out: Path, case: str, tensors: dict[str, torch.Tensor]) -> None:
    lines = []
    for name, tensor in tensors.items():
        tensor.detach().float().contiguous().numpy().tofile(out / f"{case}_{name}.bin")
        lines.append(name + " " + " ".join(str(d) for d in tensor.shape))
    (out / f"{case}.txt").write_text("\n".join(lines) + "\n")


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


def gan_step_case(out: Path, seed: int) -> None:
    """One adversarial training step as train_photo_enhancer.py takes it, on 64 x 64 crops: the discriminator's step
    on the augmented real and fake pairs (non-saturating loss, Adam lr 2e-4), then the generator's with the
    photometric losses, the adversarial loss (0.3) and feature matching (2) against the updated discriminator (no R1).
    The augmentations' random numbers are given (training_reference.augment_with())."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hash_texture import SurfaceTexture
    from train_photo_enhancer import Generator, MultiScaleDiscriminator
    from training_reference import augment_with

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
    """One training step of the latent renderer (latent_renderer.py: texture, encoder, decoder; L1, block-mean L1, VGG
    and KL; clipping and Adam) on a batch of two 32 x 32 crops with given noise."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hash_texture import SurfaceTexture
    from latent_renderer import LatentRenderer, kl_divergence
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
    for name, value in parameters.items():
        if value is not None:
            tensors["gradient." + name] = value.grad.detach().clone()
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


def mlp_texture_cases(out: Path, seed: int, perceptual: bool = True, case: str = "mlp_texture_step") -> None:
    """One latent renderer step with the latent-conditioned MLP texture (latent_renderer.MlpTexture) as latent_cases()'
    (texture MLP, encoder, decoder; L1, block-mean L1, VGG unless not `perceptual`, and KL; clipping and Adam) as
    `case`."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from latent_renderer import LatentRenderer, MlpTexture, kl_divergence
    from train_photo_enhancer import Vgg

    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    texture = MlpTexture(latent=4, frequencies=3, width=16, hidden=2, outputs=5)
    model = LatentRenderer(texture, geometry=4, latent=4, objects=1)
    with torch.no_grad():
        model.generator.out.weight.copy_(torch.randn(model.generator.out.weight.shape, generator=generator) * 0.2)
    b, size = 2, 32
    geometry = torch.rand(b, 4, size, size, generator=generator)
    coordinates = torch.rand(b, 3, size, size, generator=generator)
    classes = torch.zeros(b, size, size, dtype=torch.long)
    classes[:, 6:26, 8:28] = 3
    objects = (classes == 3).long()
    photos = torch.rand(b, 3, size, size, generator=generator)
    epsilon = torch.randn(b, 1, 4, generator=generator)
    parameters = {**{f"generator.texture.{k[4:]}": v for k, v in texture.named_parameters()},
                  **{f"generator.{k}": v for k, v in model.generator.named_parameters()},
                  **{f"encoder.{k}": v for k, v in model.encoder.named_parameters()}}
    tensors = {"geometry": geometry, "coordinates": coordinates, "classes": classes.float(),
               "objects": objects.float(), "photos": photos, "epsilon": epsilon}
    for name, value in parameters.items():
        tensors["before." + name] = value.detach().clone()
    mu, log_variance, latent_counts = model.encode(photos, objects)
    latents = mu + (0.5 * log_variance).exp() * epsilon
    image = model(geometry, coordinates, classes, objects, latents)
    photometric = 0.5 * F.l1_loss(image, photos) + 5.0 * F.l1_loss(F.avg_pool2d(image, 8), F.avg_pool2d(photos, 8))
    perceptual = Vgg().eval().distance(image, photos) if perceptual else torch.zeros(())
    kl = kl_divergence(mu, log_variance, latent_counts)
    loss = photometric + perceptual + 1e-2 * kl
    network = list(texture.parameters()) + list(model.generator.parameters()) + list(model.encoder.parameters())
    optimiser = torch.optim.Adam(network, lr=1e-4, betas=(0.5, 0.999))
    optimiser.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network, 1.0)
    for name, value in parameters.items():
        tensors["gradient." + name] = value.grad.detach().clone()
    optimiser.step()
    for name, value in parameters.items():
        tensors["after." + name] = value.detach().clone()
    tensors.update({"image": image, "photometric": photometric.reshape(1), "perceptual": perceptual.reshape(1),
                    "kl": kl.reshape(1), "mu": mu, "log_variance": log_variance})
    save(out, case, tensors)
    (out / f"{case}_parameters.txt").write_text("\n".join(parameters) + "\n")
    (out / f"{case}_texture.txt").write_text(
        f"width {size}\nheight {size}\nlevels 0\nfeatures 2\ntable 1\nresolutions\nclasses 3\n"
        f"mlp {texture.frequencies} 16 {texture.channels} 2\n")


def latent_table_cases(out: Path, seed: int, residual: tuple[tuple[int, int], ...] = (),
                       case: str = "latent_table_step", basis: tuple[int, ...] = (), basis_decay: float = 0.0,
                       basis_smoothness: float = 0.0) -> None:
    """One step of the latent renderer with a learned latent table instead of the encoder (an auto-decoder) and the
    latents only in the MLP texture (latent_to_decoder False): rows 3 and 1 of a 5-row table, L1 and block-mean L1
    plus the table's L2 penalty (1e-2 x the batch's mean squared row norm); the network clipped and stepped by Adam
    (1e-4), the table by its own Adam (5e-3, betas 0.9 and 0.99) unclipped. With `residual`, the MLP texture has
    those residual blocks (latent_renderer.ResidualMlpTexture), as `case`; with `basis` the code-mixed feature texture
    of these resolutions instead (latent_renderer.TextureBasis, tables from N(0, 0.1), stepped by the texture's Adam:
    1e-2, betas 0.9 and 0.99, unclipped), regularised by texture_basis_penalty() with `basis_decay` and
    `basis_smoothness` times each level's resolution over the coarsest's."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from latent_renderer import LatentRenderer, ResidualMlpTexture, TextureBasis, texture_basis_penalty

    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    if basis:
        texture = TextureBasis(latent=4, outputs=5, resolutions=basis)
        with torch.no_grad():
            for t in texture.tables:
                t.copy_(torch.randn(t.shape, generator=generator) * 0.1)
    else:
        texture = ResidualMlpTexture(latent=4, frequencies=3, width=16, hidden=2, outputs=5, residual=residual)
    model = LatentRenderer(texture, geometry=4, latent=4, objects=1, latent_to_decoder=False)
    table = torch.nn.Embedding(5, 4)
    with torch.no_grad():
        model.generator.out.weight.copy_(torch.randn(model.generator.out.weight.shape, generator=generator) * 0.2)
        table.weight.copy_(torch.randn(table.weight.shape, generator=generator))
    b, size = 2, 32
    geometry = torch.rand(b, 4, size, size, generator=generator)
    coordinates = torch.rand(b, 3, size, size, generator=generator)
    classes = torch.zeros(b, size, size, dtype=torch.long)
    classes[:, 6:26, 8:28] = 3
    objects = (classes == 3).long()
    photos = torch.rand(b, 3, size, size, generator=generator)
    rows = torch.tensor([3, 1])
    texture_parameters = {f"texture.basis.{level}": t for level, t in enumerate(texture.tables)} if basis else {}
    network_parameters = {}
    if not basis:
        network_parameters.update({f"generator.texture.{k[4:]}": v for k, v in texture.first.named_parameters()})
        for j, block in enumerate(texture.residual):
            network_parameters.update(
                {f"generator.texture_residual.{j}.{k[4:]}": v for k, v in block.named_parameters()})
    network_parameters.update({f"generator.{k}": v for k, v in model.generator.named_parameters()})
    parameters = {**texture_parameters, **network_parameters, "latents.weight": table.weight}
    tensors = {"geometry": geometry, "coordinates": coordinates, "classes": classes.float(),
               "objects": objects.float(), "photos": photos, "rows": rows.float()}
    for name, value in parameters.items():
        tensors["before." + name] = value.detach().clone()
    latents = table(rows)[:, None, :]
    image = model(geometry, coordinates, classes, objects, latents)
    photometric = 0.5 * F.l1_loss(image, photos) + 5.0 * F.l1_loss(F.avg_pool2d(image, 8), F.avg_pool2d(photos, 8))
    loss = photometric + 1e-2 * table(rows).pow(2).sum(1).mean()
    for level, resolution in enumerate(basis):
        scale = resolution / basis[0]
        loss = loss + texture_basis_penalty(texture.tables[level], scale * basis_decay, scale * basis_smoothness)
    network = list(network_parameters.values())
    groups = [{"params": network, "lr": 1e-4}, {"params": [table.weight], "lr": 5e-3, "betas": (0.9, 0.99)}]
    if basis:
        groups.append({"params": list(texture_parameters.values()), "lr": 1e-2, "betas": (0.9, 0.99)})
    optimiser = torch.optim.Adam(groups, lr=1e-4, betas=(0.5, 0.999))
    optimiser.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network, 1.0)
    for name, value in parameters.items():
        tensors["gradient." + name] = value.grad.detach().clone()
    optimiser.step()
    for name, value in parameters.items():
        tensors["after." + name] = value.detach().clone()
    tensors.update({"image": image, "photometric": photometric.reshape(1), "perceptual": torch.zeros(1),
                    "kl": torch.zeros(1)})
    save(out, case, tensors)
    (out / f"{case}_parameters.txt").write_text("\n".join(parameters) + "\n")
    description = f"width {size}\nheight {size}\nlevels 0\nfeatures 2\ntable 1\nresolutions\nclasses 3\n"
    if basis:
        description += f"basis {texture.channels} 6 " + " ".join(str(r) for r in basis) + "\n"
        if basis_decay > 0 or basis_smoothness > 0:
            description += f"basis-regularize {basis_decay} {basis_smoothness}\n"
    else:
        description += f"mlp {texture.first.frequencies} 16 {texture.channels} 2\n"
        if residual:
            description += "mlp-residual" + "".join(f" {f} {w}" for f, w in residual) + "\n"
    (out / f"{case}_texture.txt").write_text(description)


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


def temporal_case(out: Path, seed: int) -> None:
    """One latent renderer step as latent_cases()' with the temporal consistency loss (weight 0.5): a second view of
    each sample rendered with the same latents, the first render warped onto it by a given flow and mask."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from hash_texture import SurfaceTexture
    from latent_renderer import LatentRenderer, flow_warp_loss, kl_divergence
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
    # The second views: other geometry and coordinates, the crate shifted, and their flow onto the first views.
    geometry2 = torch.rand(b, 4, size, size, generator=generator)
    coordinates2 = torch.rand(b, 3, size, size, generator=generator)
    classes2 = torch.zeros(b, size, size, dtype=torch.long)
    classes2[:, 8:28, 6:24] = 3
    objects2 = (classes2 == 3).long()
    flow = torch.stack([torch.rand(b, size, size, generator=generator) * (size + 4) - 2,
                        torch.rand(b, size, size, generator=generator) * (size + 4) - 2], dim=1)
    valid = (torch.rand(b, size, size, generator=generator) > 0.3).float()
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
    second = model(geometry2, coordinates2, classes2, objects2, latents.detach())
    temporal = flow_warp_loss(image, second, flow, valid)
    loss = photometric + perceptual + 1e-2 * kl + 0.5 * temporal
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
                    "kl": kl.reshape(1), "mu": mu, "log_variance": log_variance, "temporal": temporal.reshape(1),
                    "geometry2": geometry2, "coordinates2": coordinates2, "classes2": classes2.float(),
                    "objects2": objects2.float(), "flow": flow, "valid": valid, "second": second})
    save(out, "temporal_step", tensors)
    (out / "temporal_step_parameters.txt").write_text("\n".join(parameters) + "\n")
    grid = texture.grids[0]
    (out / "temporal_step_texture.txt").write_text(
        f"width {size}\nheight {size}\nlevels {grid.levels}\nfeatures {grid.features}\ntable {grid.table}\n"
        "resolutions " + " ".join(str(r) for r in grid.resolutions) + "\nclasses 3\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    generator_step_case(args.out, 14)
    generator_step_case(args.out, 14, "generator_step_vgg", 1.0)
    gan_step_case(args.out, 16)
    latent_cases(args.out, 17)
    latent_check_case(args.out, 18)
    temporal_case(args.out, 20)
    mlp_texture_cases(args.out, 21)
    mlp_texture_cases(args.out, 21, perceptual=False, case="mlp_texture_step_l1")
    latent_table_cases(args.out, 23)
    latent_table_cases(args.out, 25, residual=((5, 12), (7, 8)), case="residual_mlp_step")
    latent_table_cases(args.out, 29, case="basis_texture_step", basis=(4, 8))
    latent_table_cases(args.out, 31, case="basis_regularized_step", basis=(4, 8), basis_decay=0.02,
                       basis_smoothness=0.05)
    print(f"fixtures in {args.out}")


if __name__ == "__main__":
    main()
