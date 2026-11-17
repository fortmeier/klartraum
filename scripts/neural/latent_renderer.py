# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""The latent-conditioned neural texture renderer (the reference for Klartraum's latent mode of
neural::PhotoEnhancerTraining, used by its tests): per pixel g = (geometry channels, texture features t), the decoder
renders I = f(g | L) from g and a latent G-buffer L that carries every object's latent at the object's pixels. The
latents come from a VAE encoder of the photo J: convolutions down to 1/8, a 1 x 1 head to mean and log-variance per
pixel, averaged over each object's pixels, L = mu + sigma eps, and the KL divergence to N(0, 1).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


from train_photo_enhancer import ChannelNorm, Generator, INPUT_CHANNELS
from training_reference import (  # noqa: F401 (re-exported for the fixtures and scripts)
    flow_warp_loss,
    kl_divergence,
    object_mean,
    positional_encoding,
    sample_texture_basis,
    splat,
    texture_basis_penalty,
)


class Encoder(nn.Module):
    """J [N, 3, H, W] -> mean and log-variance [N, 2 latent, H/8, W/8]: three 3 x 3 stride-2 convolutions with
    ChannelNorm and ELU, then a 1 x 1 head."""

    def __init__(self, latent: int = 4, widths: tuple[int, ...] = (32, 64, 128)):
        super().__init__()
        layers, previous = [], 3
        for width in widths:
            layers += [nn.Conv2d(previous, width, 3, stride=2, padding=1), ChannelNorm(width), nn.ELU()]
            previous = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(previous, 2 * latent, 1)

    def forward(self, photo: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(photo))


class LatentRenderer(nn.Module):
    """Texture (hash_texture.SurfaceTexture), encoder and decoder (the local U-Net of train_photo_enhancer.py:
    3 levels, ChannelNorm) on geometry channels, texture features and the latent G-buffer."""

    def __init__(self, texture: nn.Module, geometry: int = 4, latent: int = 4, objects: int = 1,
                 widths: tuple[int, ...] = (32, 64, 128), latent_to_decoder: bool = True):
        super().__init__()
        self.texture = texture
        self.latent, self.objects = latent, objects
        # False: only the texture sees the latents; the generator's latent channels are zero.
        self.latent_to_decoder = latent_to_decoder
        channels = geometry + texture.channels + latent
        self.generator = Generator(widths=widths, extra=channels - INPUT_CHANNELS, blur_depth=False, norm="channel")
        self.encoder = Encoder(latent)

    def encode(self, photo: torch.Tensor, objects: torch.Tensor):
        """The objects' mean and log-variance [N, objects, latent] from the photo and the full-resolution object map
        (sampled at the encoder's pixel centres: row and column 8 k + 4)."""
        statistics = self.encoder(photo)
        pooled, counts = object_mean(statistics, objects[:, 4::8, 4::8], self.objects)
        return pooled[..., :self.latent], pooled[..., self.latent:], counts

    def forward(self, geometry, coordinates, classes, objects, latents):
        latent_gbuffer = splat(latents, objects)
        if getattr(self.texture, "conditioned", False):
            features = self.texture(coordinates, classes, latent_gbuffer)
        else:
            features = self.texture(coordinates, classes)
        decoder_latents = latent_gbuffer if self.latent_to_decoder else torch.zeros_like(latent_gbuffer)
        x = torch.cat([geometry, features, decoder_latents], dim=1)
        return self.generator(x)


class MlpTexture(nn.Module):
    """The latent-conditioned texture: a per-pixel MLP (1 x 1 convolutions with ELU) from the positional encoding of
    the surface coordinates and the latent G-buffer to `outputs` texture features."""

    conditioned = True

    def __init__(self, latent: int, frequencies: int = 6, width: int = 32, hidden: int = 2, outputs: int = 8,
                 classes: tuple[int, ...] = (3,)):
        super().__init__()
        self.frequencies, self.classes, self.channels = frequencies, classes, outputs
        layers, previous = [], 3 * (2 * frequencies + 1) + latent
        for _ in range(hidden):
            layers += [nn.Conv2d(previous, width, 1), nn.ELU()]
            previous = width
        layers.append(nn.Conv2d(previous, outputs, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, coordinates, classes, latent_gbuffer):
        encoding = positional_encoding(coordinates, classes, self.frequencies, self.classes)
        return self.mlp(torch.cat([encoding, latent_gbuffer], dim=1))


class ResidualMlpTexture(nn.Module):
    """A multi-resolution MLP texture: a first MlpTexture and residual blocks (each an MlpTexture of its own octaves
    and hidden width, `residual` pairs of frequencies and width), all giving `outputs` features that are summed, so
    the finer blocks correct what the coarser ones leave."""

    conditioned = True

    def __init__(self, latent: int, frequencies: int = 6, width: int = 32, hidden: int = 2, outputs: int = 8,
                 classes: tuple[int, ...] = (3,), residual: tuple[tuple[int, int], ...] = ()):
        super().__init__()
        self.channels = outputs
        self.first = MlpTexture(latent, frequencies, width, hidden, outputs, classes)
        self.residual = nn.ModuleList(MlpTexture(latent, f, w, hidden, outputs, classes) for f, w in residual)

    def forward(self, coordinates, classes, latent_gbuffer):
        features = self.first(coordinates, classes, latent_gbuffer)
        for block in self.residual:
            features = features + block(coordinates, classes, latent_gbuffer)
        return features


class TextureBasis(nn.Module):
    """The code-mixed feature texture (as Deferred Neural Rendering's neural texture, mixed by the latent): per face
    and resolution a base layer and one layer per latent number of `outputs` features, the resolutions summed."""

    conditioned = True

    def __init__(self, latent: int, outputs: int = 8, resolutions: tuple[int, ...] = (16, 32, 64), faces: int = 6,
                 classes: tuple[int, ...] = (3,)):
        super().__init__()
        self.channels, self.classes = outputs, classes
        self.tables = nn.ParameterList(
            nn.Parameter((torch.rand(faces, latent + 1, outputs, r, r) * 2 - 1) * 1e-4) for r in resolutions)

    def forward(self, coordinates, classes, latent_gbuffer):
        features = 0
        for table in self.tables:
            features = features + sample_texture_basis(table, coordinates, classes, latent_gbuffer, self.classes)
        return features


def latent_check(model: LatentRenderer, geometry: torch.Tensor, coordinates: torch.Tensor, classes: torch.Tensor,
                 objects: torch.Tensor, latents: torch.Tensor, permutation: torch.Tensor):
    """The latent check: every view rendered again with the latents of view permutation[k] (the same latents in another
    pose), encoded by the encoder held fixed, and the mean absolute difference between the encoded means and those
    latents. The latents are targets (no gradient); the gradient reaches the decoder and the texture only.
    Returns the loss and the second renders."""
    swapped = latents[permutation].detach()
    image = model(geometry, coordinates, classes, objects, swapped)
    encoder = list(model.encoder.parameters())
    for parameter in encoder:
        parameter.requires_grad_(False)
    statistics = model.encoder(image)
    for parameter in encoder:
        parameter.requires_grad_(True)
    pooled, _ = object_mean(statistics, objects[:, 4::8, 4::8], model.objects)
    return F.l1_loss(pooled[..., :model.latent], swapped), image
