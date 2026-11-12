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


def object_mean(values: torch.Tensor, objects: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """values [N, C, h, w], object indices [N, h, w] (0: none) -> the mean of each object's pixels [N, count, C] (0
    for objects without pixels) and the pixel counts [N, count]."""
    n, c = values.shape[:2]
    means = torch.zeros(n, count, c, dtype=values.dtype)
    counts = torch.zeros(n, count)
    for k in range(count):
        mask = (objects == k + 1).to(values.dtype)[:, None]
        pixels = mask.sum(dim=(1, 2, 3))
        counts[:, k] = pixels
        means[:, k] = (values * mask).sum(dim=(2, 3)) / pixels.clamp(min=1.0)[:, None]
    return means, counts


def splat(latents: torch.Tensor, objects: torch.Tensor) -> torch.Tensor:
    """latents [N, count, D], object indices [N, H, W] -> the latent G-buffer [N, D, H, W] (0 off the objects)."""
    n, count, d = latents.shape
    out = torch.zeros(n, d, *objects.shape[1:], dtype=latents.dtype)
    for k in range(count):
        mask = (objects == k + 1).to(latents.dtype)[:, None]
        out = out + mask * latents[:, k, :, None, None]
    return out


class LatentRenderer(nn.Module):
    """Texture (hash_texture.SurfaceTexture), encoder and decoder (the local U-Net of train_photo_enhancer.py:
    3 levels, ChannelNorm) on geometry channels, texture features and the latent G-buffer."""

    def __init__(self, texture: nn.Module, geometry: int = 4, latent: int = 4, objects: int = 1,
                 widths: tuple[int, ...] = (32, 64, 128)):
        super().__init__()
        self.texture = texture
        self.latent, self.objects = latent, objects
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
        x = torch.cat([geometry, self.texture(coordinates, classes), splat(latents, objects)], dim=1)
        return self.generator(x)


def kl_divergence(mu: torch.Tensor, log_variance: torch.Tensor, counts: torch.Tensor,
                  free_bits: float = 0.0) -> torch.Tensor:
    """KL(N(mu, sigma) || N(0, 1)) summed over the latent and the objects that have pixels, averaged over the batch.

    With free bits, each latent number's KL counts as at least free_bits nats, so the first free_bits nats of each
    number cost nothing (Kingma et al. 2016)."""
    terms = (-0.5 * (1.0 + log_variance - mu * mu - log_variance.exp())).clamp(min=free_bits).sum(dim=2)
    return (terms * (counts > 0).to(terms.dtype)).sum() / mu.shape[0]
