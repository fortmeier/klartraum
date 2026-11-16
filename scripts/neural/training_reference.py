# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""PyTorch references of Klartraum's training elements (klartraum::training), for the fixtures of
make_element_fixtures.py and for the training scripts that build networks of the same layers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ChannelNorm(nn.Module):
    """Normalizes every pixel over its channels (a layer norm per pixel), with a learnable scale and shift per channel:
    unlike GroupNorm no statistics of the whole image, so each output depends on its receptive field alone."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = int(x.shape[1])
        mean = x.mean(dim=1, keepdim=True)
        centred = x - mean
        variance = (centred * centred).mean(dim=1, keepdim=True)
        normed = centred / torch.sqrt(variance + self.eps)
        return normed * self.weight.reshape(1, c, 1, 1) + self.bias.reshape(1, c, 1, 1)


class Vgg(nn.Module):
    """VGG-16 features (relu1_2, relu2_2, relu3_3, relu4_3) of ImageNet-normalised images, for the perceptual loss."""

    def __init__(self):
        super().__init__()
        from torchvision.models import VGG16_Weights, vgg16

        features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
        self.slices = nn.ModuleList([features[:4], features[4:9], features[9:16], features[16:23]])
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.requires_grad_(False)

    def distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x, y = (a - self.mean) / self.std, (b - self.mean) / self.std
        total = 0.0
        for weight, s in zip((1 / 32, 1 / 16, 1 / 8, 1 / 4), self.slices):
            x, y = s(x), s(y)
            total = total + weight * F.l1_loss(x, y)
        return total


def augment_with(pair: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
    """DiffAugment of discriminator pairs (G-buffer channels, then an image) with its random numbers given, per
    sample: brightness, contrast (both on the image only), the roll (y, x) and the cutout centre (y, x)."""
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


def kl_divergence(mu: torch.Tensor, log_variance: torch.Tensor, counts: torch.Tensor,
                  free_bits: float = 0.0) -> torch.Tensor:
    """KL(N(mu, sigma) || N(0, 1)) summed over the latent and the objects that have pixels, averaged over the batch.

    With free bits, each latent number's KL counts as at least free_bits nats, so the first free_bits nats of each
    number cost nothing (Kingma et al. 2016)."""
    terms = (-0.5 * (1.0 + log_variance - mu * mu - log_variance.exp())).clamp(min=free_bits).sum(dim=2)
    return (terms * (counts > 0).to(terms.dtype)).sum() / mu.shape[0]


def flow_warp_loss(first: torch.Tensor, second: torch.Tensor, flow: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """The temporal consistency loss: the first render [N, 3, H, W] warped onto the second view by the flow [N, 2, H, W]
    (each of the second view's pixels' position in the first, in pixels with centres at i + 0.5; bilinear, clamped to
    the border), and the mean over all of the second render's values of valid [N, H, W] x the squared difference.
    The warped first render is the target: no gradient into it."""
    height, width = second.shape[2:]
    grid = torch.stack([flow[:, 0] / width * 2.0 - 1.0, flow[:, 1] / height * 2.0 - 1.0], dim=-1)
    warped = F.grid_sample(first.detach(), grid, mode="bilinear", padding_mode="border", align_corners=False)
    return ((warped - second) ** 2 * valid[:, None]).mean()


def positional_encoding(coordinates: torch.Tensor, classes: torch.Tensor, frequencies: int,
                        texture_classes: tuple[int, ...]) -> torch.Tensor:
    """coordinates [N, 3, H, W] (u, v, id in [0, 1]) -> [N, 3 (2 F + 1), H, W]: per coordinate c and frequency
    f < F the pair sin(2^f pi x_c), cos(2^f pi x_c), then the three raw coordinates; zero where the pixel's class
    has no texture."""
    channels = []
    for c in range(3):
        for f in range(frequencies):
            angle = (2.0 ** f) * torch.pi * coordinates[:, c]
            channels += [torch.sin(angle), torch.cos(angle)]
    channels += [coordinates[:, c] for c in range(3)]
    mask = torch.zeros_like(classes, dtype=coordinates.dtype)
    for texture_class in texture_classes:
        mask = mask + (classes == texture_class).to(coordinates.dtype)
    return torch.stack(channels, dim=1) * mask[:, None]


def sample_texture_basis(tables: torch.Tensor, coordinates: torch.Tensor, classes: torch.Tensor,
                         latent_gbuffer: torch.Tensor, textured: tuple[int, ...] = (3,)) -> torch.Tensor:
    """The code-mixed feature texture of one resolution: `tables` [faces, layers, channels, R, R]; every pixel of a
    textured class gets its face's (floor(id x faces) of the third coordinate) layer 0 plus layer k + 1 times its
    latent k, sampled bilinearly at (u, v) with texel centres at (i + 0.5) / R, clamped; zero elsewhere."""
    faces, layers, channels, resolution, _ = tables.shape
    n, _, h, w = coordinates.shape
    u, v, ids = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]
    face = (ids * faces).floor().clamp(0, faces - 1).long()
    x = (u * resolution - 0.5).clamp(0, resolution - 1)
    y = (v * resolution - 0.5).clamp(0, resolution - 1)
    x0, y0 = x.floor().long(), y.floor().long()
    x1, y1 = (x0 + 1).clamp(max=resolution - 1), (y0 + 1).clamp(max=resolution - 1)
    ax, ay = (x - x0.float())[:, None], (y - y0.float())[:, None]
    flat = tables.reshape(faces, layers * channels, resolution * resolution)

    def at(yy, xx):  # [n, layers * channels, h, w]
        index = (yy * resolution + xx).reshape(n, 1, h * w).expand(n, layers * channels, h * w)
        rows = flat[face.reshape(n, h * w)]  # [n, h w, layers * channels, R R]
        picked = rows.gather(3, index.permute(0, 2, 1).unsqueeze(3)).squeeze(3)  # [n, h w, layers * channels]
        return picked.permute(0, 2, 1).reshape(n, layers * channels, h, w)

    sample = ((1 - ax) * (1 - ay) * at(y0, x0) + ax * (1 - ay) * at(y0, x1) + (1 - ax) * ay * at(y1, x0)
              + ax * ay * at(y1, x1)).reshape(n, layers, channels, h, w)
    scales = torch.cat([torch.ones(n, 1, h, w, dtype=sample.dtype), latent_gbuffer], dim=1)
    features = (sample * scales[:, :, None]).sum(1)
    mask = torch.zeros_like(classes, dtype=sample.dtype)
    for c in textured:
        mask = mask + (classes == c).to(sample.dtype)
    return features * mask[:, None]


def texture_basis_penalty(table: torch.Tensor, decay: float, smoothness: float) -> torch.Tensor:
    """decay x the sum of squared texels + smoothness x the sum of squared differences between neighbouring texels of
    the same plane (the last two dimensions), horizontally and vertically."""
    dx = table[..., :, 1:] - table[..., :, :-1]
    dy = table[..., 1:, :] - table[..., :-1, :]
    return decay * table.pow(2).sum() + smoothness * (dx.pow(2).sum() + dy.pow(2).sum())
