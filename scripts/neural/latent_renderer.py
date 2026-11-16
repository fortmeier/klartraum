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


def kl_divergence(mu: torch.Tensor, log_variance: torch.Tensor, counts: torch.Tensor,
                  free_bits: float = 0.0) -> torch.Tensor:
    """KL(N(mu, sigma) || N(0, 1)) summed over the latent and the objects that have pixels, averaged over the batch.

    With free bits, each latent number's KL counts as at least free_bits nats, so the first free_bits nats of each
    number cost nothing (Kingma et al. 2016)."""
    terms = (-0.5 * (1.0 + log_variance - mu * mu - log_variance.exp())).clamp(min=free_bits).sum(dim=2)
    return (terms * (counts > 0).to(terms.dtype)).sum() / mu.shape[0]


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


def flow_warp_loss(first: torch.Tensor, second: torch.Tensor, flow: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """The temporal consistency loss: the first render [N, 3, H, W] warped onto the second view by the flow [N, 2, H, W]
    (each of the second view's pixels' position in the first, in pixels with centres at i + 0.5; bilinear, clamped to
    the border), and the mean over all of the second render's values of valid [N, H, W] x the squared difference.
    The warped first render is the target: no gradient into it."""
    height, width = second.shape[2:]
    grid = torch.stack([flow[:, 0] / width * 2.0 - 1.0, flow[:, 1] / height * 2.0 - 1.0], dim=-1)
    warped = F.grid_sample(first.detach(), grid, mode="bilinear", padding_mode="border", align_corners=False)
    return ((warped - second) ** 2 * valid[:, None]).mean()


class AlbedoTexture(nn.Module):
    """The multi-resolution albedo texture: residual blocks, each an MLP (1 x 1 convolutions with ELU, `hidden`
    layers of its width) on the positional encoding of its octaves and the latent G-buffer, giving 3 values; their
    sum, each block weighted (a coarse-to-fine fade-in), is the albedo's logit."""

    def __init__(self, latent: int, blocks: tuple[tuple[int, int], ...] = ((2, 8), (4, 16), (6, 32)), hidden: int = 2,
                 classes: tuple[int, ...] = (3,)):
        super().__init__()
        self.classes, self.octaves = classes, [frequencies for frequencies, _ in blocks]
        self.blocks = nn.ModuleList()
        for frequencies, width in blocks:
            layers, previous = [], 3 * (2 * frequencies + 1) + latent
            for _ in range(hidden):
                layers += [nn.Conv2d(previous, width, 1), nn.ELU()]
                previous = width
            layers.append(nn.Conv2d(previous, 3, 1))
            self.blocks.append(nn.Sequential(*layers))

    def forward(self, coordinates, classes, latent_gbuffer, weights=None):
        logit = 0.0
        for k, (block, frequencies) in enumerate(zip(self.blocks, self.octaves)):
            encoding = positional_encoding(coordinates, classes, frequencies, self.classes)
            weight = 1.0 if weights is None else weights[k]
            logit = logit + weight * block(torch.cat([encoding, latent_gbuffer], dim=1))
        return logit


def compose(albedo_logit: torch.Tensor, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The image from the albedo's logit [N, 3, H, W], the decoder's 4 raw outputs (shading, background RGB) and the
    object mask [N, H, W]: mask x sigmoid(albedo) x 2 sigmoid(shading) + (1 - mask) x sigmoid(background)."""
    m = mask[:, None]
    shading = 2.0 * torch.sigmoid(logits[:, :1])
    return m * torch.sigmoid(albedo_logit) * shading + (1.0 - m) * torch.sigmoid(logits[:, 1:])


class AlbedoRenderer(LatentRenderer):
    """The latent renderer with the albedo texture: the decoder sees geometry and the latent G-buffer only and gives
    shading and background; the texture gives the albedo (compose())."""

    def __init__(self, texture: AlbedoTexture, geometry: int = 4, latent: int = 4, objects: int = 1,
                 widths: tuple[int, ...] = (32, 64, 128)):
        texture.channels = 0
        super().__init__(texture, geometry, latent, objects, widths)
        self.generator.out = nn.Conv2d(widths[0], 4, 1)
        nn.init.zeros_(self.generator.out.weight)
        nn.init.zeros_(self.generator.out.bias)
        self.generator.raw = True
        self.weights = None

    def forward(self, geometry, coordinates, classes, objects, latents):
        latent_gbuffer = splat(latents, objects)
        logits = self.generator(torch.cat([geometry, latent_gbuffer], dim=1))
        albedo_logit = self.texture(coordinates, classes, latent_gbuffer, self.weights)
        return compose(albedo_logit, logits, (objects > 0).to(geometry.dtype))
