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
