# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""PyTorch references of Klartraum's training elements (klartraum::training), for the fixtures of
make_element_fixtures.py and for the training scripts that build networks of the same layers."""

from __future__ import annotations

import torch
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
