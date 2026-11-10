# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""A learnable 3D texture over surface coordinates (u, v, object id), encoded as a multi-resolution hash grid
(Mueller et al., Instant Neural Graphics Primitives, 2022), in PyTorch so that it trains on MPS and gradients from any
loss reach the table entries a pixel uses (deferred neural rendering, Thies et al. 2019).

Level l has the resolution floor(base * growth^l) per axis (growth so that the last level reaches `finest`); a point
is looked up at the 8 corners of its cell, by direct index where the level's grid fits the table and by the spatial
hash of Instant NGP (x * 1 xor y * 2654435761 xor z * 805459861 mod T) where it does not, and the corners' features
are blended trilinearly. The output is the levels' features side by side, [N, levels * features].

SurfaceTexture holds one grid per surface class (the renderer's stems, leaves, ground) and gives the pixels of other
classes zero features.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

PRIMES = (1, 2654435761, 805459861)


class HashGrid3D(nn.Module):
    def __init__(self, levels: int = 12, features: int = 2, log2_table: int = 17, base: int = 16, finest: int = 2048):
        super().__init__()
        self.levels, self.features, self.table = levels, features, 1 << log2_table
        growth = math.exp((math.log(finest) - math.log(base)) / max(levels - 1, 1))
        self.resolutions = [int(math.floor(base * growth**level)) for level in range(levels)]
        self.embeddings = nn.Parameter(torch.empty(levels, self.table, features).uniform_(-1e-4, 1e-4))

    def index(self, corner: torch.Tensor, level: int) -> torch.Tensor:
        """Table index of integer corners [N, 3] at `level`."""
        resolution = self.resolutions[level] + 1
        if resolution**3 <= self.table:
            return corner[:, 0] + resolution * (corner[:, 1] + resolution * corner[:, 2])
        hashed = corner[:, 0] * PRIMES[0]
        hashed = hashed ^ (corner[:, 1] * PRIMES[1])
        hashed = hashed ^ (corner[:, 2] * PRIMES[2])
        return hashed % self.table

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """points [N, 3] in [0, 1] -> features [N, levels * features]."""
        out = []
        for level, resolution in enumerate(self.resolutions):
            scaled = points * resolution
            cell = torch.floor(scaled).long().clamp(0, resolution - 1)
            weight = scaled - cell.to(points.dtype)
            features = torch.zeros(points.shape[0], self.features, device=points.device, dtype=points.dtype)
            for corner_bits in range(8):
                offset = torch.tensor([(corner_bits >> axis) & 1 for axis in range(3)], device=points.device)
                corner = cell + offset
                w = torch.where(offset.bool(), weight, 1.0 - weight).prod(dim=1, keepdim=True)
                features = features + w * self.embeddings[level][self.index(corner, level)]
            out.append(features)
        return torch.cat(out, dim=1)


class SurfaceTexture(nn.Module):
    """One HashGrid3D per textured surface class; per pixel the grid of its class at (u, v, id)."""

    def __init__(self, classes: tuple[int, ...] = (2, 3, 4), **grid):
        super().__init__()
        self.classes = classes
        self.grids = nn.ModuleList([HashGrid3D(**grid) for _ in classes])
        self.channels = self.grids[0].levels * self.grids[0].features

    def forward(self, coordinates: torch.Tensor, classes: torch.Tensor) -> torch.Tensor:
        """coordinates [B, 3, H, W] (u, v, id in [0, 1]) and classes [B, H, W] -> features [B, channels, H, W]."""
        b, _, h, w = coordinates.shape
        points = coordinates.permute(0, 2, 3, 1).reshape(-1, 3)
        flat_classes = classes.reshape(-1)
        out = torch.zeros(points.shape[0], self.channels, device=points.device, dtype=points.dtype)
        for c, grid in zip(self.classes, self.grids):
            mask = flat_classes == c
            if mask.any():
                out[mask] = grid(points[mask])
        return out.reshape(b, h, w, self.channels).permute(0, 3, 1, 2)
