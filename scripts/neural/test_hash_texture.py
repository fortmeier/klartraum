# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""
TESTS:
- matches_reference: HashGrid3D's features of random points equal a straightforward NumPy implementation (direct
  index on coarse levels, Instant NGP's spatial hash on fine ones, trilinear blending of the 8 corners)
- corners_reproduce_entries: at a grid corner the features are exactly that corner's table entries
- gradients_reach_used_entries: a loss on the features of a few points has gradients exactly on the table entries of
  their cells' corners and nowhere else
- classes_select_grids: SurfaceTexture gives each class's pixels the features of its own grid and other classes zero

    uv run --project scripts/qwen_image python scripts/neural/test_hash_texture.py
"""

from __future__ import annotations

import numpy as np
import torch

from hash_texture import PRIMES, HashGrid3D, SurfaceTexture


def reference(grid: HashGrid3D, points: np.ndarray) -> np.ndarray:
    table = grid.embeddings.detach().numpy().astype(np.float64)
    out = []
    for level, resolution in enumerate(grid.resolutions):
        features = np.zeros((len(points), grid.features))
        for n, p in enumerate(points):
            scaled = p * resolution
            cell = np.clip(np.floor(scaled).astype(np.int64), 0, resolution - 1)
            weight = scaled - cell
            for bits in range(8):
                offset = np.array([(bits >> a) & 1 for a in range(3)])
                corner = cell + offset
                w = np.prod(np.where(offset == 1, weight, 1.0 - weight))
                r = resolution + 1
                if r**3 <= grid.table:
                    index = corner[0] + r * (corner[1] + r * corner[2])
                else:
                    index = (int(corner[0]) * PRIMES[0]) ^ (int(corner[1]) * PRIMES[1]) ^ (int(corner[2]) * PRIMES[2])
                    index %= grid.table
                features[n] += w * table[level, index]
        out.append(features)
    return np.concatenate(out, axis=1)


def matches_reference() -> None:
    torch.manual_seed(0)
    grid = HashGrid3D(levels=6, features=2, log2_table=12, base=4, finest=256)
    with torch.no_grad():
        grid.embeddings.uniform_(-1, 1)
    # float32 points on both sides: at resolution 256 their rounding alone moves the weights by ~1e-5.
    points = np.random.default_rng(1).random((50, 3)).astype(np.float32)
    actual = grid(torch.from_numpy(points)).detach().numpy()
    expected = reference(grid, points.astype(np.float64))
    # float32 positions scaled to 256 cells keep ~3e-5 of their fraction: the weights' precision.
    assert np.abs(actual - expected).max() < 1e-4, np.abs(actual - expected).max()
    assert any((r + 1) ** 3 > grid.table for r in grid.resolutions) and any((r + 1) ** 3 <= grid.table for r in grid.resolutions)


def corners_reproduce_entries() -> None:
    grid = HashGrid3D(levels=1, features=3, log2_table=12, base=8, finest=8)
    with torch.no_grad():
        grid.embeddings.uniform_(-1, 1)
    corner = torch.tensor([[3, 5, 2]])
    point = corner.float() / grid.resolutions[0]
    expected = grid.embeddings[0][grid.index(corner, 0)]
    assert torch.allclose(grid(point), expected, atol=1e-6)


def gradients_reach_used_entries() -> None:
    grid = HashGrid3D(levels=2, features=2, log2_table=10, base=4, finest=64)
    points = torch.tensor([[0.31, 0.62, 0.17], [0.81, 0.11, 0.55]])
    grid(points).sum().backward()
    used = set()
    for level, resolution in enumerate(grid.resolutions):
        cells = torch.floor(points * resolution).long().clamp(0, resolution - 1)
        for bits in range(8):
            offset = torch.tensor([(bits >> a) & 1 for a in range(3)])
            for index in grid.index(cells + offset, level).tolist():
                used.add((level, index))
    nonzero = {(level, index) for level, index in zip(*torch.nonzero(grid.embeddings.grad.abs().sum(-1)).T.tolist())}
    assert nonzero <= used and len(nonzero) > 0, (len(nonzero), len(used))


def classes_select_grids() -> None:
    texture = SurfaceTexture(classes=(3, 4), levels=2, features=2, log2_table=10, base=4, finest=16)
    with torch.no_grad():
        texture.grids[0].embeddings.fill_(1.0)
        texture.grids[1].embeddings.fill_(2.0)
    coordinates = torch.rand(1, 3, 2, 3)
    classes = torch.tensor([[[3, 4, 0], [4, 3, 2]]])
    features = texture(coordinates, classes)
    assert torch.allclose(features[0, :, 0, 0], torch.ones(4))
    assert torch.allclose(features[0, :, 0, 1], torch.full((4,), 2.0))
    assert torch.allclose(features[0, :, 0, 2], torch.zeros(4))
    assert torch.allclose(features[0, :, 1, 2], torch.zeros(4))


if __name__ == "__main__":
    for test in (matches_reference, corners_reproduce_entries, gradients_reach_used_entries, classes_select_grids):
        test()
        print(f"ok  {test.__name__}")
