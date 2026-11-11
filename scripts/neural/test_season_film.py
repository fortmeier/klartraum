# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""
TESTS:
- season_code_is_cyclic: season_code() of S and S + 4 agree, and winter (0) lies as close to late autumn (3.8) as to
  early spring (0.2)
- season_of_reads_labels: season_of() takes a flight's season from the table, else from a c<S>_ flight name, else
  summer
- new_film_is_identity: a generator with a fresh FiLM gives the same image as the same generator without FiLM, for
  every season
- film_follows_season: once the FiLM heads are trained (here: set at random), different seasons give different
  images and the same season the same one
- discriminator_sees_season: the season-conditioned discriminator's logits change with the season
- export_has_season_input: the ONNX export of a FiLM generator has the inputs "input" and "season" and writes the
  season's reference value

    uv run --project scripts/qwen_image python scripts/neural/test_season_film.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from train_photo_enhancer import (INPUT_CHANNELS, Generator, MultiScaleDiscriminator, TexturedModel, export,
                                  season_code, season_of)

WIDTHS = (8, 8, 8, 8, 8)


def season_code_is_cyclic() -> None:
    codes = season_code(torch.tensor([0.0, 4.0, 3.8, 0.2, 1.3, 5.3]))
    assert torch.allclose(codes[0], codes[1], atol=1e-6)
    assert torch.allclose(codes[4], codes[5], atol=1e-6)
    assert abs((codes[0] - codes[2]).norm() - (codes[0] - codes[3]).norm()) < 1e-5


def season_of_reads_labels() -> None:
    table = {"s113_winter": 0.25}
    assert season_of("s113_winter_0120", table) == 0.25
    assert season_of("c3.55_s71_0060", table) == 3.55
    assert season_of("seed11_az0.6_0320", table) == 2.0


def new_film_is_identity() -> None:
    torch.manual_seed(0)
    plain = Generator(WIDTHS).eval()
    film = Generator(WIDTHS, film=True).eval()
    film.load_state_dict(plain.state_dict(), strict=False)
    # The output convolution starts at zero (a grey image); give it weights so the comparison means something.
    torch.nn.init.normal_(plain.out.weight)
    film.out.load_state_dict(plain.out.state_dict())
    x = torch.rand(2, INPUT_CHANNELS, 32, 48)
    with torch.no_grad():
        expected = plain(x)
        for s in (0.0, 1.0, 2.5, 3.9):
            assert torch.allclose(film(x, season_code(torch.tensor([s, s]))), expected, atol=1e-6)


def film_follows_season() -> None:
    torch.manual_seed(1)
    model = Generator(WIDTHS, film=True).eval()
    torch.nn.init.normal_(model.out.weight)
    for head in list(model.film.gammas) + list(model.film.betas):
        torch.nn.init.normal_(head.weight, std=0.3)
    x = torch.rand(1, INPUT_CHANNELS, 32, 48).repeat(2, 1, 1, 1)
    with torch.no_grad():
        both = model(x, season_code(torch.tensor([0.2, 2.2])))
        again = model(x, season_code(torch.tensor([0.2, 0.2])))
    assert (both[0] - both[1]).abs().mean() > 1e-3
    assert torch.allclose(again[0], again[1], atol=1e-6)
    assert torch.allclose(both[0], again[0], atol=1e-6)


def discriminator_sees_season() -> None:
    torch.manual_seed(2)
    discriminator = MultiScaleDiscriminator(season=True).eval()
    pair = torch.rand(1, INPUT_CHANNELS + 3, 64, 64)
    with torch.no_grad():
        winter = discriminator(pair, torch.tensor([0.2]))
        summer = discriminator(pair, torch.tensor([2.2]))
    assert (winter[0][1] - summer[0][1]).abs().max() > 1e-4


def export_has_season_input() -> None:
    import onnx

    model = TexturedModel(texture=False, film=True)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "enhancer.onnx"
        export(model, path, 32, 48)
        names = [i.name for i in onnx.load(path).graph.input]
        assert names[:2] == ["input", "season"], names
        assert path.with_name("enhancer_season_f32.bin").stat().st_size == 2 * 4


if __name__ == "__main__":
    for test in (season_code_is_cyclic, season_of_reads_labels, new_film_is_identity, film_follows_season,
                 discriminator_sees_season, export_has_season_input):
        test()
        print(f"ok  {test.__name__}")
