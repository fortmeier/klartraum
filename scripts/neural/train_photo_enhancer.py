# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "torchvision", "numpy", "pillow", "onnx"]
# ///

"""Trains the photo enhancer: a conditional GAN that turns a rendering of the bonsai landscape and its G-buffers
into a photo-realistic image, learnt from Qwen-Image photographs made from the same views (text-to-image with the
rendering's depth as ControlNet condition; prepare_photo_pairs.py, qwen_image_example --batch).

Input (11 channels): the rendering's RGB, the inverse depth as the ControlNet saw it, and the one-hot rendered class
(TREE_CLASS_*, 7). The generator is a U-Net (GroupNorm, ELU, bilinear upsampling, no noise input, so equal inputs give
equal outputs from frame to frame). Losses: L1, VGG-16 perceptual distance and the discriminator's feature matching
(pix2pixHD), and the non-saturating adversarial loss of a conditional PatchGAN at two scales with R1 on real pairs;
the discriminator sees DiffAugment's translation and cutout (on input and image together) and colour (on the image).

Samples whose photo was not written yet are skipped; the training split is read again every --reload steps, so
training can run while more photos are made. Every --every steps it writes validation numbers (L1 and VGG distance
on the held-out split), a sample grid (input | photo | output) and the ONNX model (input [1, 11, 464, 832] in
[0, 1] -> image [1, 3, 464, 832] in [0, 1]) into the output directory.

    uv run --project scripts/qwen_image python scripts/neural/train_photo_enhancer.py build/TestingOutput/gan_data \\
        build/TestingOutput/gan_runs/v1 [--steps 40000]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

CLASSES = 7
INPUT_CHANNELS = 3 + 1 + CLASSES


def usable(classes: np.ndarray) -> bool:
    """Whether a view shows landscape: hardly any pot (around which Qwen-Image invents a scene of its own and whose
    inside it paints freely) and at most 60 % background (the sky, which it paints as sky)."""
    return (classes == 1).mean() <= 0.02 and (classes == 0).mean() <= 0.6


MIN_ALIGNMENT = 0.0  # set from --min-alignment
SUMMER = 2.0  # the season of flights without a known one


def season_of(sample: str, seasons: dict[str, float]) -> float:
    """The season S in [0, 4) (0 winter, 1 spring, 2 summer, 3 autumn, as tree::treeSeasonWrap()) of a sample
    <flight>_<frame>: from the data's seasons.json ({flight: S}), else from a flight named c<S>_..., else summer."""
    flight = sample.rsplit("_", 1)[0]
    if flight in seasons:
        return float(seasons[flight])
    if flight.startswith("c"):
        try:
            return float(flight[1:].split("_", 1)[0])
        except ValueError:
            pass
    return SUMMER


def season_code(season: torch.Tensor) -> torch.Tensor:
    """[N] seasons -> [N, 2] (sin, cos) of the season's angle on the year, so winter follows autumn."""
    angle = season.float() * (torch.pi / 2.0)
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=1)
TEXTURE = False  # set from --texture: samples need <split>/coords/<sample>.npz (add_surface_coordinates.py)


def load_split(directory: Path):
    """The usable samples with a photo: their ids, inputs [N, 5, H, W] (uint8 RGB, depth, class index), photos,
    with TEXTURE their surface coordinates [N, 4, H, W] float32 (u, v, object id, precise inverse depth; samples
    without them are left out) else None, and their seasons [N] (season_of() with <data>/seasons.json).
    With MIN_ALIGNMENT, samples whose photo follows its rendering's depth worse than that (check_photo_alignment.py's
    alignment.json; samples not scored yet count as aligned) are left out."""
    alignment_path = directory / "alignment.json"
    alignment = json.loads(alignment_path.read_text()) if alignment_path.exists() else {}
    # Samples found faulty by eye (e.g. a symbol painted where the 8-bit depth shows contour lines).
    reject_path = directory / "reject.txt"
    rejected = set(reject_path.read_text().split()) if reject_path.exists() else set()
    seasons_path = directory.parent / "seasons.json"
    season_table = json.loads(seasons_path.read_text()) if seasons_path.exists() else {}
    ids, inputs, photos, coords, seasons = [], [], [], [], []
    for photo in sorted((directory / "photo").glob("*.ppm")):
        sample = photo.stem
        if alignment.get(sample, 1.0) < MIN_ALIGNMENT or sample in rejected:
            continue
        coords_path = directory / "coords" / f"{sample}.npz"
        if TEXTURE and not coords_path.exists():
            continue
        try:
            target = np.asarray(Image.open(photo).convert("RGB"))
        except Exception:
            continue
        render = np.asarray(Image.open(directory / "render" / f"{sample}.ppm").convert("RGB"))
        depth = np.asarray(Image.open(directory / "depth" / f"{sample}.ppm").convert("L"))
        classes = np.asarray(Image.open(directory / "class" / f"{sample}.pgm"))
        if not usable(classes):
            continue
        inputs.append(np.concatenate([render, depth[..., None], classes[..., None]], axis=2))
        photos.append(target)
        ids.append(sample)
        seasons.append(season_of(sample, season_table))
        if TEXTURE:
            stored = np.load(coords_path)
            coords.append(np.concatenate([stored["uvid"], stored["depth"][None]]).astype(np.float32))
    if not ids:
        return ids, torch.empty(0), torch.empty(0), None, torch.empty(0)
    return (ids, torch.from_numpy(np.stack(inputs)).permute(0, 3, 1, 2).contiguous(),
            torch.from_numpy(np.stack(photos)).permute(0, 3, 1, 2).contiguous(),
            torch.from_numpy(np.stack(coords)) if TEXTURE else None, torch.tensor(seasons, dtype=torch.float32))


def network_input(raw: torch.Tensor, coords: torch.Tensor | None = None) -> torch.Tensor:
    """[N, 5, H, W] uint8 (RGB, depth, class) -> [N, 11, H, W] float in [0, 1]; with coords, the precise inverse depth
    (their channel 3) in place of the 8-bit one."""
    rgb_depth = raw[:, :4].float() / 255.0
    if coords is not None:
        rgb_depth = torch.cat([rgb_depth[:, :3], coords[:, 3:4].to(rgb_depth.device)], dim=1)
    one_hot = F.one_hot(raw[:, 4].long().clamp(0, CLASSES - 1), CLASSES).permute(0, 3, 1, 2).float()
    return torch.cat([rgb_depth, one_hot], dim=1)


class GroupNorm(nn.GroupNorm):
    """nn.GroupNorm written with the tensor's sizes as Python numbers, so that its ONNX export holds no Shape
    operations (only Reshape, ReduceMean and arithmetic, which Klartraum imports)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = (int(v) for v in x.shape)
        grouped = x.reshape(n, self.num_groups, (c // self.num_groups) * h * w)
        mean = grouped.mean(dim=2, keepdim=True)
        centred = grouped - mean
        variance = (centred * centred).mean(dim=2, keepdim=True)
        normed = (centred / torch.sqrt(variance + self.eps)).reshape(n, c, h, w)
        return normed * self.weight.reshape(1, c, 1, 1) + self.bias.reshape(1, c, 1, 1)


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


class Block(nn.Sequential):
    """Two 3 x 3 convolutions, each followed by a normalization (which keeps the activations from growing from level to
    level; "group": GroupNorm, "channel": ChannelNorm) and ELU."""

    def __init__(self, inputs: int, outputs: int, stride: int = 1, norm: str = "group"):
        def normalization(channels: int) -> nn.Module:
            return GroupNorm(8, channels) if norm == "group" else ChannelNorm(channels)

        super().__init__(nn.Conv2d(inputs, outputs, 3, stride=stride, padding=1), normalization(outputs), nn.ELU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1), normalization(outputs), nn.ELU())


class SeasonFilm(nn.Module):
    """FiLM by a condition vector (the season's code season_code() and/or a per-image appearance code): a scale
    1 + gamma and a shift beta per channel of every block's output. The heads start at zero, so a new FiLM leaves the
    network as it was."""

    def __init__(self, widths: list[int], hidden: int = 64, inputs: int = 2):
        super().__init__()
        self.widths = list(widths)
        self.mlp = nn.Sequential(nn.Linear(inputs, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU())
        self.gammas = nn.ModuleList([nn.Linear(hidden, w) for w in widths])
        self.betas = nn.ModuleList([nn.Linear(hidden, w) for w in widths])
        for head in list(self.gammas) + list(self.betas):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, code: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """[N, inputs] -> per block (scale, shift), each [N, width, 1, 1]."""
        h = self.mlp(code)
        n = int(code.shape[0])
        return [((1.0 + g(h)).reshape(n, w, 1, 1), b(h).reshape(n, w, 1, 1))
                for g, b, w in zip(self.gammas, self.betas, self.widths)]


class Generator(nn.Module):
    """U-Net: 5 levels (1, 1/2, ..., 1/16), bilinear upsampling (no transposed convolutions, which leave grids); with
    `film` (the season) and/or `appearance` (the size of a per-image appearance code), every block's output is
    modulated by them (SeasonFilm)."""

    def __init__(self, widths=(32, 64, 128, 256, 256), extra: int = 0, blur_depth: bool = True, film: bool = False,
                 appearance: int = 0, norm: str = "group"):
        super().__init__()
        self.widths = tuple(widths)
        self.norm = norm
        self.blur_depth = blur_depth
        self.season = film
        self.appearance = appearance
        self.down = nn.ModuleList([Block(INPUT_CHANNELS + extra, widths[0], norm=norm)] +
                                  [Block(widths[i - 1], widths[i], stride=2, norm=norm) for i in range(1, len(widths))])
        self.up = nn.ModuleList([Block(widths[i] + widths[i - 1], widths[i - 1], norm=norm)
                                 for i in range(len(widths) - 1, 0, -1)])
        # The depth channel is smoothed first: its 8-bit steps would otherwise show as contour lines (rings on near
        # trunks). A fixed 9 x 9 Gaussian of sigma 2.5 px, not a parameter, so older checkpoints load.
        radius = torch.arange(-4, 5, dtype=torch.float32)
        gauss = torch.exp(-radius**2 / (2 * 2.5**2))
        kernel = torch.outer(gauss, gauss)
        self.register_buffer("depth_blur", (kernel / kernel.sum()).view(1, 1, 9, 9), persistent=False)
        inputs = (2 if film else 0) + appearance
        self.film = SeasonFilm(list(widths) + list(widths[-2::-1]), inputs=inputs) if inputs else None
        self.out = nn.Conv2d(widths[0], 3, 1)
        # Starts as a neutral grey, away from the sigmoid's flat ends.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, season: torch.Tensor | None = None,
                appearance: torch.Tensor | None = None) -> torch.Tensor:
        """x [N, channels, H, W]; season [N, 2] the season's code (season_code()), needed with the season's FiLM;
        appearance [N, D] the appearance code, needed with an appearance FiLM."""
        if self.blur_depth:
            depth = F.conv2d(x[:, 3:4], self.depth_blur, padding=4)
            x = torch.cat([x[:, :3], depth, x[:, 4:]], dim=1)
        condition = [c for c, used in ((season, self.season), (appearance, self.appearance > 0)) if used]
        # A single condition goes in as it is (an ONNX Concat of one input is not imported).
        joined = (condition[0] if len(condition) == 1 else torch.cat(condition, dim=1)) if condition else None
        modulation = iter(self.film(joined)) if self.film is not None else None

        def modulated(y: torch.Tensor) -> torch.Tensor:
            if modulation is None:
                return y
            scale, shift = next(modulation)
            return y * scale + shift

        skips = []
        for block in self.down:
            x = modulated(block(x))
            skips.append(x)
        x = skips.pop()
        for block in self.up:
            skip = skips.pop()
            # Sizes that are multiples of 16 halve exactly at every level, so each level doubles the previous one.
            x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
            x = modulated(block(torch.cat([x, skip], dim=1)))
        return torch.sigmoid(self.out(x))


class TexturedModel(nn.Module):
    """The generator, with a learnable surface texture (hash_texture.SurfaceTexture) whose features at each pixel's
    surface coordinates join its input (deferred neural rendering); without texture just the generator. With
    `appearances` > 0, a learnable appearance code of `appearance_size` per training image (in the order of
    `codes`' rows), as in NeRF in the Wild: what differs between the photographs (light, wear, the crate Qwen-Image
    painted) goes there, the texture stays shared."""

    def __init__(self, texture: bool = False, film: bool = False, appearances: int = 0, appearance_size: int = 0,
                 levels: int = 5, norm: str = "group"):
        super().__init__()
        from hash_texture import SurfaceTexture

        self.texture = SurfaceTexture() if texture else None
        extra = self.texture.channels if texture else 0
        # The precise depth of the coordinate G-buffers needs no smoothing. Fewer levels keep the generator local (3:
        # about 40 pixels around each output pixel), so that it cannot recognise a whole view and has to render from
        # the texture features and the geometry nearby (with norm "channel", which shares no statistics over the image).
        self.generator = Generator(widths=(32, 64, 128, 256, 256)[:levels], extra=extra, blur_depth=not texture,
                                   film=film, appearance=appearance_size if appearances else 0, norm=norm)
        self.codes = nn.Embedding(appearances, appearance_size) if appearances else None
        if self.codes is not None:
            nn.init.normal_(self.codes.weight, std=0.1)

    def forward(self, x: torch.Tensor, uvid: torch.Tensor | None = None, classes: torch.Tensor | None = None,
                season: torch.Tensor | None = None, appearance: torch.Tensor | None = None):
        """season: [N] seasons S, for a generator with the season's FiLM; appearance: [N] rows of the training
        images' codes (integers), or [N, appearance_size] codes themselves (e.g. fitted to a new image)."""
        if self.texture is not None:
            x = torch.cat([x, self.texture(uvid, classes)], dim=1)
        code = None
        if self.codes is not None:
            code = self.codes(appearance) if appearance.dtype in (torch.int32, torch.int64) else appearance
        return self.generator(x, season_code(season) if self.generator.season else None, code)


class PatchDiscriminator(nn.Module):
    """Conditional PatchGAN on the input's channels and an image; returns its layers' features (for feature
    matching) and the logits per patch."""

    def __init__(self, channels: int = INPUT_CHANNELS + 3):
        super().__init__()
        widths = [64, 128, 256, 512]
        layers, previous = [], channels
        for i, w in enumerate(widths):
            layers.append(nn.Sequential(nn.Conv2d(previous, w, 4, stride=2 if i < 3 else 1, padding=1),
                                        nn.LeakyReLU(0.2)))
            previous = w
        self.layers = nn.ModuleList(layers)
        self.logits = nn.Conv2d(previous, 1, 4, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        features = []
        x = x * 2.0 - 1.0
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features, self.logits(x)


class MultiScaleDiscriminator(nn.Module):
    """PatchGANs at full and half resolution; with `season`, the season's code joins the pair as two constant
    planes, so that the discriminator judges whether the image fits its season."""

    def __init__(self, scales: int = 2, season: bool = False):
        super().__init__()
        self.season = season
        channels = INPUT_CHANNELS + 3 + (2 if season else 0)
        self.discriminators = nn.ModuleList([PatchDiscriminator(channels) for _ in range(scales)])

    def forward(self, x: torch.Tensor, season: torch.Tensor | None = None):
        if self.season:
            planes = season_code(season).to(x.device)[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
            x = torch.cat([planes, x], dim=1)
        results = []
        for k, d in enumerate(self.discriminators):
            results.append(d(x if k == 0 else F.avg_pool2d(x, 2 ** k)))
        return results


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


def augment_pair(pair: torch.Tensor) -> torch.Tensor:
    """DiffAugment for the discriminator: colour on the image part, translation and cutout on input and image."""
    b, c, h, w = pair.shape
    dev = pair.device
    image = pair[:, -3:]
    image = image + (torch.rand(b, 1, 1, 1, device=dev) - 0.5) * 0.4
    mean = image.mean(dim=1, keepdim=True)
    image = (image - mean) * (torch.rand(b, 1, 1, 1, device=dev) * 1.0 + 0.5) + mean
    pair = torch.cat([pair[:, :-3], image], dim=1)
    shift = torch.randint(-w // 8, w // 8 + 1, (2,)).tolist()
    pair = torch.roll(pair, shifts=(shift[0] // 2, shift[1]), dims=(2, 3))
    cy, cx = int(torch.randint(0, h, (1,))), int(torch.randint(0, w, (1,)))
    mask = torch.ones(1, 1, h, w, device=dev)
    mask[:, :, max(0, cy - h // 4):cy + h // 4, max(0, cx - w // 4):cx + w // 4] = 0
    return pair * mask


def crops(inputs: torch.Tensor, photos: torch.Tensor, count: int, size: int, rng: np.random.Generator,
          coords: torch.Tensor | None = None, seasons: torch.Tensor | None = None):
    """Random crops (at scale 1 or 1/2 of the sample, flipped half the time): network inputs, photos, with coords
    the surface coordinates (u, v, id) else None, class indices for the texture (subsampled, never blended), the
    samples' seasons [count] and the samples' indices [count]."""
    xs, ys, uvids, classes, picked = [], [], [], [], []
    n, _, h, w = inputs.shape
    for _ in range(count):
        k = int(rng.integers(n))
        picked.append(k)
        c = coords[k:k + 1] if coords is not None else None
        x, y = network_input(inputs[k:k + 1], c), photos[k:k + 1].float() / 255.0
        uvid = c[:, :3] if c is not None else None
        cls = inputs[k:k + 1, 4].long()
        if rng.random() < 0.3 and min(h, w) // 2 >= size:
            x = F.interpolate(x, scale_factor=0.5, mode="area")
            y = F.interpolate(y, scale_factor=0.5, mode="area")
            uvid = uvid[:, :, ::2, ::2] if uvid is not None else None
            cls = cls[:, ::2, ::2]
        hh, ww = x.shape[2:]
        top, left = int(rng.integers(hh - size + 1)), int(rng.integers(ww - size + 1))
        x, y = x[:, :, top:top + size, left:left + size], y[:, :, top:top + size, left:left + size]
        cls = cls[:, top:top + size, left:left + size]
        uvid = uvid[:, :, top:top + size, left:left + size] if uvid is not None else None
        if rng.random() < 0.5:
            x, y, cls = x.flip(3), y.flip(3), cls.flip(2)
            uvid = uvid.flip(3) if uvid is not None else None
        # Dither the 8-bit depth by up to one grey level, so its steps carry no information.
        if coords is None:
            x[:, 3:4] = x[:, 3:4] + (torch.rand_like(x[:, 3:4]) - 0.5) / 255.0
        xs.append(x)
        ys.append(y)
        classes.append(cls)
        if uvid is not None:
            uvids.append(uvid)
    season = seasons[picked] if seasons is not None else torch.full((count,), SUMMER)
    return (torch.cat(xs), torch.cat(ys), (torch.cat(uvids) if uvids else None), torch.cat(classes), season,
            torch.tensor(picked))


def r1_penalty(discriminator: nn.Module, real_pair: torch.Tensor, season: torch.Tensor) -> torch.Tensor:
    """R1: the squared gradient of the discriminator's score per sample (the mean of its patch logits) with respect
    to the real pairs, averaged over the batch."""
    real_pair = real_pair.detach().requires_grad_(True)
    total = sum(logits.mean(dim=(1, 2, 3)).sum() for _, logits in discriminator(real_pair, season))
    (gradient,) = torch.autograd.grad(total, real_pair, create_graph=True)
    return gradient.pow(2).flatten(1).sum(1).mean()


def write_grid(path: Path, inputs: torch.Tensor, photos: torch.Tensor, outputs: torch.Tensor) -> None:
    rows = []
    for x, y, z in zip(inputs, photos, outputs):
        rows.append(torch.cat([x[:3], y, z], dim=2))
    grid = torch.cat(rows, dim=1).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    Image.fromarray((grid * 255).round().astype(np.uint8)).save(path)


def export(generator: nn.Module, path: Path, height: int, width: int, appearance_ids: list[str] = ()) -> None:
    """The generator as a fixed-shape ONNX graph Klartraum imports (shapes static, constants as initializers), with a
    reference input and output (`<path stem>_input_f32.bin`, `_output_f32.bin`) for the parity test. With FiLM, the
    graph has a second input "season" [1, 2], the season's code (season_code()), and its reference value goes to
    `_season_f32.bin`. With appearance codes, an input "appearance" [1, D] (reference `_appearance_input_f32.bin`),
    and the training images' codes go to `_appearance_codes_f32.bin` [images][D] with their ids, one per line, in
    `_appearance_ids.txt`."""
    import sys

    import onnx

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "landscape"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qwen_image"))
    from export_depth import constants_to_initializers
    from export_vae import add_initializer_value_info, remove_dead_nodes, require_concrete_shapes

    textured = generator.texture is not None
    seasonal, size = generator.generator.season, generator.generator.appearance
    model = Generator(widths=generator.generator.widths, extra=generator.texture.channels if textured else 0,
                      blur_depth=not textured, film=seasonal, norm=generator.generator.norm,
                      appearance=size).eval()
    model.load_state_dict({k: v.cpu() for k, v in generator.generator.state_dict().items()})
    if textured:
        # The texture's tables, [classes][levels][table][features], for the Klartraum lookup.
        texture = generator.texture
        tables = torch.stack([grid.embeddings.detach().cpu() for grid in texture.grids])
        tables.numpy().astype(np.float32).tofile(path.with_name(path.stem + "_texture_f32.bin"))
        # The shape for Klartraum (neural::HashTextureShape::fromFile()) and a reference for its parity test: random
        # coordinates over every class, with and without a grid, and PyTorch's features of them.
        grid = texture.grids[0]
        lines = [f"width {width}", f"height {height}", f"levels {grid.levels}", f"features {grid.features}",
                 f"table {grid.table}", "resolutions " + " ".join(str(r) for r in grid.resolutions),
                 "classes " + " ".join(str(c) for c in texture.classes)]
        path.with_name(path.stem + "_texture.txt").write_text("\n".join(lines) + "\n")
        reference = torch.Generator().manual_seed(1)
        coordinates = torch.rand(1, 3, height, width, generator=reference)
        classes = torch.randint(0, CLASSES, (1, height, width), generator=reference)
        device = grid.embeddings.device
        with torch.no_grad():
            features = texture(coordinates.to(device), classes.to(device)).cpu()
        coordinates.numpy().tofile(path.with_name(path.stem + "_texture_coordinates_f32.bin"))
        classes.float().numpy().tofile(path.with_name(path.stem + "_texture_classes_f32.bin"))
        features.numpy().tofile(path.with_name(path.stem + "_texture_features_f32.bin"))
    channels = INPUT_CHANNELS + (generator.texture.channels if textured else 0)
    example = torch.rand(1, channels, height, width, generator=torch.Generator().manual_seed(0))
    season = season_code(torch.tensor([3.3]))
    code = (generator.codes.weight[:1].detach().cpu() if generator.codes is not None
            else torch.zeros(1, size)) + 0.05
    arguments, names = [example], ["input"]
    if seasonal:
        arguments.append(season)
        names.append("season")
    if size:
        arguments.append(code)
        names.append("appearance")

    class Inputs(nn.Module):
        """The generator with its graph inputs as positional arguments: input, then season and appearance where it
        has them."""

        def __init__(self, generator: nn.Module):
            super().__init__()
            self.generator = generator

        def forward(self, x: torch.Tensor, *rest: torch.Tensor) -> torch.Tensor:
            rest = list(rest)
            return self.generator(x, rest.pop(0) if seasonal else None, rest.pop(0) if size else None)

    wrapped = Inputs(model).eval()
    with torch.no_grad():
        torch.onnx.export(wrapped, tuple(arguments), str(path), input_names=names, output_names=["image"],
                          opset_version=17, dynamo=False, do_constant_folding=True)
        expected = wrapped(*arguments)
    graph = onnx.shape_inference.infer_shapes(onnx.load(path), strict_mode=True)
    constants_to_initializers(graph)
    remove_dead_nodes(graph)
    add_initializer_value_info(graph)
    require_concrete_shapes(graph)
    onnx.save(graph, str(path))
    example.numpy().tofile(path.with_name(path.stem + "_input_f32.bin"))
    if seasonal:
        season.numpy().tofile(path.with_name(path.stem + "_season_f32.bin"))
    if size:
        code.numpy().tofile(path.with_name(path.stem + "_appearance_input_f32.bin"))
        generator.codes.weight.detach().cpu().numpy().astype(np.float32).tofile(
            path.with_name(path.stem + "_appearance_codes_f32.bin"))
        path.with_name(path.stem + "_appearance_ids.txt").write_text("\n".join(appearance_ids) + "\n")
    expected.numpy().tofile(path.with_name(path.stem + "_output_f32.bin"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--l1", type=float, default=1.0)
    parser.add_argument("--perceptual", type=float, default=1.0)
    parser.add_argument("--colour", type=float, default=0.0,
                        help="weight of L1 at 1/8 resolution: colour mood and large-scale brightness only")
    parser.add_argument("--feature-matching", type=float, default=2.0)
    parser.add_argument("--adversarial", type=float, default=0.1)
    parser.add_argument("--adversarial-start", type=int, default=2000, help="steps of reconstruction losses only")
    parser.add_argument("--r1", type=float, default=1.0)
    parser.add_argument("--reload", type=int, default=2000)
    parser.add_argument("--every", type=int, default=2000)
    parser.add_argument("--min-alignment", type=float, default=0.0,
                        help="leave out photos following their depth worse than this (check_photo_alignment.py)")
    parser.add_argument("--texture", action="store_true",
                        help="learn a hash-grid surface texture over the coordinate G-buffers (add_surface_coordinates.py)")
    parser.add_argument("--init-from", type=Path,
                        help="start the generator from this checkpoint's (new input channels start at zero)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", help="torch device (default: mps where available, else cpu)")
    parser.add_argument("--appearance", type=int, default=0,
                        help="size of a learnable appearance code per training image (NeRF-W style, by FiLM); 0: none")
    parser.add_argument("--appearance-decay", type=float, default=1e-2, help="L2 penalty on the appearance codes")
    parser.add_argument("--levels", type=int, default=5,
                        help="U-Net levels of the generator (fewer: a smaller receptive field, a more local generator)")
    parser.add_argument("--norm", choices=("group", "channel"), default="group",
                        help="the generator's normalization; channel: per pixel, no statistics over the whole image")
    parser.add_argument("--season-film", action="store_true",
                        help="condition generator and discriminator on the sample's season (seasons.json, FiLM)")
    parser.add_argument("--reset-discriminator", action="store_true", help="with --resume: start a new discriminator")
    args = parser.parse_args()
    global MIN_ALIGNMENT, TEXTURE
    MIN_ALIGNMENT = args.min_alignment
    TEXTURE = args.texture
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    rng = np.random.default_rng(1)
    torch.manual_seed(1)

    # With appearance codes, one row per training image; the rows' order is kept in the checkpoint.
    appearance_ids: list[str] = []
    if args.appearance:
        appearance_ids = list(load_split(args.data / "train")[0])
        if args.resume and (args.out / "checkpoint.pt").exists():
            appearance_ids = torch.load(args.out / "checkpoint.pt", map_location="cpu")["appearance_ids"]
    rows = {sample: k for k, sample in enumerate(appearance_ids)}
    generator = TexturedModel(args.texture, args.season_film, len(appearance_ids), args.appearance, args.levels, args.norm).to(device)
    discriminator = MultiScaleDiscriminator(season=args.season_film).to(device)
    vgg = Vgg().to(device)
    if args.init_from:
        start = torch.load(args.init_from, map_location=device)["ema"]
        # A textured checkpoint's texture carries over to a textured model.
        if generator.texture is not None and any(k.startswith("texture.") for k in start):
            generator.texture.load_state_dict({k.removeprefix("texture."): v for k, v in start.items()
                                               if k.startswith("texture.")})
        own = generator.generator.state_dict()
        for key, value in start.items():
            if key.startswith("texture."):
                continue
            key = key.removeprefix("generator.")
            if key in own and own[key].shape != value.shape:
                # The first convolution's new input channels (the texture features) start at zero.
                padded = torch.zeros_like(own[key])
                padded[:, :value.shape[1]] = value
                value = padded
            own[key] = value
        generator.generator.load_state_dict(own)
    # The texture's table entries learn faster (Instant NGP uses 1e-2 for the encoding).
    texture_params = list(generator.texture.parameters()) if generator.texture is not None else []
    network_params = list(generator.generator.parameters())
    groups = [{"params": network_params, "lr": 1e-4}]
    if texture_params:
        groups.append({"params": texture_params, "lr": 1e-2, "betas": (0.9, 0.99)})
    if generator.codes is not None:
        groups.append({"params": list(generator.codes.parameters()), "lr": 5e-3, "betas": (0.9, 0.99)})
    optimise_g = torch.optim.Adam(groups, lr=1e-4, betas=(0.5, 0.999))
    optimise_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    ema = TexturedModel(args.texture, args.season_film, len(appearance_ids), args.appearance, args.levels, args.norm).to(device)
    ema.load_state_dict(generator.state_dict())
    ema.requires_grad_(False)
    step = 0
    checkpoint = args.out / "checkpoint.pt"
    if args.resume and checkpoint.exists():
        state = torch.load(checkpoint, map_location=device)
        generator.load_state_dict(state["generator"])
        ema.load_state_dict(state["ema"])
        optimise_g.load_state_dict(state["optimise_g"])
        if not args.reset_discriminator:
            discriminator.load_state_dict(state["discriminator"])
            optimise_d.load_state_dict(state["optimise_d"])
        step = state["step"]
    # The best validation score (L1 + VGG of the EMA generator) so far, of best_checkpoint.pt.
    best = state.get("best", float("inf")) if args.resume and checkpoint.exists() else float("inf")
    (args.out / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))

    ids, inputs, photos, coords, seasons = [], None, None, None, None
    validation = load_split(args.data / "validation")
    log = open(args.out / "log.txt", "a")
    started = time.time()
    while step < args.steps:
        if inputs is None or step % args.reload == 0:
            ids, inputs, photos, coords, seasons = load_split(args.data / "train")
            print(f"step {step}: {len(ids)} training pairs, {len(validation[0])} validation pairs", flush=True)
            if len(ids) < 4:
                time.sleep(120)
                inputs = None
                continue
        x, y, uvid, cls, season, picked = crops(inputs, photos, args.batch, args.crop, rng, coords, seasons)
        season = season.to(device)
        # The appearance codes' rows of the crops' images (images added after the start have none and are skipped).
        row = torch.tensor([rows.get(ids[int(k)], 0) for k in picked], device=device) if rows else None
        x, y, cls = x.to(device), y.to(device), cls.to(device)
        uvid = uvid.to(device) if uvid is not None else None
        adversarial = step >= args.adversarial_start

        fake = generator(x, uvid, cls, season, row)
        if adversarial:
            real_pair, fake_pair = torch.cat([x, y], 1), torch.cat([x, fake.detach()], 1)
            loss_d = 0.0
            for (_, real_logits), (_, fake_logits) in zip(discriminator(augment_pair(real_pair), season),
                                                          discriminator(augment_pair(fake_pair), season)):
                loss_d = loss_d + F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()
            if step % 16 == 0:
                loss_d = loss_d + args.r1 * 8 * r1_penalty(discriminator, augment_pair(real_pair), season)
            optimise_d.zero_grad(set_to_none=True)
            loss_d.backward()
            optimise_d.step()

        loss_l1 = F.l1_loss(fake, y)
        loss_vgg = vgg.distance(fake, y)
        loss_g = args.l1 * loss_l1 + args.perceptual * loss_vgg
        if args.colour > 0:
            loss_g = loss_g + args.colour * F.l1_loss(F.avg_pool2d(fake, 8), F.avg_pool2d(y, 8))
        if generator.codes is not None:
            loss_g = loss_g + args.appearance_decay * generator.codes(row).pow(2).sum(1).mean()
        loss_adv = loss_fm = torch.zeros((), device=device)
        if not adversarial:
            loss_d = torch.zeros((), device=device)
        if adversarial:
            fake_pair, real_pair = torch.cat([x, fake], 1), torch.cat([x, y], 1)
            fake_results, real_results = discriminator(fake_pair, season), discriminator(real_pair, season)
            for (fake_features, fake_logits), (real_features, _) in zip(fake_results, real_results):
                loss_adv = loss_adv + F.softplus(-fake_logits).mean()
                for f, r in zip(fake_features, real_features):
                    loss_fm = loss_fm + F.l1_loss(f, r.detach()) / len(fake_features)
            loss_g = loss_g + args.adversarial * loss_adv + args.feature_matching * loss_fm
        optimise_g.zero_grad(set_to_none=True)
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(network_params, 1.0)
        optimise_g.step()
        with torch.no_grad():
            for e, g in zip(ema.parameters(), generator.parameters()):
                e.lerp_(g, 1e-3 if step > 1000 else 1.0)
        step += 1

        if step % 100 == 0:
            line = (f"step {step} l1 {loss_l1.item():.4f} vgg {loss_vgg.item():.4f} adv {loss_adv.item():.3f} "
                    f"d {float(loss_d):.3f} "
                    f"fm {loss_fm.item():.4f} ({(time.time() - started) / 100:.2f} s/step)")
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
            started = time.time()
        if step % args.every == 0 or step == args.steps:
            if not validation[0] or step % args.reload == 0:
                validation = load_split(args.data / "validation")
            # Until held-out pairs exist, the grid shows training pairs (marked in the file name).
            shown = validation if validation[0] else (ids[:6], inputs[:6], photos[:6],
                                                      coords[:6] if coords is not None else None, seasons[:6])
            shown_rows = None if validation[0] or not rows else torch.tensor([rows[i] for i in ids[:6]])
            score = evaluate(ema, vgg, shown, args.out, step, device, log, "validation" if validation[0] else "train",
                             shown_rows)
            if validation[0] and score < best:
                best = score
                torch.save({"ema": ema.state_dict(), "step": step, "score": score, "appearance_ids": appearance_ids},
                           args.out / "best_checkpoint.pt")
                export(ema, args.out / "photo_enhancer_best.onnx", inputs.shape[2], inputs.shape[3], appearance_ids)
                line = f"best validation so far at step {step}: {score:.4f}"
                print(line, flush=True)
                log.write(line + "\n")
            torch.save({"generator": generator.state_dict(), "discriminator": discriminator.state_dict(),
                        "ema": ema.state_dict(), "optimise_g": optimise_g.state_dict(),
                        "optimise_d": optimise_d.state_dict(), "step": step, "best": best,
                        "appearance_ids": appearance_ids}, checkpoint)
            export(ema, args.out / "photo_enhancer.onnx", inputs.shape[2], inputs.shape[3], appearance_ids)


def fit_code(model, vgg, x, uvid, classes, season, target, steps: int = 40) -> torch.Tensor:
    """The appearance code under which the frozen model renders `target` best (L1 + VGG), as for a new image of NeRF
    in the Wild: from the training codes' mean, `steps` Adam steps on the code alone."""
    code = model.codes.weight.mean(dim=0, keepdim=True).detach().clone().requires_grad_(True)
    optimiser = torch.optim.Adam([code], lr=5e-2)
    with torch.enable_grad():
        for _ in range(steps):
            z = model(x, uvid, classes, season, code)
            loss = F.l1_loss(z, target) + vgg.distance(z, target)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
    return code.detach()


@torch.no_grad()
def evaluate(model, vgg, validation, out: Path, step: int, device, log, split: str = "validation",
             rows: torch.Tensor | None = None) -> float:
    """Logs the mean L1 and VGG distance of the pairs and writes a grid of the first six; returns L1 + VGG. With
    appearance codes, training pairs use their own (`rows`); held-out pairs get one fitted (fit_code()), and the
    scores with the training codes' mean are logged as well."""
    ids, inputs, photos, coords, seasons = validation
    if not ids:
        return float("inf")
    l1 = perceptual = mean_l1 = mean_perceptual = 0.0
    shown_x, shown_y, shown_z = [], [], []
    for k in range(len(ids)):
        c = coords[k:k + 1] if coords is not None else None
        x = network_input(inputs[k:k + 1], c).to(device)
        y = photos[k:k + 1].float().to(device) / 255.0
        uvid = c[:, :3].to(device) if c is not None else None
        classes = inputs[k:k + 1, 4].long().to(device)
        season = seasons[k:k + 1].to(device)
        code = None
        if model.codes is not None:
            if rows is not None:
                code = model.codes(rows[k:k + 1].to(device))
            else:
                mean = model.codes.weight.mean(dim=0, keepdim=True)
                z = model(x, uvid, classes, season, mean)
                mean_l1 += F.l1_loss(z, y).item()
                mean_perceptual += vgg.distance(z, y).item()
                code = fit_code(model, vgg, x, uvid, classes, season, y)
        z = model(x, uvid, classes, season, code)
        l1 += F.l1_loss(z, y).item()
        perceptual += vgg.distance(z, y).item()
        if k < 6:
            shown_x.append(F.interpolate(x, scale_factor=0.5, mode="area")[0])
            shown_y.append(F.interpolate(y, scale_factor=0.5, mode="area")[0])
            shown_z.append(F.interpolate(z, scale_factor=0.5, mode="area")[0])
    n = len(ids)
    line = f"{split} step {step}: L1 {l1 / n:.4f}, VGG {perceptual / n:.4f} over {n} pairs"
    if model.codes is not None and rows is None:
        line += f" (fitted codes; mean code: L1 {mean_l1 / n:.4f}, VGG {mean_perceptual / n:.4f})"
    print(line, flush=True)
    log.write(line + "\n")
    log.flush()
    write_grid(out / f"{split}_{step:06d}.png", shown_x, shown_y, shown_z)
    return (l1 + perceptual) / n


if __name__ == "__main__":
    main()
