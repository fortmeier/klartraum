# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Draws the compute graphs of the tech demo's chapters as 1920 x 1080 screens.

Each graph becomes at most twelve blocks on one line, in the order the GPU runs
them: a block is a stage (an ONNX network, the splatting, a render pass, ...)
with its compute dispatches and raster draws (counted with api_dump) and its
measured GPU time. The machine the times were measured on is named at the bottom (detected, or --hardware).
Below the line, a bar splits the graph's total GPU time into
the blocks and the time between them (the barriers the graph puts between its
elements, and elements that record nothing), so the parts add up to the total.
The upper right corner stays free for a picture in picture of the chapter.

    uv run --with pillow scripts/techdemo000/draw_graphs.py [NAME ...]
        [--dir build/TestingOutput/techdemo000_graphs] [--fonts build/fonts]
    uv run --with pillow scripts/techdemo000/draw_graphs.py --chain DIR

Inputs: <name>.json with GPU times (techdemo000_graphs, or techdemo000_example
--dump-graphs for the SD1.5 chain) and <name>.commands.json
(graph_dispatches.py).
"""

import argparse
import collections
import json
import platform
import re
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1920, 1080
SS = 2  # supersampling
LEFT, RIGHT = 70, WIDTH - 70
LINE_Y = 540  # centre of the blocks

BACKGROUND = (0, 0, 0)
TEXT = (240, 240, 240)
DIM = (150, 150, 150)
FAINT = (95, 95, 100)
BOX_FILL = (18, 18, 20)
BOX_LINE = (200, 200, 200)
EDGE = (130, 130, 138)
SEMAPHORE = (245, 166, 35)
BAR_WORK = (200, 200, 205)
BAR_SYNC = (110, 110, 116)
BAR_OUTSIDE = (55, 55, 60)

TITLES = {
    "gaussian_splatting": ("3D Gaussian splatting", "the scene rasterized as depth-sorted Gaussians"),
    "dynamic_lighting": ("Dynamic lighting", "the Gaussians relit by three lights, then splatted"),
    "neural_animation": ("Neural character animation", "the controller walks the skeleton, drawn as Gaussians"),
}

# Block names, by element name; elements of a collapsed group take the group's name.
BLOCKS = {
    # raster splatting
    "GaussianDist": "cull & depth keys",
    "RasterProject": "project",
    "RasterRadixSort": "radix sort",
    "RasterRenderPass": "rasterize",
    "RasterDepthResolve": "depth",
    # compute splatting
    "GaussianProjection": "project",
    "GaussianBinningCount": "binning",
    "GaussianBinningPrefixSum": "binning",
    "GaussianBinningScatter": "binning",
    "ExtractSortKeys": "radix sort",
    "RadixSort": "radix sort",
    "GatherSorted": "radix sort",
    "GaussianComputeBounds": "splat tiles",
    "GaussianSplatting": "splat tiles",
    # chapters 3 and 4
    "GaussianLighting": "relight",
    "CharacterAssemble": "controller input",
    "CharacterIntegrate": "integrate pose",
    "SkeletonGaussians": "skeleton Gaussians",
    "GaussianMerge": "merge with scene",
    # SD1.5
    "CloseUpWindow": "close-up window",
    "PosePass": "pose map & masks",
    "MaskPass": "pose map & masks",
    "BodyMaskPass": "pose map & masks",
    "SceneToTensor": "image → tensor",
    "PoseToTensor": "image → tensor",
    "LatentMask": "latent mask",
    "LcmInit": "LCM start",
    "CloseUpPaste": "paste & composite",
    "CloseUpComposite": "paste & composite",
}
# Groups drawn as one block, by group name.
GROUP_BLOCKS = {"Gaussian splatting": "Gaussian splatting", "VAE encoder": "VAE encoder",
                "VAE decoder": "VAE decoder", "ControlNet": "ControlNet", "UNet": "UNet",
                "Mixture-of-experts network": "MoE network"}

# Inputs shown above the block that reads them first.
INPUTS = {"Pos3D": "scene Gaussians", "Camera": "camera", "camera": "camera", "LightingFrame": "lights",
          "SplatMaterials": "materials", "SkeletonParams": "skeleton params", "joints": "joints",
          "prompt embedding": "prompt", "LcmNoise": "noise", "frame": "frame", "character mask": "mask"}

INPUT_TYPES = {"BufferElementSinglePath", "UniformBufferObject", "HostValues", "ImageViewSrc", "TensorElement",
               "TensorElementSinglePath"}

TRANSFER = {"vkCmdFillBuffer", "vkCmdClearColorImage", "vkCmdCopyBuffer", "vkCmdCopyBufferToImage",
            "vkCmdUpdateBuffer", "vkCmdCopyImage", "vkCmdCopyImageToBuffer", "vkCmdBlitImage"}
OP_NAME = re.compile(r"^([A-Za-z]+)_(\d+)_(.*)$")


def work(commands):
    dispatches = commands.get("vkCmdDispatch", 0) + commands.get("vkCmdDispatchIndirect", 0)
    draws = sum(v for k, v in commands.items() if k.startswith("vkCmdDraw"))
    transfers = sum(v for k, v in commands.items() if k in TRANSFER)
    return dispatches, draws, transfers


def run(command):
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def hardware():
    """What the measurements ran on: the GPU, CPU cores, memory, OS and the Vulkan driver the loader picks first."""
    parts = []
    summary = run(["vulkaninfo", "--summary"])
    device = re.search(r"deviceName\s*=\s*(.+)", summary)
    if platform.system() == "Darwin":
        chip = run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
        displays = run(["system_profiler", "SPDisplaysDataType"])
        gpu_cores = re.search(r"Total Number of Cores:\s*(\d+)", displays)
        parts.append(chip + (f", {gpu_cores.group(1)}-core GPU" if gpu_cores else ""))
        performance = run(["sysctl", "-n", "hw.perflevel0.physicalcpu"]).strip()
        efficiency = run(["sysctl", "-n", "hw.perflevel1.physicalcpu"]).strip()
        if performance and efficiency:
            parts.append(f"{performance} + {efficiency} CPU cores")
        memory = run(["sysctl", "-n", "hw.memsize"]).strip()
        if memory:
            parts.append(f"{int(memory) // 2 ** 30} GB unified memory")
        parts.append(f"macOS {platform.mac_ver()[0]}")
    elif device:
        parts.append(device.group(1).strip())
    api = re.search(r"apiVersion\s*=\s*(\d+\.\d+)", summary)
    driver = re.search(r"driverName\s*=\s*(.+)", summary)
    driver_info = re.search(r"driverInfo\s*=\s*(.+)", summary)
    if api:
        parts.append(f"Vulkan {api.group(1)}" + (f" via {driver.group(1).strip()}" if driver else "") +
                     (f" {driver_info.group(1).strip()}" if driver_info else ""))
    return " · ".join(parts)


def plural(count, word):
    return f"{count:,} {word}" + ("" if count == 1 else "es" if word.endswith(("ch", "sh")) else "s")


def ms_text(ms):
    return f"{ms:.0f} ms" if ms >= 100 else f"{ms:.1f} ms" if ms >= 10 else f"{ms:.2f} ms"


def counts_text(dispatches, draws):
    parts = []
    if dispatches:
        parts.append(f"{dispatches:,} compute")
    if draws:
        parts.append(f"{draws} raster")
    return " · ".join(parts) or "copies only"


class Block:
    def __init__(self, label, order):
        self.label = label
        self.order = order
        self.commands = collections.Counter()
        self.ms = 0.0
        self.total_ms = 0.0
        self.time_text = ""
        self.share_text = ""
        self.inputs = []
        self.link = 0  # which graph of a chain
        self.x = self.w = self.h = 0.0


class GraphBlocks:
    """The blocks of one graph, the edges between them, and the GPU time outside any block."""

    def __init__(self, graph, expand_splatting):
        elements = graph["elements"]
        self.total_ms = sum(e.get("ms", 0.0) for e in elements)
        groups = [dict(g, size=0, type=elements[g["element"]]["type"]) for g in graph["groups"]]
        group_elements = {g["element"]: gi for gi, g in enumerate(groups)}
        for e in elements:
            for g in e["groups"]:
                groups[g]["size"] += 1
        collapsed = {gi for gi, g in enumerate(groups) if g["name"] in GROUP_BLOCKS and not (
            g["name"] == "Gaussian splatting" and expand_splatting)}
        preds, succs = collections.defaultdict(list), collections.defaultdict(list)
        for a, b in graph["edges"]:
            preds[b].append(a)
            succs[a].append(b)

        block_of, blocks = {}, {}
        for e in elements:
            i = e["id"]
            d, dr, _ = work(e["commands"])
            chain = sorted(e["groups"], key=lambda g: -groups[g]["size"])
            if i in group_elements:
                chain = chain + [group_elements[i]]
            if d + dr and not OP_NAME.match(e["name"]):
                # The program around a network (e.g. assembling its input) is not part of it.
                chain = [g for g in chain if groups[g]["type"] != "OnnxNetwork" or group_elements.get(i) == g]
            group = next((g for g in chain if g in collapsed), None)
            if group is not None:
                label = GROUP_BLOCKS[groups[group]["name"]]
                if not (d + dr) and i not in group_elements:
                    block_of[i] = label  # buffers and tensors of a block's group count with it
                    continue
            elif d + dr:
                label = BLOCKS.get(e["name"], e["name"] or e["type"])
            else:
                continue
            if label not in blocks:
                blocks[label] = Block(label, i)
            blocks[label].order = min(blocks[label].order, i)
            block_of[i] = label
        for i, label in list(block_of.items()):
            if label not in blocks:
                blocks[label] = Block(label, i)

        # Fills and clears (of buffers and targets that record nothing else) count with the block that reads them.
        def first_block(i, step):
            frontier, seen = list(step[i]), set()
            while frontier:
                nxt = []
                for j in sorted(frontier):
                    if j in seen:
                        continue
                    seen.add(j)
                    if j in block_of:
                        return block_of[j]
                    nxt.extend(step[j])
                frontier = nxt
            return None

        for e in elements:
            i = e["id"]
            if i not in block_of and sum(work(e["commands"])):
                label = first_block(i, succs) or first_block(i, preds)
                if label:
                    block_of[i] = label

        for e in elements:
            label = block_of.get(e["id"])
            if label is not None:
                blocks[label].commands.update(e["commands"])
                blocks[label].ms += e.get("ms", 0.0)
        self.blocks = sorted(blocks.values(), key=lambda b: b.order)
        self.sync_ms = self.total_ms - sum(b.ms for b in self.blocks)

        # Inputs: named sources, shown at the first block reading them.
        for e in elements:
            name = INPUTS.get(e["name"])
            # Only data the graph does not produce: shared buffers, host values, images of other graphs.
            if name is None or e["type"] not in INPUT_TYPES or (preds[e["id"]] and e["type"] != "ImageViewSrc"):
                continue
            label = first_block(e["id"], succs)
            if label and name not in blocks[label].inputs:
                blocks[label].inputs.append(name)

        # Edges between blocks, through the elements outside any block.
        self.edges = set()
        for e in elements:
            b = block_of.get(e["id"])
            if b is None:
                continue
            stack, seen = list(preds[e["id"]]), set()
            while stack:
                p = stack.pop()
                if p in seen:
                    continue
                seen.add(p)
                a = block_of.get(p)
                if a is None:
                    stack.extend(preds[p])
                elif a != b:
                    self.edges.add((a, b))


def load(directory, name):
    graph = json.loads((directory / f"{name}.json").read_text())
    commands_path = directory / f"{name}.commands.json"
    commands = json.loads(commands_path.read_text()) if commands_path.exists() else [{}] * len(graph["elements"])
    for element, counts in zip(graph["elements"], commands):
        element["commands"] = counts
    return graph


# ---------------------------------------------------------------- drawing


class Fonts:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.cache = {}

    def get(self, family, size, weight=400):
        key = (family, size, weight)
        if key not in self.cache:
            f = ImageFont.truetype(str(self.directory / f"{family}.ttf"), round(size * SS))
            try:
                f.set_variation_by_axes([weight])
            except (OSError, AttributeError):
                pass
            self.cache[key] = f
        return self.cache[key]

    def width(self, text, size, family="JetBrainsMono"):
        return self.get(family, size).getlength(text) / SS


class Canvas:
    def __init__(self, fonts):
        self.fonts = fonts
        self.img = Image.new("RGBA", (WIDTH * SS, HEIGHT * SS), BACKGROUND + (255,))
        self.d = ImageDraw.Draw(self.img)

    def rect(self, x0, y0, x1, y1, fill=None, outline=None, width=1, radius=0):
        box = [x0 * SS, y0 * SS, x1 * SS, y1 * SS]
        if radius:
            self.d.rounded_rectangle(box, radius=radius * SS, fill=fill, outline=outline, width=round(width * SS))
        else:
            self.d.rectangle(box, fill=fill, outline=outline, width=round(width * SS))

    def text(self, x, y, text, size, fill, family="JetBrainsMono", weight=400, anchor="la"):
        self.d.text((x * SS, y * SS), text, font=self.fonts.get(family, size, weight), fill=fill, anchor=anchor)

    def line(self, points, fill, width=1.0):
        self.d.line([(x * SS, y * SS) for x, y in points], fill=fill, width=max(1, round(width * SS)), joint="curve")

    def triangle(self, tip, base_a, base_b, fill):
        self.d.polygon([(x * SS, y * SS) for x, y in (tip, base_a, base_b)], fill=fill)

    def arrow(self, x0, x1, y, fill, width=1.6):
        self.line([(x0, y), (x1 - 2, y)], fill, width)
        self.triangle((x1, y), (x1 - 8, y - 4.5), (x1 - 8, y + 4.5), fill)

    def arc(self, x0, x1, y, depth, fill, width=1.0):
        """A curve from (x0, y) to (x1, y) bulging by `depth` (positive: down)."""
        points = []
        for i in range(41):
            t = i / 40
            u = 1 - t
            points.append((u * u * u * x0 + 3 * u * u * t * x0 + 3 * u * t * t * x1 + t * t * t * x1,
                           y + (3 * u * u * t + 3 * u * t * t) * depth))
        self.line(points, fill, width)

    def save(self, path):
        self.img.convert("RGB").resize((WIDTH, HEIGHT), Image.LANCZOS).save(path)


def wrap(text, width):
    """`text` in lines of at most `width` characters, broken at spaces."""
    lines, line = [], ""
    for word in text.split(" "):
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}" if line else word
    return lines + [line]


def size_blocks(blocks, fonts, scale):
    label, small, time = round(19 * scale), round(14 * scale), round(16 * scale)
    pad = round(11 * scale)
    lines = max(len(wrap(b.label, 11)) for b in blocks)
    for b in blocks:
        d, dr, _ = work(b.commands)
        name = wrap(b.label, 11)
        name += [""] * (lines - len(name))
        counts = [f"{d:,} compute" if d else "", f"{dr} raster" if dr else ""]
        b.lines = ([(text, label, TEXT) for text in name] + [(text, small, DIM) for text in counts] +
                   [(b.time_text, time, TEXT), (b.share_text, small, DIM)])
        b.w = max(fonts.width(text, size) for text, size, _ in b.lines) + 2 * pad
        b.h = sum(size + 6 for _, size, _ in b.lines) + 2 * pad
        b.pad = pad


def draw_screen(blocks, edges, segments, total_ms, sync_ms, wall_ms, heading, sub, stats, out, fonts, machine=""):
    """Blocks in GPU order; segments of a chain: (label, first block index, last block index, repeat)."""
    c = Canvas(fonts)
    segment_gap = 34 * max(0, len(segments) - 1)
    for scale in (1.0, 0.92, 0.85, 0.78, 0.72, 0.66, 0.6, 0.55):
        size_blocks(blocks, fonts, scale)
        if sum(b.w for b in blocks) + 28 * (len(blocks) - 1) + segment_gap <= RIGHT - LEFT:
            break
    gap = min(90, (RIGHT - LEFT - sum(b.w for b in blocks) - segment_gap) / max(1, len(blocks) - 1))
    starts = {s[1] for s in segments[1:]}
    x = LEFT + (RIGHT - LEFT - sum(b.w for b in blocks) - gap * (len(blocks) - 1) - segment_gap) / 2
    for i, b in enumerate(blocks):
        if i in starts:
            x += 34
        b.x = x
        x += b.w + gap

    index = {b.label: i for i, b in enumerate(blocks)}
    height = max(b.h for b in blocks)
    top, bottom = LINE_Y - height / 2, LINE_Y + height / 2
    in_loop = {i for s in segments if s[3] > 1 for i in range(s[1], s[2] + 1)}

    # Data that skips blocks: arcs below the line.
    for a, b in sorted(edges, key=lambda e: index[e[1]] - index[e[0]]):
        ia, ib = index[a], index[b]
        if ib - ia <= 1:
            continue
        xa, xb = blocks[ia].x + blocks[ia].w * 0.6, blocks[ib].x + blocks[ib].w * 0.4
        c.arc(xa, xb, bottom, min(100, 24 + 12 * (ib - ia)), EDGE + (255,), 1.1)

    # The order of the GPU work: arrows between neighbours, amber where the next command buffer of a chain starts
    # (it waits for a semaphore).
    for i in range(len(blocks) - 1):
        semaphore = blocks[i + 1].link != blocks[i].link
        colour = SEMAPHORE if semaphore else EDGE
        c.arrow(blocks[i].x + blocks[i].w + 4, blocks[i + 1].x - 4, LINE_Y, colour + (255,), 2.2 if semaphore else 1.6)

    # A repeated segment: the way back.
    for label, first, last, repeat in segments:
        if repeat > 1:
            xa, xb = blocks[first].x + 24, blocks[last].x + blocks[last].w - 24
            y = top - 26
            c.line([(xb, top - 3), (xb, y), (xa, y), (xa, top - 9)], SEMAPHORE + (255,), 2.2)
            c.triangle((xa, top - 2), (xa - 4.5, top - 10), (xa + 4.5, top - 10), SEMAPHORE + (255,))
            c.text((xa + xb) / 2, y - 8, f"{repeat} ×, once per LCM step", 15, SEMAPHORE, anchor="md")

    # Blocks, and their inputs above them.
    for i, b in enumerate(blocks):
        y0 = LINE_Y - b.h / 2
        c.rect(b.x, y0, b.x + b.w, y0 + b.h, fill=BOX_FILL + (255,), outline=BOX_LINE + (255,), width=1.2, radius=7)
        y = y0 + b.pad - 2
        for text, size, colour in b.lines:
            c.text(b.x + b.pad, y, text, size, colour)
            y += size + 6
        if b.inputs:
            lift = 54 if i in in_loop else 0
            cx = b.x + b.w / 2
            c.line([(cx, y0 - 14 - lift), (cx, y0 - 2)], FAINT + (255,), 1.0)
            for k, name in enumerate(reversed(b.inputs)):
                c.text(cx, y0 - 18 - lift - 19 * k, name, 13, DIM, anchor="md")

    # Segments of a chain, below the arcs.
    seg_y = bottom + 124
    for label, first, last, repeat in segments:
        xa, xb = blocks[first].x, blocks[last].x + blocks[last].w
        c.line([(xa, seg_y - 7), (xa, seg_y), (xb, seg_y), (xb, seg_y - 7)], FAINT + (255,), 1.2)
        c.text((xa + xb) / 2, seg_y + 9, label, 14, DIM, anchor="ma")

    # The time from submission to finish, split into the blocks, the time between them, and what the timestamps
    # of the elements do not cover (between command buffers, outside the elements).
    bar_y = HEIGHT - 150
    uncovered = max(0.0, wall_ms - total_ms)
    whole = max(wall_ms, total_ms)
    headline = f"per frame, submit to finish: {ms_text(whole)}"
    c.text(LEFT, bar_y - 40, headline, 20, TEXT, "Unbounded", 500)
    note = f"= blocks (GPU timestamps) + {ms_text(sync_ms)} barriers"
    if uncovered > 0:
        note += f" + {ms_text(uncovered)} outside the timestamps (sync, driver)"
    c.text(LEFT + fonts.width(headline, 20, "Unbounded") + 18, bar_y - 35, note, 14, DIM)
    parts = [(b.label, b.total_ms, BAR_WORK) for b in blocks] + [("barriers", sync_ms, BAR_SYNC)]
    if uncovered > 0:
        parts.append(("outside the timestamps", uncovered, BAR_OUTSIDE))
    scale = (RIGHT - LEFT) / max(whole, 1e-9)
    x = LEFT
    label_end = LEFT - 10
    for label, ms, colour in parts:
        w = ms * scale
        if w >= 0.5:
            c.rect(x, bar_y, x + max(0.5, w - 2), bar_y + 24, fill=colour + (255,), radius=3 if w > 10 else 0)
        text = f"{label} {ms_text(ms)}"
        if w > fonts.width(text, 13) + 8 and x >= label_end:  # only labels that fit under their segment
            c.text(x + 3, bar_y + 32, text, 13, DIM)
            label_end = x + fonts.width(text, 13) + 12
        x += w

    if machine:
        c.text(LEFT, HEIGHT - 52, f"measured on {machine}", 15, DIM)

    c.text(70, 58, heading, 40, TEXT, "Unbounded", 500)
    c.text(72, 118, sub, 20, DIM)
    c.text(72, 154, stats, 18, TEXT)
    c.save(out)


def chapter_screen(directory, name, out, fonts, machine=""):
    graph = load(directory, name)
    blocks = GraphBlocks(graph, expand_splatting=True)
    total = max(blocks.total_ms, 1e-9)
    wall_path = directory / f"{name}.wall.json"
    wall = json.loads(wall_path.read_text())["wall_ms"] if wall_path.exists() else total
    whole = max(wall, total)
    for b in blocks.blocks:
        b.total_ms = b.ms
        b.time_text = ms_text(b.ms)
        b.share_text = f"{100 * b.ms / whole:.0f} %"
    count = collections.Counter()
    for e in graph["elements"]:
        count.update(e["commands"])
    d, dr, _ = work(count)
    heading, sub = TITLES[name]
    stats = "   ·   ".join([plural(len(blocks.blocks), "block"), f"{d:,} compute dispatches",
                            plural(dr, "raster draw"), "1 command buffer, 1 submit"])
    draw_screen(blocks.blocks, blocks.edges, [], total, blocks.sync_ms, wall, heading, sub, stats, out, fonts, machine)
    return len(blocks.blocks), whole


def chain_screen(directory, out, fonts, machine=""):
    """The SD1.5 chain: prepare, the denoiser and an LCM step once per step, decode."""
    chain = json.loads((directory / "chain.json").read_text())
    step_names = [name for name in chain["compiled"] if name.startswith("lcm_step")]
    repeat = len(step_names)
    prepare = GraphBlocks(load(directory, "prepare"), False)
    denoise = GraphBlocks(load(directory, "denoise"), False)
    steps = [GraphBlocks(load(directory, name), False) for name in step_names]
    decode = GraphBlocks(load(directory, "decode"), False)
    # The steps differ only in their constants: one block with their mean time.
    step = steps[0]
    step.blocks[0].label = "LCM step"
    step.blocks[0].ms = sum(g.blocks[0].ms for g in steps) / repeat
    step.total_ms = sum(g.total_ms for g in steps) / repeat
    step.sync_ms = step.total_ms - step.blocks[0].ms

    blocks, edges = [], set()
    for link, (part, times) in enumerate([(prepare, 1), (denoise, repeat), (step, repeat), (decode, 1)]):
        for b in part.blocks:
            b.link = link
            b.total_ms = b.ms * times
            b.time_text = ms_text(b.ms) + (f" ×{times}" if times > 1 else "")
            blocks.append(b)
        edges.update(part.edges)
    total = max(prepare.total_ms + repeat * (denoise.total_ms + step.total_ms) + decode.total_ms, 1e-9)
    sync = prepare.sync_ms + repeat * (denoise.sync_ms + step.sync_ms) + decode.sync_ms
    wall = chain.get("wall_ms", total)
    whole = max(wall, total)
    for b in blocks:
        b.share_text = f"{100 * b.total_ms / whole:.0f} %"
    n_prepare, n_loop = len(prepare.blocks), len(denoise.blocks) + 1
    loop_ms = denoise.total_ms + step.total_ms
    segments = [(f"prepare · {ms_text(prepare.total_ms)}", 0, n_prepare - 1, 1),
                (f"denoise + LCM step · {repeat} × {ms_text(loop_ms)}", n_prepare, n_prepare + n_loop - 1, repeat),
                (f"decode · {ms_text(decode.total_ms)}", n_prepare + n_loop, len(blocks) - 1, 1)]
    count = collections.Counter()
    for link in chain["links"]:
        for e in load(directory, link)["elements"]:
            count.update(e["commands"])
    d, dr, _ = work(count)
    stats = "   ·   ".join([plural(len(blocks), "block"), f"{d:,} compute dispatches", plural(dr, "raster draw"),
                            f"{len(chain['links'])} command buffers, 1 submit"])
    draw_screen(blocks, edges, segments, total, sync, wall, "SD1.5 inpainting on the GPU",
                "close-up, VAE encoder, ControlNet + UNet with LCM, VAE decoder: one chain of graphs", stats, out,
                fonts, machine)
    return len(blocks), whole


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*")
    parser.add_argument("--dir", default="build/TestingOutput/techdemo000_graphs")
    parser.add_argument("--fonts", default="build/fonts")
    parser.add_argument("--chain", help="draw the SD1.5 chain written by techdemo000_example --dump-graphs")
    parser.add_argument("--hardware", help="the machine the graphs were measured on (default: this one)")
    args = parser.parse_args()
    fonts = Fonts(args.fonts)
    machine = args.hardware if args.hardware is not None else hardware()
    if args.chain:
        directory = Path(args.chain)
        out = directory.parent / "sd15_chain.png"
        count, total = chain_screen(directory, out, fonts, machine)
        print(f"sd15_chain: {count} blocks, {total:.1f} ms, {out}")
        return
    directory = Path(args.dir)
    for name in args.names or [n for n in TITLES if (directory / f"{n}.json").exists()]:
        out = directory / f"{name}.png"
        count, total = chapter_screen(directory, name, out, fonts, machine)
        print(f"{name}: {count} blocks, {total:.2f} ms, {out}")


if __name__ == "__main__":
    main()
