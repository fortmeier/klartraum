# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Draws the compute graphs of the tech demo's chapters as 1920 x 1080 screens.

Each graph becomes at most twelve blocks: a block is a stage (an ONNX network,
the splatting, a render pass, ...) with its compute dispatches and raster draws
(counted with api_dump) and its share of the GPU time of all blocks. The shares
come from a run with GPU timestamps, which slow the graph down; the time per
frame in the headline comes from a run without them. The blocks stand in the order
the GPU runs them, one column each from left to right, numbered along an axis
below them; their rows (lanes) group the stages one stream of data passes
through, and arrows between the blocks show where data goes to another stage.
The machine the times were measured on is named at the bottom (detected, or
--hardware). The upper right corner stays free for a picture in picture of the
chapter.

    uv run --with pillow scripts/techdemo000/draw_graphs.py [NAME ...]
        [--dir build/TestingOutput/techdemo000_graphs] [--fonts build/fonts] [--scene data/lantern.spz]
    uv run --with pillow scripts/techdemo000/draw_graphs.py --chain DIR --chain-log LOG

Inputs: <name>.json with GPU times (techdemo000_graphs, or techdemo000_example
--dump-graphs for the SD1.5 chain), <name>.commands.json (graph_dispatches.py)
and the time per frame without timestamps: unprofiled_wall_ms of
<name>.wall.json (techdemo000_graphs), for the SD1.5 chain the "person
generation" line of the log of a render of techdemo000_example.
"""

import argparse
import collections
import json
import platform
import re
import struct
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1920, 1080
SS = 2  # supersampling
LEFT, RIGHT = 70, WIDTH - 70
PIP_LEFT = WIDTH - 70 - 480  # the picture in picture of compose_chapter8.py, at the upper right
GRAPH_TOP, GRAPH_BOTTOM = 196, 994  # the lanes and the order axis

BACKGROUND = (0, 0, 0)
TEXT = (240, 240, 240)
DIM = (150, 150, 150)
FAINT = (95, 95, 100)
BOX_FILL = (18, 18, 20)
BOX_LINE = (200, 200, 200)
EDGE = (130, 130, 138)
SEMAPHORE = (245, 166, 35)

# Heading and subtitle; {gaussians} is the number of Gaussians in the scene.
TITLES = {
    "gaussian_splatting": ("3D Gaussian splatting", "rasterize {gaussians} Gaussians"),
    "dynamic_lighting": ("Dynamic lighting", "additional relighting by three lights"),
    "neural_animation": ("Neural character animation",
                         "MoE network and skeleton visualization by Gaussians as compute graph elements before splatting"),
    "sd15": ("SD1.5 inpainting on the GPU",
             "rasterization of Gaussians, pose image generation, diffusion and composition in simultaneously "
             "submitted graphs"),
}

# The lane of each block (0 at the top, default 0): the main flow of data in lane 0, side branches below it.
LANES = {
    "gaussian_splatting": {"project": 1},
    "dynamic_lighting": {"project": 1},
    "neural_animation": {"project": 1},
    "sd15": {"pose crop window": 1, "pose map & masks": 1, "pose → tensor": 1},
}

# Dependencies that only order two blocks, without data between them: the LCM start depends on the pose tensor so
# that the prepare graph, compiled from the LCM start, contains it.
ORDER_ONLY = {("pose → tensor", "LCM start")}

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
    "CloseUpWindow": "pose crop window",
    "PosePass": "pose map & masks",
    "MaskPass": "pose map & masks",
    "BodyMaskPass": "pose map & masks",
    "SceneToTensor": "scene → tensor",
    "PoseToTensor": "pose → tensor",
    "LatentMask": "pose map & masks",
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


def gaussian_count(spz):
    """The number of Gaussians in an uncompressed .spz file (magic, version, count)."""
    magic, _, count = struct.unpack("<4sII", Path(spz).read_bytes()[:12])
    if magic != b"NGSP":
        raise SystemExit(f"{spz}: not an uncompressed .spz file")
    return count


def share_text(ms, total):
    share = 100 * ms / total
    return f"{share:.0f} %" if share >= 0.5 else "<1 %"


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
        self.share_text = ""
        self.inputs = []
        self.link = 0  # which graph of a chain
        self.lane = 0
        self.x = self.w = self.h = 0.0


class GraphBlocks:
    """The blocks of one graph, the edges between them, and the GPU time outside any block.

    `uses` names the blocks using each element (by key), which links the blocks of the graphs of a chain. A
    buffer's writers depend on it like its readers do, so the graph does not tell them apart; in a chain, the
    order of the graphs does.
    """

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
                label = "LCM step" if e["name"].startswith("LcmStep") else BLOCKS.get(e["name"], e["name"] or e["type"])
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

        def nearest(i):
            """The blocks reached first from element i along its successors."""
            found, stack, seen = set(), list(succs[i]), set()
            while stack:
                j = stack.pop()
                if j in seen:
                    continue
                seen.add(j)
                if j in block_of:
                    found.add(block_of[j])
                else:
                    stack.extend(succs[j])
            return found

        self.uses = collections.defaultdict(set)
        for e in elements:
            self.uses[e["key"]] |= {block_of[e["id"]]} if e["id"] in block_of else nearest(e["id"])


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
    """The lines and size of each block; the blocks of a lane have the same lines, the same height."""
    label, small, time = round(19 * scale), round(14 * scale), round(16 * scale)
    pad = round(11 * scale)
    for b in blocks:
        lane = [o for o in blocks if o.lane == b.lane]
        lines = max(len(wrap(o.label, 11)) for o in lane)
        any_compute = any(work(o.commands)[0] for o in lane)
        any_raster = any(work(o.commands)[1] for o in lane)
        d, dr, _ = work(b.commands)
        name = wrap(b.label, 11)
        name += [""] * (lines - len(name))
        counts = ([f"{d:,} compute" if d else ""] if any_compute else []) + (
            [f"{dr} raster" if dr else ""] if any_raster else [])
        b.lines = ([(text, label, TEXT) for text in name] + [(text, small, DIM) for text in counts] +
                   [(b.share_text, time, TEXT)])
        b.w = max(fonts.width(text, size) for text, size, _ in b.lines) + 2 * pad
        b.h = sum(size + 6 for _, size, _ in b.lines) + 2 * pad
        b.pad = pad


INPUTS_ROW = 30  # space above a lane for the inputs of its blocks
CORRIDOR = 26  # space below a lane, for the arrows between lanes


def lane_heights(blocks, lanes):
    """The space above each lane, the height of its blocks and the whole height of the lanes."""
    above = [INPUTS_ROW if any(b.inputs for b in blocks if b.lane == lane) else 10 for lane in range(lanes)]
    height = [max(b.h for b in blocks if b.lane == lane) for lane in range(lanes)]
    return above, height, sum(above) + sum(height) + (lanes - 1) * CORRIDOR


def shown_edges(blocks, edges):
    """The edges drawn: those carrying data forward in the GPU order (a loop shows as the repeated segment), without
    those whose data also takes a way through other blocks."""
    index = {b.label: i for i, b in enumerate(blocks)}
    forward = {(a, b) for a, b in edges if index[a] < index[b] and (a, b) not in ORDER_ONLY}
    succs = collections.defaultdict(set)
    for a, b in forward:
        succs[a].add(b)

    def detour(a, b):
        stack, seen = [n for n in succs[a] if n != b], set()
        while stack:
            n = stack.pop()
            if n == b:
                return True
            if n not in seen:
                seen.add(n)
                stack.extend(succs[n])
        return False

    return {(a, b) for a, b in forward if not detour(a, b)}


def draw_screen(blocks, edges, segments, heading, sub, stats, out, fonts, machine=""):
    """Blocks in GPU order, each with its lane; segments of a chain: (label, first block index, last block index,
    repeat)."""
    c = Canvas(fonts)
    lanes = max(b.lane for b in blocks) + 1
    segment_gap = 34 * max(0, len(segments) - 1)
    order_space = 102 if segments else 56  # the order axis below the lanes, its caption and the segments
    for scale in (1.0, 0.92, 0.85, 0.78, 0.72, 0.66, 0.6, 0.55):
        size_blocks(blocks, fonts, scale)
        _, _, lanes_height = lane_heights(blocks, lanes)
        if (sum(b.w for b in blocks) + 28 * (len(blocks) - 1) + segment_gap <= RIGHT - LEFT and
                lanes_height + order_space <= GRAPH_BOTTOM - GRAPH_TOP):
            break
    gap = min(90, (RIGHT - LEFT - sum(b.w for b in blocks) - segment_gap) / max(1, len(blocks) - 1))
    starts = {s[1] for s in segments[1:]}
    x = LEFT + (RIGHT - LEFT - sum(b.w for b in blocks) - gap * (len(blocks) - 1) - segment_gap) / 2
    for i, b in enumerate(blocks):
        if i in starts:
            x += 34
        b.x = x
        x += b.w + gap

    # The lanes, from the top; the whole graph centred in its area.
    above, height, lanes_height = lane_heights(blocks, lanes)
    y = GRAPH_TOP + (GRAPH_BOTTOM - GRAPH_TOP - lanes_height - order_space) / 2
    lane_top, lane_y = [], []
    for lane in range(lanes):
        lane_top.append(y)
        y += above[lane]
        lane_y.append(y + height[lane] / 2)
        y += height[lane] + CORRIDOR
    axis_y = y + 22
    index = {b.label: i for i, b in enumerate(blocks)}

    # Data from one block to another, left to right: along the lanes, and down or up in the gaps between the
    # columns. An arrow stays in a lane only where no other block of that lane is in the way.
    def blocked(lane, a, b):
        return any(o.lane == lane and a.x < o.x < b.x for o in blocks)

    channels = collections.Counter()

    def channel(after):
        """x of a vertical run in the gap after block `after`, next to the runs already there."""
        k = channels[after]
        channels[after] += 1
        b = blocks[after]
        return b.x + b.w + gap / 2 + (k - 0.5 * (k > 0)) * 7 * (1 if k % 2 else -1)

    for a, b in sorted(shown_edges(blocks, edges), key=lambda e: (index[e[0]], index[e[1]])):
        a, b = blocks[index[a]], blocks[index[b]]
        ya, yb = lane_y[a.lane], lane_y[b.lane]
        x0, x1 = a.x + a.w, b.x
        if a.lane == b.lane and not blocked(a.lane, a, b):
            points = [(x0, ya), (x1, yb)]
        elif a.lane != b.lane and not blocked(b.lane, a, b):
            cx = channel(index[a.label])
            points = [(x0, ya), (cx, ya), (cx, yb), (x1, yb)]
        elif a.lane != b.lane and not blocked(a.lane, a, b):
            cx = channel(index[b.label] - 1)
            points = [(x0, ya), (cx, ya), (cx, yb), (x1, yb)]
        else:
            # Through the corridor below the upper of the two lanes.
            upper = min(a.lane, b.lane)
            corridor = lane_y[upper] + height[upper] / 2 + CORRIDOR / 2
            ca, cb = channel(index[a.label]), channel(index[b.label] - 1)
            points = [(x0, ya), (ca, ya), (ca, corridor), (cb, corridor), (cb, yb), (x1, yb)]
        points[-1] = (x1 - 2, yb)
        c.line(points, EDGE + (255,), 1.4)
        c.triangle((x1, yb), (x1 - 8, yb - 4.5), (x1 - 8, yb + 4.5), EDGE + (255,))

    # Blocks, and their inputs above them.
    for i, b in enumerate(blocks):
        y0 = lane_y[b.lane] - b.h / 2
        c.rect(b.x, y0, b.x + b.w, y0 + b.h, fill=BOX_FILL + (255,), outline=BOX_LINE + (255,), width=1.2, radius=7)
        y = y0 + b.pad - 2
        for text, size, colour in b.lines:
            c.text(b.x + b.pad, y, text, size, colour)
            y += size + 6
        if b.inputs:
            cx = b.x + b.w / 2
            c.line([(cx, y0 - 14), (cx, y0 - 2)], FAINT + (255,), 1.0)
            c.text(cx, y0 - 18, " · ".join(b.inputs), 13, DIM, anchor="md")

    # The order the GPU runs the blocks in: one after another, left to right. Amber: the next command buffer of a
    # chain, which waits for a semaphore.
    for i, b in enumerate(blocks):
        c.text(b.x + b.w / 2, axis_y, str(i + 1), 15, DIM, anchor="mm")
        if i + 1 < len(blocks):
            n = blocks[i + 1]
            semaphore = n.link != b.link
            colour = SEMAPHORE if semaphore else FAINT
            c.arrow(b.x + b.w / 2 + 14, n.x + n.w / 2 - 14, axis_y, colour + (255,), 2.0 if semaphore else 1.2)

    # A repeated segment: the way back.
    for label, first, last, repeat in segments:
        if repeat > 1:
            xa, xb = blocks[first].x + blocks[first].w / 2, blocks[last].x + blocks[last].w / 2
            y = axis_y + 24
            c.line([(xb, axis_y + 12), (xb, y), (xa, y), (xa, axis_y + 18)], SEMAPHORE + (255,), 2.0)
            c.triangle((xa, axis_y + 11), (xa - 4.5, axis_y + 19), (xa + 4.5, axis_y + 19), SEMAPHORE + (255,))
            c.text((xa + xb) / 2, y + 6, f"{repeat} ×", 15, SEMAPHORE, anchor="ma")

    # Segments of a chain: its graphs.
    if segments:
        seg_y = axis_y + 54
        for label, first, last, repeat in segments:
            xa, xb = blocks[first].x, blocks[last].x + blocks[last].w
            c.line([(xa, seg_y - 7), (xa, seg_y), (xb, seg_y), (xb, seg_y - 7)], FAINT + (255,), 1.2)
            c.text((xa + xb) / 2, seg_y + 9, label, 14, DIM, anchor="ma")

    if machine:
        c.text(LEFT, HEIGHT - 52, f"measured on {machine}", 15, DIM)
    c.text(blocks[0].x, axis_y + 24, "GPU order: the blocks run one after another, left to right", 13, FAINT, anchor="lm")

    c.text(70, 58, heading, 40, TEXT, "Unbounded", 500)
    # The subtitle ends left of the picture in picture (compose_chapter8.py).
    size = next(size for size in (20, 19, 18, 17, 16) if 72 + fonts.width(sub, size) <= PIP_LEFT - 30 or size == 16)
    c.text(72, 118, sub, size, DIM)
    c.text(72, 154, stats, 18, TEXT)
    c.save(out)


def chapter_screen(directory, name, out, fonts, machine="", gaussians=0):
    graph = load(directory, name)
    blocks = GraphBlocks(graph, expand_splatting=True)
    real = json.loads((directory / f"{name}.wall.json").read_text())["unprofiled_wall_ms"]
    total = max(sum(b.ms for b in blocks.blocks), 1e-9)
    for b in blocks.blocks:
        b.lane = LANES[name].get(b.label, 0)
        b.share_text = share_text(b.ms, total)
    count = collections.Counter()
    for e in graph["elements"]:
        count.update(e["commands"])
    d, dr, _ = work(count)
    heading, sub = TITLES[name]
    stats = "   ·   ".join([f"{ms_text(real)} per frame", f"{d:,} compute dispatches", plural(dr, "raster draw"),
                            "1 command buffer, 1 submit"])
    draw_screen(blocks.blocks, blocks.edges, [], heading, sub.format(gaussians=f"{gaussians:,}"), stats, out, fonts,
                machine)
    return len(blocks.blocks), real


def chain_screen(directory, real, out, fonts, machine=""):
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
    step.blocks[0].ms = sum(g.blocks[0].ms for g in steps) / repeat
    step.total_ms = sum(g.total_ms for g in steps) / repeat

    parts = [prepare, denoise, step, decode]
    blocks, edges = [], set()
    for link, (part, times) in enumerate(zip(parts, [1, repeat, repeat, 1])):
        for b in part.blocks:
            b.link = link
            b.lane = LANES["sd15"].get(b.label, 0)
            b.total_ms = b.ms * times
            blocks.append(b)
        edges.update(part.edges)
    # Data one graph of the chain leaves for a later one.
    for i, earlier in enumerate(parts):
        for later in parts[i + 1:]:
            for key, users in earlier.uses.items():
                edges.update((a, b) for a in users for b in later.uses.get(key, ()) if a != b)
    total = max(sum(b.total_ms for b in blocks), 1e-9)
    for b in blocks:
        b.share_text = share_text(b.total_ms, total)
    n_prepare, n_loop = len(prepare.blocks), len(denoise.blocks) + 1
    # One command buffer per graph in the chain: the denoiser's four times, and one LCM step graph per step.
    segments = [("prepare · 1 command buffer", 0, n_prepare - 1, 1),
                (f"denoise ×{repeat} + LCM step 1–{repeat} · {2 * repeat} command buffers", n_prepare,
                 n_prepare + n_loop - 1, repeat),
                ("decode · 1 command buffer", n_prepare + n_loop, len(blocks) - 1, 1)]
    count = collections.Counter()
    for link in chain["links"]:
        for e in load(directory, link)["elements"]:
            count.update(e["commands"])
    d, dr, _ = work(count)
    stats = "   ·   ".join([f"{ms_text(real)} per frame", f"{d:,} compute dispatches", plural(dr, "raster draw"),
                            f"{len(chain['links'])} command buffers, 1 submit"])
    heading, sub = TITLES["sd15"]
    draw_screen(blocks, edges, segments, heading, sub, stats, out, fonts, machine)
    return len(blocks), real


def chain_time(log):
    """The GPU path's time per generated frame from the log of a render of techdemo000_example."""
    m = re.search(r"person generation: \d+ frames.*gpu path ([\d.]+) ms per frame", Path(log).read_text())
    if not m:
        raise SystemExit(f"{log}: no time of the GPU path")
    return float(m.group(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*")
    parser.add_argument("--dir", default="build/TestingOutput/techdemo000_graphs")
    parser.add_argument("--fonts", default="build/fonts")
    parser.add_argument("--chain", help="draw the SD1.5 chain written by techdemo000_example --dump-graphs")
    parser.add_argument("--chain-log", help="log of a render without --dump-graphs, for the chain's time per frame")
    parser.add_argument("--hardware", help="the machine the graphs were measured on (default: this one)")
    parser.add_argument("--scene", default="data/lantern.spz", help="the scene the graphs splat")
    args = parser.parse_args()
    fonts = Fonts(args.fonts)
    machine = args.hardware if args.hardware is not None else hardware()
    if args.chain:
        directory = Path(args.chain)
        out = directory.parent / "sd15_chain.png"
        if not args.chain_log:
            parser.error("--chain needs --chain-log")
        count, total = chain_screen(directory, chain_time(args.chain_log), out, fonts, machine)
        print(f"sd15_chain: {count} blocks, {total:.1f} ms, {out}")
        return
    directory = Path(args.dir)
    for name in args.names or [n for n in TITLES if (directory / f"{n}.json").exists()]:
        out = directory / f"{name}.png"
        count, total = chapter_screen(directory, name, out, fonts, machine, gaussian_count(args.scene))
        print(f"{name}: {count} blocks, {total:.2f} ms, {out}")


if __name__ == "__main__":
    main()
