# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Build the Gaussian depth article from the example's 64-case matrix output.

Run from the repository root:
  python scripts/dev/gaussian_depth_report.py build/TestingOutput/gaussian-depth-composition
No packages beyond the Python standard library are needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
STATES = [(0, 0, "Off"), (1, 0, "Test"), (0, 1, "Write"), (1, 1, "Both")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    metadata = json.loads((args.results / "metadata.json").read_text())
    if not metadata["profiling"] or metadata["validation"]:
        parser.error("use a profiled Release matrix with validation disabled")
    with (args.results / "timings.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    cases = {row["case"]: row for row in rows}
    if len(rows) != 64 or len(cases) != 64:
        parser.error("expected exactly 64 distinct combinations")
    assets = ROOT / "docs/articles/assets/gaussian-depth-composition"
    assets.mkdir(parents=True, exist_ok=True)
    expected = set()
    for backend in ["raster", "compute"]:
        for order in ["raster-first", "gs-first"]:
            for gt, gw, _ in STATES:
                for rt, rw, _ in STATES:
                    key = f"{backend}_{order}_gs-t{gt}w{gw}_rp-t{rt}w{rw}"
                    expected.add(key)
                    if key not in cases or not (args.results / (key + ".png")).is_file():
                        parser.error(f"missing case or image: {key}")
    if set(cases) != expected:
        parser.error("unexpected case names")
    for key in cases:
        shutil.copyfile(args.results / (key + ".png"), assets / (key + ".png"))
    for name in ["metadata.json", "timings.csv", "elements.csv", "runs.csv"]:
        (assets / name).write_text((args.results / name).read_text(), encoding="utf-8", newline="\n")
    base = "assets/gaussian-depth-composition/"
    page = ["""# Gaussian splatting and rasterized boxes

This example renders the lantern together with two opaque triangle meshes.
The orange box is nearer the camera on the left; the blue box is farther away
on the right. Both overlap the lantern's silhouette. Switching the pass order
and depth controls makes it possible to see which renderer hides which surface.

```{image} assets/gaussian-depth-composition/raster_raster-first_gs-t1w1_rp-t1w1.png
:alt: Lantern between an orange box in front and a blue box behind, with depth enabled
:width: 512px
```

The example is `examples/gaussian_depth_composition_example.cpp`. It uses the
same camera for both renderers and one shared `DepthAttachment`, cleared to 1
each execution. Both passes preserve the existing color image. The box meshes
cull back faces and use ordinary hardware rasterization; their fragment shader
does not sample Gaussian depth or perform an extra software depth test.

## What to look for

With depth working, the orange box covers the overlapping part of the lantern,
while the lantern covers the overlapping part of the blue box. With depth
disabled, the later pass paints over the earlier one in their overlapping area.

Depth **testing** decides whether a fragment may contribute color. Depth
**writing** leaves a surface depth for subsequent work to test against. Across
these two passes, the important pair is the first pass's write switch and the
second pass's test switch:

| Order | Controls needed for mutual occlusion |
|---|---|
| Boxes, then Gaussians | Render-pass writing + Gaussian testing |
| Gaussians, then boxes | Gaussian writing + render-pass testing |

The other controls are still exercised in every combination. A test against
freshly cleared depth has no earlier surface to reject; a write in the last
pass has no later pass to influence. Depth can also resolve occlusion within
a mesh pass, though the boxes here have little mutual screen overlap.

### A Gaussian depth is an estimated surface

The lantern consists of partially transparent splats, blended into the image
{cite:p}`kerbl2023_3Dgaussians`. Its color can come from several depths. A single
depth value cannot describe all those translucent layers, so correct depth
controls do not make the two pass orders pixel-identical.

The gallery uses the backend defaults, with an opacity threshold of 0.5:

| Backend | Depth written at a pixel |
|---|---|
| Compute | The splat where accumulated opacity first reaches 0.5 |
| Raster | The nearest splat whose individual opacity at that pixel reaches 0.5 |

The raster rule can leave holes where several translucent splats together
produce visible color but no individual splat reaches the threshold. Drawing
the boxes afterward can then paint through those areas. Drawing opaque geometry
first lets the Gaussian renderer reject contributions behind it before blending,
and is generally the better order for this scene. This is a limitation of the
depth estimate, rather than an exact representation of transparent geometry.

`--depth-mode nearest` or `--depth-mode accumulated` selects the same rule on
both backends. The raster accumulated-opacity method adds a depth-only compute
pipeline and can cost substantially more. The 64 images below use `default`;
they do not compare all depth-estimation methods.

## All 64 combinations

Each grid has Gaussian settings in its **rows**, and box render-pass settings
in its **columns**. **Off** means neither testing nor writing; **Test** means
testing only; **Write** means writing only; **Both** enables both. Click an image
to open it at full resolution. The number below each image is the measured
whole-graph GPU time in milliseconds, including both renderers and shared-depth
work. Equal-looking cells are intentional: some controls do not affect this
particular scene and pass order. In this capture, each grid has two distinct
images, selected by the first-pass write and second-pass test controls above.
"""]
    for backend in ["raster", "compute"]:
        for order in ["raster-first", "gs-first"]:
            title = "Boxes first" if order == "raster-first" else "Gaussians first"
            page += [f"\n### {backend.capitalize()} backend: {title.lower()}\n\n",
                     "::::{list-table}\n:header-rows: 1\n:stub-columns: 1\n:widths: 12 22 22 22 22\n\n",
                     "* - Gaussian / render pass\n  - Off\n  - Test\n  - Write\n  - Both\n"]
            for gt, gw, label in STATES:
                page.append(f"* - {label}\n")
                for rt, rw, rp_label in STATES:
                    key = f"{backend}_{order}_gs-t{gt}w{gw}_rp-t{rt}w{rw}"
                    link = base + key + ".png"
                    alt = f"{backend}, {title.lower()}, Gaussians {label.lower()}, render pass {rp_label.lower()}"
                    time = float(cases[key]["gpu_ms"])
                    page.append(f"  - ```{{image}} {link}\n    :alt: {alt}\n    :width: 160px\n    ```\n\n    {time:.3f} ms\n")
            page.append("::::\n")
    page += ["\n## Profiling\n\n",
             f"Measured on **{metadata['device']}**, {metadata.get('platform', platform.system())}, Release build, validation disabled; "
             f"**{metadata['width']} × {metadata['height']} pixels**, **{metadata['gaussians']:,} Gaussians**, "
             "one execution path and a fixed camera.\n\n",
             f"For each configuration, the app runs {metadata['runs']} repetitions. Each repetition discards "
             f"{metadata['warmup_per_run']} warm-up executions and averages {metadata['samples_per_run']} "
             "measured executions. The gallery reports the median of those run means. The order of the "
             "64 configurations is shuffled with a fixed seed to mix backends and settings.\n\n",
             f"The largest ratio between a configuration's maximum and minimum run means was "
             f"{max(float(row['max_run_ms']) / float(row['min_run_ms']) - 1 for row in rows) * 100:.1f}%. "
             "Use the small timing differences between cells as exploratory measurements, rather than a reliable ranking.\n\n",
             "`ComputeGraph::enableProfiling()` places Vulkan timestamp queries around graph elements. "
             "The graph span includes their barriers, color and depth clears, both renderers, and any "
             "depth image/buffer transfers. Each sample uses `submitAndWait()`. Loading, graph compilation, "
             "CPU submission time, PNG export and presentation are outside the GPU span. Per-element "
             "timestamp instrumentation remains enabled, so these numbers describe the profiled graph; "
             "they are not uninstrumented frame throughput.\n\n",
             "| Backend / order | All controls off | Both enabled on both passes | Difference |\n",
             "|---|---:|---:|---:|\n"]
    for backend in ["raster", "compute"]:
        for order in ["raster-first", "gs-first"]:
            off = float(cases[f"{backend}_{order}_gs-t0w0_rp-t0w0"]["gpu_ms"])
            on = float(cases[f"{backend}_{order}_gs-t1w1_rp-t1w1"]["gpu_ms"])
            page.append(f"| {backend}, {order} | {off:.3f} ms | {on:.3f} ms | {(on / off - 1) * 100:+.1f}% |\n")
    page += ["""
The off baseline still has a shared attachment and its clear; it disables both
renderers' depth testing and writing. The difference includes both the work of
depth handling and any rendering work saved through occlusion. It therefore
does not isolate the implementation overhead of depth handling. Small timing
differences can also be measurement noise. The backends use different Gaussian
depth estimates in this gallery, so their images and depth costs need to be
interpreted with that distinction in mind.

Download the measurements for individual combinations and graph elements:

""",
             f"- {{download}}`Configuration timings <{base}timings.csv>` (including the minimum and maximum run means).\n",
             f"- {{download}}`Per-element timings <{base}elements.csv>`.\n",
             f"- {{download}}`Individual run means <{base}runs.csv>`.\n",
             f"- {{download}}`Device and sampling settings <{base}metadata.json>`.\n",
             r"""
See [Profiling](../guides/profiling.md) for interpreting graph and element timings.

## Reproduce or change a configuration

Build and run from the repository root. For Visual Studio:

```powershell
cmake --build build --config Release --target gaussian_depth_composition_example
.\build\examples\Release\gaussian_depth_composition_example.exe --matrix
python scripts/dev/gaussian_depth_report.py build/TestingOutput/gaussian-depth-composition
```

For a single-configuration Release build, the executable is
`build/examples/gaussian_depth_composition_example`. The report script uses only
Python's standard library; it copies the images and CSV files into this article
and regenerates its comparison grids.

For one configuration, choose the backend, order and four switches explicitly:

```powershell
.\build\examples\Release\gaussian_depth_composition_example.exe `
    --backend raster --order raster-first `
    --gs-test on --gs-write off --raster-test off --raster-write on `
    --out-dir build/TestingOutput/depth-one
```

Both depth testing and writing default to `on`. `--matrix` replaces these four
switches, backend and order with every combination. It preserves the chosen
resolution, depth method and sampling settings. To compare depth estimates, run
the matrix again with `--depth-mode nearest` or `--depth-mode accumulated` into
a different `--out-dir`. `--profile off` records no timestamps and leaves timing
fields empty; use that for rendering without profiling instrumentation.

## References

```{bibliography}
:filter: docname in docnames
:labelprefix: Depth-
```
"""]
    (ROOT / "docs/articles/gaussian-depth-composition.md").write_text("".join(page), encoding="utf-8", newline="\n")
    print(f"Wrote the article, {len(cases)} images, and profiling data to {assets}")


if __name__ == "__main__":
    main()
