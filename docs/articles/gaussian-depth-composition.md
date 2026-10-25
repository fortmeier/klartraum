# Gaussian splatting and rasterized boxes

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

### Raster backend: boxes first

::::{list-table}
:header-rows: 1
:stub-columns: 1
:widths: 12 22 22 22 22

* - Gaussian / render pass
  - Off
  - Test
  - Write
  - Both
* - Off
  - ```{image} assets/gaussian-depth-composition/raster_raster-first_gs-t0w0_rp-t0w0.png
    :alt: raster, boxes first, Gaussians off, render pass off
    :width: 160px
    ```

    1.338 ms
  - ```{image} assets/gaussian-depth-composition/raster_raster-first_gs-t0w0_rp-t1w0.png
    :alt: raster, boxes first, Gaussians off, render pass test
    :width: 160px
    ```

    1.359 ms
  - ```{image} assets/gaussian-depth-composition/raster_raster-first_gs-t0w0_rp-t0w1.png
    :alt: raster, boxes first, Gaussians off, render pass write
    :width: 160px
    ```

    1.397 ms
  - ```{image} assets/gaussian-depth-composition/raster_raster-first_gs-t0w0_rp-t1w1.png
    :alt: raster, boxes first, Gaussians off, render pass both
    :width: 160px
    ```

    1.436 ms
* - Test
  - ```{image} assets/gaussian-depth-composition/raster_raster-first_gs-t1w0_rp-t0w0.png
    :alt: raster, boxes first, Gaussians test, render pass off
    :width: 160px
    ```

    1.339 ms
  - ```{image} assets/gaussian-depth-composition/raster_raster-first_gs-t1w0_rp-t1w0.png
    :alt: raster, boxes first, Gaussians test, render pass test
    :width: 160px
    ```

    1.458 ms
  - ```{image} assets/gaussian-depth-composition/raster_raster-first_gs-t1w0_rp-t0w1.png
    :alt: raster, boxes first, Gaussians test, render pass write
    :width: 160px
    ```

    1.431 ms
  - ```{image} assets/gaussian-depth-composition/raster_raster-first_gs-t1w0_rp-t1w1.png
    :alt: raster, boxes first, Gaussians test, render pass both
    :width: 160px
    ```

    1.334 ms
* - Write
  - ```{image} assets/gaussian-depth-composition/raster_raster-first_gs-t0w1_rp-t0w0.png
    :alt: raster, boxes first, Gaussians write, render pass off
    :width: 160px
    ```

    1.457 ms
  - ```{image} assets/gaussian-depth-composition/raster_raster-first_gs-t0w1_rp-t1w0.png
    :alt: raster, boxes first, Gaussians write, render pass test
    :width: 160px
    ```

    1.428 ms
  - ```{image} assets/gaussian-depth-composition/raster_raster-first_gs-t0w1_rp-t0w1.png
    :alt: raster, boxes first, Gaussians write, render pass write
    :width: 160px
    ```

    1.429 ms
  - ```{image} assets/gaussian-depth-composition/raster_raster-first_gs-t0w1_rp-t1w1.png
    :alt: raster, boxes first, Gaussians write, render pass both
    :width: 160px
    ```

    1.513 ms
* - Both
  - ```{image} assets/gaussian-depth-composition/raster_raster-first_gs-t1w1_rp-t0w0.png
    :alt: raster, boxes first, Gaussians both, render pass off
    :width: 160px
    ```

    1.432 ms
  - ```{image} assets/gaussian-depth-composition/raster_raster-first_gs-t1w1_rp-t1w0.png
    :alt: raster, boxes first, Gaussians both, render pass test
    :width: 160px
    ```

    1.437 ms
  - ```{image} assets/gaussian-depth-composition/raster_raster-first_gs-t1w1_rp-t0w1.png
    :alt: raster, boxes first, Gaussians both, render pass write
    :width: 160px
    ```

    1.457 ms
  - ```{image} assets/gaussian-depth-composition/raster_raster-first_gs-t1w1_rp-t1w1.png
    :alt: raster, boxes first, Gaussians both, render pass both
    :width: 160px
    ```

    1.438 ms
::::

### Raster backend: gaussians first

::::{list-table}
:header-rows: 1
:stub-columns: 1
:widths: 12 22 22 22 22

* - Gaussian / render pass
  - Off
  - Test
  - Write
  - Both
* - Off
  - ```{image} assets/gaussian-depth-composition/raster_gs-first_gs-t0w0_rp-t0w0.png
    :alt: raster, gaussians first, Gaussians off, render pass off
    :width: 160px
    ```

    1.336 ms
  - ```{image} assets/gaussian-depth-composition/raster_gs-first_gs-t0w0_rp-t1w0.png
    :alt: raster, gaussians first, Gaussians off, render pass test
    :width: 160px
    ```

    1.336 ms
  - ```{image} assets/gaussian-depth-composition/raster_gs-first_gs-t0w0_rp-t0w1.png
    :alt: raster, gaussians first, Gaussians off, render pass write
    :width: 160px
    ```

    1.360 ms
  - ```{image} assets/gaussian-depth-composition/raster_gs-first_gs-t0w0_rp-t1w1.png
    :alt: raster, gaussians first, Gaussians off, render pass both
    :width: 160px
    ```

    1.346 ms
* - Test
  - ```{image} assets/gaussian-depth-composition/raster_gs-first_gs-t1w0_rp-t0w0.png
    :alt: raster, gaussians first, Gaussians test, render pass off
    :width: 160px
    ```

    1.390 ms
  - ```{image} assets/gaussian-depth-composition/raster_gs-first_gs-t1w0_rp-t1w0.png
    :alt: raster, gaussians first, Gaussians test, render pass test
    :width: 160px
    ```

    1.336 ms
  - ```{image} assets/gaussian-depth-composition/raster_gs-first_gs-t1w0_rp-t0w1.png
    :alt: raster, gaussians first, Gaussians test, render pass write
    :width: 160px
    ```

    1.339 ms
  - ```{image} assets/gaussian-depth-composition/raster_gs-first_gs-t1w0_rp-t1w1.png
    :alt: raster, gaussians first, Gaussians test, render pass both
    :width: 160px
    ```

    1.400 ms
* - Write
  - ```{image} assets/gaussian-depth-composition/raster_gs-first_gs-t0w1_rp-t0w0.png
    :alt: raster, gaussians first, Gaussians write, render pass off
    :width: 160px
    ```

    1.495 ms
  - ```{image} assets/gaussian-depth-composition/raster_gs-first_gs-t0w1_rp-t1w0.png
    :alt: raster, gaussians first, Gaussians write, render pass test
    :width: 160px
    ```

    1.472 ms
  - ```{image} assets/gaussian-depth-composition/raster_gs-first_gs-t0w1_rp-t0w1.png
    :alt: raster, gaussians first, Gaussians write, render pass write
    :width: 160px
    ```

    1.457 ms
  - ```{image} assets/gaussian-depth-composition/raster_gs-first_gs-t0w1_rp-t1w1.png
    :alt: raster, gaussians first, Gaussians write, render pass both
    :width: 160px
    ```

    1.437 ms
* - Both
  - ```{image} assets/gaussian-depth-composition/raster_gs-first_gs-t1w1_rp-t0w0.png
    :alt: raster, gaussians first, Gaussians both, render pass off
    :width: 160px
    ```

    1.428 ms
  - ```{image} assets/gaussian-depth-composition/raster_gs-first_gs-t1w1_rp-t1w0.png
    :alt: raster, gaussians first, Gaussians both, render pass test
    :width: 160px
    ```

    1.428 ms
  - ```{image} assets/gaussian-depth-composition/raster_gs-first_gs-t1w1_rp-t0w1.png
    :alt: raster, gaussians first, Gaussians both, render pass write
    :width: 160px
    ```

    1.428 ms
  - ```{image} assets/gaussian-depth-composition/raster_gs-first_gs-t1w1_rp-t1w1.png
    :alt: raster, gaussians first, Gaussians both, render pass both
    :width: 160px
    ```

    1.446 ms
::::

### Compute backend: boxes first

::::{list-table}
:header-rows: 1
:stub-columns: 1
:widths: 12 22 22 22 22

* - Gaussian / render pass
  - Off
  - Test
  - Write
  - Both
* - Off
  - ```{image} assets/gaussian-depth-composition/compute_raster-first_gs-t0w0_rp-t0w0.png
    :alt: compute, boxes first, Gaussians off, render pass off
    :width: 160px
    ```

    28.369 ms
  - ```{image} assets/gaussian-depth-composition/compute_raster-first_gs-t0w0_rp-t1w0.png
    :alt: compute, boxes first, Gaussians off, render pass test
    :width: 160px
    ```

    26.980 ms
  - ```{image} assets/gaussian-depth-composition/compute_raster-first_gs-t0w0_rp-t0w1.png
    :alt: compute, boxes first, Gaussians off, render pass write
    :width: 160px
    ```

    27.005 ms
  - ```{image} assets/gaussian-depth-composition/compute_raster-first_gs-t0w0_rp-t1w1.png
    :alt: compute, boxes first, Gaussians off, render pass both
    :width: 160px
    ```

    26.662 ms
* - Test
  - ```{image} assets/gaussian-depth-composition/compute_raster-first_gs-t1w0_rp-t0w0.png
    :alt: compute, boxes first, Gaussians test, render pass off
    :width: 160px
    ```

    35.786 ms
  - ```{image} assets/gaussian-depth-composition/compute_raster-first_gs-t1w0_rp-t1w0.png
    :alt: compute, boxes first, Gaussians test, render pass test
    :width: 160px
    ```

    34.119 ms
  - ```{image} assets/gaussian-depth-composition/compute_raster-first_gs-t1w0_rp-t0w1.png
    :alt: compute, boxes first, Gaussians test, render pass write
    :width: 160px
    ```

    32.272 ms
  - ```{image} assets/gaussian-depth-composition/compute_raster-first_gs-t1w0_rp-t1w1.png
    :alt: compute, boxes first, Gaussians test, render pass both
    :width: 160px
    ```

    32.535 ms
* - Write
  - ```{image} assets/gaussian-depth-composition/compute_raster-first_gs-t0w1_rp-t0w0.png
    :alt: compute, boxes first, Gaussians write, render pass off
    :width: 160px
    ```

    35.562 ms
  - ```{image} assets/gaussian-depth-composition/compute_raster-first_gs-t0w1_rp-t1w0.png
    :alt: compute, boxes first, Gaussians write, render pass test
    :width: 160px
    ```

    36.276 ms
  - ```{image} assets/gaussian-depth-composition/compute_raster-first_gs-t0w1_rp-t0w1.png
    :alt: compute, boxes first, Gaussians write, render pass write
    :width: 160px
    ```

    36.718 ms
  - ```{image} assets/gaussian-depth-composition/compute_raster-first_gs-t0w1_rp-t1w1.png
    :alt: compute, boxes first, Gaussians write, render pass both
    :width: 160px
    ```

    36.273 ms
* - Both
  - ```{image} assets/gaussian-depth-composition/compute_raster-first_gs-t1w1_rp-t0w0.png
    :alt: compute, boxes first, Gaussians both, render pass off
    :width: 160px
    ```

    35.944 ms
  - ```{image} assets/gaussian-depth-composition/compute_raster-first_gs-t1w1_rp-t1w0.png
    :alt: compute, boxes first, Gaussians both, render pass test
    :width: 160px
    ```

    33.560 ms
  - ```{image} assets/gaussian-depth-composition/compute_raster-first_gs-t1w1_rp-t0w1.png
    :alt: compute, boxes first, Gaussians both, render pass write
    :width: 160px
    ```

    34.915 ms
  - ```{image} assets/gaussian-depth-composition/compute_raster-first_gs-t1w1_rp-t1w1.png
    :alt: compute, boxes first, Gaussians both, render pass both
    :width: 160px
    ```

    34.266 ms
::::

### Compute backend: gaussians first

::::{list-table}
:header-rows: 1
:stub-columns: 1
:widths: 12 22 22 22 22

* - Gaussian / render pass
  - Off
  - Test
  - Write
  - Both
* - Off
  - ```{image} assets/gaussian-depth-composition/compute_gs-first_gs-t0w0_rp-t0w0.png
    :alt: compute, gaussians first, Gaussians off, render pass off
    :width: 160px
    ```

    28.763 ms
  - ```{image} assets/gaussian-depth-composition/compute_gs-first_gs-t0w0_rp-t1w0.png
    :alt: compute, gaussians first, Gaussians off, render pass test
    :width: 160px
    ```

    26.756 ms
  - ```{image} assets/gaussian-depth-composition/compute_gs-first_gs-t0w0_rp-t0w1.png
    :alt: compute, gaussians first, Gaussians off, render pass write
    :width: 160px
    ```

    28.415 ms
  - ```{image} assets/gaussian-depth-composition/compute_gs-first_gs-t0w0_rp-t1w1.png
    :alt: compute, gaussians first, Gaussians off, render pass both
    :width: 160px
    ```

    26.717 ms
* - Test
  - ```{image} assets/gaussian-depth-composition/compute_gs-first_gs-t1w0_rp-t0w0.png
    :alt: compute, gaussians first, Gaussians test, render pass off
    :width: 160px
    ```

    33.891 ms
  - ```{image} assets/gaussian-depth-composition/compute_gs-first_gs-t1w0_rp-t1w0.png
    :alt: compute, gaussians first, Gaussians test, render pass test
    :width: 160px
    ```

    34.663 ms
  - ```{image} assets/gaussian-depth-composition/compute_gs-first_gs-t1w0_rp-t0w1.png
    :alt: compute, gaussians first, Gaussians test, render pass write
    :width: 160px
    ```

    35.911 ms
  - ```{image} assets/gaussian-depth-composition/compute_gs-first_gs-t1w0_rp-t1w1.png
    :alt: compute, gaussians first, Gaussians test, render pass both
    :width: 160px
    ```

    35.278 ms
* - Write
  - ```{image} assets/gaussian-depth-composition/compute_gs-first_gs-t0w1_rp-t0w0.png
    :alt: compute, gaussians first, Gaussians write, render pass off
    :width: 160px
    ```

    36.350 ms
  - ```{image} assets/gaussian-depth-composition/compute_gs-first_gs-t0w1_rp-t1w0.png
    :alt: compute, gaussians first, Gaussians write, render pass test
    :width: 160px
    ```

    33.399 ms
  - ```{image} assets/gaussian-depth-composition/compute_gs-first_gs-t0w1_rp-t0w1.png
    :alt: compute, gaussians first, Gaussians write, render pass write
    :width: 160px
    ```

    34.176 ms
  - ```{image} assets/gaussian-depth-composition/compute_gs-first_gs-t0w1_rp-t1w1.png
    :alt: compute, gaussians first, Gaussians write, render pass both
    :width: 160px
    ```

    33.878 ms
* - Both
  - ```{image} assets/gaussian-depth-composition/compute_gs-first_gs-t1w1_rp-t0w0.png
    :alt: compute, gaussians first, Gaussians both, render pass off
    :width: 160px
    ```

    33.454 ms
  - ```{image} assets/gaussian-depth-composition/compute_gs-first_gs-t1w1_rp-t1w0.png
    :alt: compute, gaussians first, Gaussians both, render pass test
    :width: 160px
    ```

    36.493 ms
  - ```{image} assets/gaussian-depth-composition/compute_gs-first_gs-t1w1_rp-t0w1.png
    :alt: compute, gaussians first, Gaussians both, render pass write
    :width: 160px
    ```

    35.540 ms
  - ```{image} assets/gaussian-depth-composition/compute_gs-first_gs-t1w1_rp-t1w1.png
    :alt: compute, gaussians first, Gaussians both, render pass both
    :width: 160px
    ```

    33.918 ms
::::

## Profiling

Measured on **NVIDIA GeForce RTX 2070 SUPER**, Windows, Release build, validation disabled; **512 × 512 pixels**, **145,562 Gaussians**, one execution path and a fixed camera.

For each configuration, the app runs 3 repetitions. Each repetition discards 10 warm-up executions and averages 30 measured executions. The gallery reports the median of those run means. The order of the 64 configurations is shuffled with a fixed seed to mix backends and settings.

The largest ratio between a configuration's maximum and minimum run means was 11.3%. Use the small timing differences between cells as exploratory measurements, rather than a reliable ranking.

`ComputeGraph::enableProfiling()` places Vulkan timestamp queries around graph elements. The graph span includes their barriers, color and depth clears, both renderers, and any depth image/buffer transfers. Each sample uses `submitAndWait()`. Loading, graph compilation, CPU submission time, PNG export and presentation are outside the GPU span. Per-element timestamp instrumentation remains enabled, so these numbers describe the profiled graph; they are not uninstrumented frame throughput.

| Backend / order | All controls off | Both enabled on both passes | Difference |
|---|---:|---:|---:|
| raster, raster-first | 1.338 ms | 1.438 ms | +7.5% |
| raster, gs-first | 1.336 ms | 1.446 ms | +8.2% |
| compute, raster-first | 28.369 ms | 34.266 ms | +20.8% |
| compute, gs-first | 28.763 ms | 33.918 ms | +17.9% |

The off baseline still has a shared attachment and its clear; it disables both
renderers' depth testing and writing. The difference includes both the work of
depth handling and any rendering work saved through occlusion. It therefore
does not isolate the implementation overhead of depth handling. Small timing
differences can also be measurement noise. The backends use different Gaussian
depth estimates in this gallery, so their images and depth costs need to be
interpreted with that distinction in mind.

Download the measurements for individual combinations and graph elements:

- {download}`Configuration timings <assets/gaussian-depth-composition/timings.csv>` (including the minimum and maximum run means).
- {download}`Per-element timings <assets/gaussian-depth-composition/elements.csv>`.
- {download}`Individual run means <assets/gaussian-depth-composition/runs.csv>`.
- {download}`Device and sampling settings <assets/gaussian-depth-composition/metadata.json>`.

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
