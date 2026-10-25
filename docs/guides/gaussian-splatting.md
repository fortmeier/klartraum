# Gaussian splatting

Klartraum renders 3D Gaussian splatting scenes stored in Niantic's
[`.spz` format](https://github.com/nianticlabs/spz).

## Loading a scene

{cpp:class}`klartraum::GaussianDataStandard` loads an `.spz` file (optionally
mirrored across the Y axis, as needed for Nerfstudio exports) and uploads the
Gaussians into GPU buffers. One instance can be shared by several backends or
viewports, so the scene is loaded only once.

## Backends

`createGaussianSplatting()` returns a {cpp:class}`klartraum::GaussianSplattingInterface`
with one of two rendering implementations. Both have the same inputs and expose
optional depth through `getDepth()`, so switching the backend does not change
the surrounding graph.

| Backend | Class | How it works |
|---|---|---|
| `GsplatBackend::Compute` | {cpp:class}`klartraum::VulkanGaussianSplatting` | Sorts per screen tile and splats in compute shaders |
| `GsplatBackend::Raster` | `klartraum::VulkanGaussianSplattingRaster` | Sorts once and draws with the hardware rasterizer |

For a walkthrough of culling, compaction and equal-depth ordering, see
{doc}`../articles/deterministic-raster-order`.

For drawing Gaussians over a background image and controlling target clearing,
see {doc}`render-targets`.

## The compute pipeline

The compute backend runs these stages every frame (shaders in
`shaders/gsplat/`):

1. **Projection:** projects each Gaussian into screen space
   (`gsplat_projection.comp`).
2. **Binning:** counts how many screen bins each Gaussian touches, computes a
   prefix sum and scatters the Gaussians into their bins
   (`gsplat_binning_count`, `gsplat_binning_prefix_sum`,
   `gsplat_binning_scatter`).
3. **Sort keys:** builds a key from bin and depth for every entry
   (`gsplat_extract_sort_keys`).
4. **Radix sort:** sorts the keys on the GPU (histogram, histogram prefix sum,
   scatter).
5. **Gather:** reorders the Gaussian data by the sorted keys
   (`gsplat_gather_sorted`).
6. **Bin bounds:** finds where each bin starts and ends in the sorted list
   (`gsplat_bin_bounds`).
7. **Splatting:** blends the Gaussians of each bin front to back into the
   image (`gsplat_binned_splatting`).

## Tuning

{cpp:struct}`klartraum::GsplatConfig` holds settings that different hardware
benefits from, for example the footprint multiplier used for binning, the
radix-sort workgroup cap, the splatting tile size and the spherical-harmonics
degree. Set them before the splatting element is created.

## Depth output

Set `GsplatConfig::outputDepth` to export a per-pixel float storage buffer,
available through `splatting->getDepth()`. It contains the selected Gaussian's
NDC depth (`clip.z / clip.w`), or positive infinity where no surface was found.
Consumers of that buffer must also depend on the splatting backend.

`depthMode` selects the surface estimate on either backend:

| Mode | Surface estimate |
|---|---|
| `GaussianDepthMode::AccumulatedOpacity` | The first Gaussian where cumulative opacity reaches `depthOpacityThreshold`. |
| `GaussianDepthMode::NearestOpaqueSplat` | The nearest Gaussian whose individual per-pixel opacity reaches `depthOpacityThreshold`. |
| `GaussianDepthMode::BackendDefault` | Accumulated opacity for compute; nearest opaque splat for raster. |

The threshold defaults to 0.5 and must be in `(0, 1]`. The estimates can differ:
several translucent splats can jointly reach 50% opacity even though none does
so individually. Depth export alone leaves RGB unchanged.

## Mixing Gaussians and raster geometry

A {cpp:class}`klartraum::DepthAttachment` owns one `D32_SFLOAT` depth image per
path. Share it between render passes and Gaussian backends to test against
previous geometry and write depths for later geometry. Attachments use normal
Vulkan depth: values from 0 to 1, with smaller values nearer and a clear value
of 1. Each participating pass must use the same camera and extent.

For a render pass, call `pass->setDepthAttachment(depth, {test, write})` before
compiling. For Gaussians, set `config.depthAttachment = depth` and
`config.depthState = {test, write}` before construction. Testing and writing
are independent; both default to enabled when an attachment is supplied.
Omitting the attachment disables both. `outputDepth` separately controls
whether `getDepth()` exports a Gaussian-only storage buffer.

Connect each stage to the previous stage's color output, and set
`RenderPass::setLoadExisting(true)` to retain earlier drawing. This orders the
shared color and depth accesses. Both Gaussian backends forward their color
image, so they can be connected directly to a later render pass.

This function renders either order with an initialized context, scene, camera
and draw component:

```cpp
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/renderpass.hpp"
#include "klartraum/rendering/gaussian_splatting_factory.hpp"
#include "klartraum/offscreen_target.hpp"

std::shared_ptr<klartraum::OffscreenTarget> renderMixed(
    klartraum::VulkanContext& vc,
    std::shared_ptr<klartraum::GaussianDataStandard> model,
    std::shared_ptr<klartraum::CameraUboType> camera,
    std::shared_ptr<klartraum::DrawComponent> geometry,
    klartraum::GsplatBackend backend, bool geometryFirst) {
    using namespace klartraum;
    const VkExtent2D extent{512, 512};
    auto target = std::make_shared<OffscreenTarget>(vc, extent, 1);
    auto depth = std::make_shared<DepthAttachment>(vc, extent, 1);
    auto drawGeometry = [&](std::shared_ptr<ImageViewSrc> source) {
        auto pass = std::make_shared<RenderPass>(vc.getSwapChainImageFormat(), extent);
        pass->setInput(source, 0);
        pass->setInput(camera, 1);
        pass->setLoadExisting(true);
        pass->setDepthAttachment(depth, {true, true});
        pass->addDrawComponent(geometry);
        return pass;
    };

    std::shared_ptr<ImageViewSrc> source = target;
    if (geometryFirst) source = drawGeometry(source);
    GsplatConfig config;
    config.depthAttachment = depth;
    config.depthState = {true, true};
    config.depthMode = GaussianDepthMode::AccumulatedOpacity;
    auto splats = createGaussianSplatting(vc, backend, source, camera, model, config);
    ComputeGraphElementPtr output = splats;
    if (!geometryFirst) output = drawGeometry(splats);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(output);
    camera->update(0);
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    return target;
}
```

The attachment clears once per graph execution. `setClear(false)` preserves
its contents across executions; call it before compilation. Paths have
independent images, and the path count and extent must match the graph.

Both render orders are supported, but a Gaussian depth is a surface estimate.
A single depth value cannot represent arbitrary translucent layers, so changing
the order need not produce identical blending at translucent edges. Rendering
opaque geometry first lets splatting reject contributions behind its depth
while blending the remaining Gaussians over its color.

### Implementation and cost

Raster passes use the depth attachment directly. Gaussian compute shaders
read a GPU buffer snapshot of it; when writing is enabled, the updated snapshot
is copied back on the GPU. Raster splatting uses native, read-only depth testing
and integer atomics for nearest-splat depth, followed by decoding and an optional
attachment update. Test-only raster rendering needs no snapshot or atomic writes.

Accumulated-opacity depth on the raster backend uses an additional depth-only
compute pipeline: projection, binning, sorting and depth selection. It is more
expensive than nearest-splat depth and uses the compute backend's footprint and
bin-capacity settings. The same path provides a fallback on devices
without fragment storage atomics. Depth-enabled raster splatting currently
uses the vertex path rather than mesh shaders.

Custom draw components must use `DrawComponent::pipelineDepthState()` when
creating their graphics pipeline to honor the render pass's depth options.
Color-only rendering needs no depth attachment or depth transfer passes.

### Profiling measurements

The lantern and two opaque boxes were measured at 512 × 512 pixels on an
NVIDIA GeForce RTX 2070 SUPER, Windows, Release build, validation disabled.
The scene contains 145,562 Gaussians, with a fixed camera and one execution path.
Each configuration ran three repetitions of 10 warm-up and 30 measured
executions; the table shows the median of the three mean GPU graph spans.

| Gaussian backend / pass order | Depth testing and writing off | Both enabled on both passes | Observed difference |
|---|---:|---:|---:|
| Raster / boxes first | 1.338 ms | 1.438 ms | +7.5% |
| Raster / Gaussians first | 1.336 ms | 1.446 ms | +8.2% |
| Compute / boxes first | 28.369 ms | 34.266 ms | +20.8% |
| Compute / Gaussians first | 28.763 ms | 33.918 ms | +17.9% |

These spans include both renderers, clears, barriers and depth transfers, with
per-element timestamp profiling enabled. Loading, graph compilation, CPU work,
image export and presentation are excluded. Even the off baseline retains the
shared attachment and its clear. The difference combines depth-handling work
with work saved through occlusion; it does not isolate the cost of depth alone.

The measurements use each backend's default depth estimate: nearest opaque
splat for raster, accumulated opacity for compute, both with a 0.5 threshold.
Across the full set of configurations, run means varied by up to 11.3% within
one configuration. Treat small differences as exploratory measurements.

See [Profiling](profiling.md) for interpreting graph and element timings.

## Operating on Gaussians

Gaussians can be changed on the GPU every frame before they are rendered:
{cpp:class}`klartraum::GaussianTransform` applies a transform from a
{cpp:class}`klartraum::TransformBuffer`, and
{cpp:class}`klartraum::GaussianMerge` combines two sets into one.

## Current limitations

- Gaussian depth is an approximate surface estimate for translucent scenes.
- The compute pipeline is not fully optimized yet.
