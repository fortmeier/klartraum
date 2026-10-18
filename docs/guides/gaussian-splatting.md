# Gaussian splatting

Klartraum renders 3D Gaussian splatting scenes stored in Niantic's
[`.spz` format](https://github.com/nianticlabs/spz).

## Loading a scene

{cpp:class}`klartraum::GaussianDataStandard` loads an `.spz` file (optionally
mirrored across the Y axis, as needed for Nerfstudio exports) and uploads the
Gaussians into GPU buffers. One instance can be shared by several backends or
viewports, so the scene is loaded only once.

## Backends

`createGaussianSplatting()` builds a splatting element with one of two
backends. Both have the same inputs, so switching the backend does not change
the surrounding graph.

| Backend | Class | How it works |
|---|---|---|
| `GsplatBackend::Compute` | {cpp:class}`klartraum::VulkanGaussianSplatting` | Sorts per screen tile and splats in compute shaders |
| `GsplatBackend::Raster` | `klartraum::VulkanGaussianSplattingRaster` | Sorts once and draws with the hardware rasterizer |

For a walkthrough of culling, compaction and equal-depth ordering, see
{doc}`../articles/deterministic-raster-order`.

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

With `GsplatConfig::outputDepth`, either backend also writes a per-pixel depth
buffer, read with `getGaussianSplattingDepth()`: the NDC depth of the scene's
surface, or +inf where there is none, so other geometry can be composited into
the scene instead of over it. The compute backend reports the median depth,
where the transmittance first falls to 0.5, and the raster backend the nearest
depth of a splat that alone covers the pixel with an opacity of at least 0.5.
Both agree on opaque surfaces. The colour image is the same with and without
depth output.

## Operating on Gaussians

Gaussians can be changed on the GPU every frame before they are rendered:
{cpp:class}`klartraum::GaussianTransform` applies a transform from a
{cpp:class}`klartraum::TransformBuffer`, and
{cpp:class}`klartraum::GaussianMerge` combines two sets into one.
{cpp:class}`klartraum::GaussianMorph` blends one set into another of the
same size along one of several paths (straight, swirl, scatter, wave,
explosions), driven by {cpp:struct}`klartraum::MorphParams` in a host-updated
buffer. {cpp:class}`klartraum::GaussianLighting` relights a scene with point
lights, using materials baked once on the CPU with `bakeSplatMaterials()`
(normal, roughness and a coarse visibility per Gaussian).

## Text as Gaussians

`rasterizeTitle()` lays out lines of text with FreeType, from any
{cpp:class}`klartraum::Font` (variable fonts at any weight), into a coverage
mask, and `sampleTextGaussians()` places Gaussians inside the letters, kept
slightly inside the outline so the soft splat edges do not embolden the font.
It can produce exactly as many Gaussians as a scene has, so that
`prepareTextMorph()` and a GaussianMorph turn the text into the scene, as
`title_morph_example` does. The build downloads a few OFL fonts into
`build/fonts/` (`-DKLARTRAUM_DOWNLOAD_FONTS=OFF` skips that).

## Current limitations

- Depth is written only on request (`GsplatConfig::outputDepth`); compositing
  other geometry with it is up to the application.
- The compute pipeline is not fully optimized yet.
