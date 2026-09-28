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

## Operating on Gaussians

Gaussians can be changed on the GPU every frame before they are rendered:
{cpp:class}`klartraum::GaussianTransform` applies a transform from a
{cpp:class}`klartraum::TransformBuffer`, and
{cpp:class}`klartraum::GaussianMerge` combines two sets into one.

## Current limitations

- Depth values are not written, so splats are not occluded by other geometry.
- The compute pipeline is not fully optimized yet.
