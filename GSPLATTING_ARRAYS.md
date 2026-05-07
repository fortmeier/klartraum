# Gaussian Splatting: Structure of Arrays (SoA) Optimization

## Motivation

Current AoS layout for Gaussian3D (236-byte struct) causes poor cache utilization.
A warp of 32 threads accessing only positions spans 32×236 = 7552 bytes (~59 cache lines, ~5% efficiency).
With SoA, positions are contiguous: 32×12 = 384 bytes (3 cache lines, 100% efficiency).

Profiling baseline (932K gaussians, 10 frames):
- GaussianProjection: 218 ms  ← primary target
- RadixSort: 885 ms
- GaussianBinningPrefixSum: 104 ms
- GaussianSplatting: 67 ms
- GatherSorted: 44 ms
- GaussianBinningScatter: 24 ms

## SoA Buffer Layout

### Gaussian3D SoA (input to projection)
- `pos_x`, `pos_y`, `pos_z`              — float[N]
- `rot_x`, `rot_y`, `rot_z`, `rot_w`    — float[N] each
- `scale_x`, `scale_y`, `scale_z`       — float[N] each
- `color_r`, `color_g`, `color_b`       — float[N] each
- `alpha`                                — float[N]
- `sh_r[0..14]`, `sh_g[0..14]`, `sh_b[0..14]` — float[N] each (15 bands each)

### Gaussian2D SoA (output of projection, input to downstream)
- `pos2d_x`, `pos2d_y`   — float[N]
- `z`                    — float[N]
- `cov_inv_a`, `cov_inv_b`, `cov_inv_c` — float[N] each
- `color2d_r`, `color2d_g`, `color2d_b`, `alpha2d` — float[N] each
- `bin_mask`             — uint[N]

## Implementation Steps

### Step 1 — Projection SoA shader
- New shader: `shaders/gsplat/gsplat_projection_soa.comp`
- Reads: Gaussian3D SoA buffers + CameraUBO
- Writes: Gaussian2D SoA buffers
- Test: `projection_soa` in `test_gaussian_splatting_arrays.cpp`
  - Run AoS and SoA back-to-back with profiling, assert same output, report speedup

### Step 2 — Binning Count SoA shader
- New shader: `shaders/gsplat/gsplat_binning_count_soa.comp`
- Reads `bin_mask` SoA array only (already a scalar per element → trivial gain)
- Writes histogram buffer
- Test: `binningCount_soa`

### Step 3 — Extract Sort Keys SoA shader
- New shader: `shaders/gsplat/gsplat_extract_sort_keys_soa.comp`
- Reads `bin_mask[]` and `z[]` SoA arrays separately (both now contiguous)
- Test: `extractSortKeys_soa`

### Step 4 — Gather Sorted SoA shader
- New shader: `shaders/gsplat/gsplat_gather_sorted_soa.comp`
- Reads Gaussian2D SoA + sorted index buffer, writes sorted Gaussian2D SoA
- Test: `gatherSorted_soa`

### Step 5 — Bin Bounds SoA shader
- New shader: `shaders/gsplat/gsplat_bin_bounds_soa.comp`
- Reads `bin_mask[]` SoA from sorted output
- Test: `binBounds_soa`

### Step 6 — Binned Splatting SoA shader
- New shader: `shaders/gsplat/gsplat_binned_splatting_soa.comp`
- Reads all Gaussian2D SoA fields needed for rendering
- Test: `binnedSplatting_soa`

### Step 7 — Full pipeline SoA test
- Test: `fullPipeline_soa` — runs complete SoA pipeline, verifies output image,
  compares profiling times vs AoS baseline, reports speedup per stage

## Test File Structure

`tests/test_gaussian_splatting_arrays.cpp`
- Reuses the same `HeadlessFrontend` fixture pattern
- Each test runs AoS version first (for baseline timing), then SoA version
- All tests must pass before moving to the next step
- Final test prints a speedup table

## Success Criteria

- All SoA tests produce identical output to their AoS counterparts
- Projection stage shows measurable speedup (target: >2x)
- No regressions in existing `test_gaussian_splatting.cpp` tests
