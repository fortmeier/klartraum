# SD1.5 Performance Log

This file tracks Stable Diffusion 1.5 inference performance as the Klartraum
ONNX shaders and compute runtime are improved. Add a row for every benchmarked
variant, including rejected experiments, so later work does not repeat them.

## Benchmark configuration

- Date: 2026-09-29
- GPU: NVIDIA GeForce RTX 2070 SUPER
- Vulkan API reported by the driver: 1.4.347
- Build: Release
- Image size: 512x512 (64x64 latent)
- Scheduler: DDIM, 30 steps
- Batch: 2 UNet samples for classifier-free guidance
- Klartraum memory plan: 336.875 MiB allocated for the UNet
- Fixture directory: `data/onnx/sd15_denoiser_512`

The one-step runs use `--profile --max-denoise-steps 1 --skip-decoder`.
Compare those results only with other profiled one-step runs. Full-pipeline
measurements use the exported prompt and negative prompt and run without GPU
profiling. Small variations between repeated runs are expected.

## UNet shader improvement runs

The incremental change is measured against the preceding retained variant.
The cumulative reduction is relative to the initial profiled implementation.

| Cycle | Variant | One UNet step | Incremental change | Cumulative reduction | Decision |
|---:|---|---:|---:|---:|---|
| 0 | Initial fused-attention implementation | 19.9447 s | baseline | baseline | Baseline |
| 1 | 16x16 shared-memory MatMul tiles | 17.0148 s | 14.69% faster | 14.69% | Kept |
| 2 | Shared-input 3x3 convolution tiles | 14.1573 s | 16.79% faster | 29.02% | Kept |
| 3 | Four attention queries sharing key/value rows | 11.4307 s | 19.26% faster | 42.69% | Kept |
| 4 | Four convolution output channels sharing an input tile | 10.6379 s | 6.94% faster | 46.66% | Kept |
| 5 | Parallel instance-normalization reductions | 9.50579 s | 10.64% faster | 52.34% | Kept |
| 6 | Eight convolution output channels per workgroup | 9.55958 s | 0.57% slower | 52.07% | Rejected; restored four channels |
| 7 | 32x16 MatMul output tiles | 9.49427 s | 0.12% faster | 52.40% | Kept |
| 8 | Tiled 1x1 convolution plus eight-query attention groups | 8.73691 s | 7.98% faster | 56.20% | Kept |
| 9 | Explicitly stage 3x3 weights in shared memory | 8.89547 s | 1.81% slower | 55.40% | Rejected; hardware cache was faster |
| 10 | Final exact-prompt validation run | 8.50328 s | informational | 57.37% | Kept configuration |

The final full run averaged **8.38943 s per UNet step**, compared with
**19.8933 s per step** before these cycles: a **57.83% reduction** or
**2.37x speedup**.

## VAE decoder improvement runs

| Cycle | Configuration | VAE decode | Change from preceding measurement | Change from baseline | Decision |
|---:|---|---:|---:|---:|---|
| 0 | Initial implementation | 83.1483 s | baseline | baseline | Baseline |
| 1 | Tiled 3x3 convolution, MatMul, and parallel normalization | 26.3624 s | 68.29% faster | 68.29% faster | Kept |
| 2 | Add tiled 1x1 convolution | 18.6949 s | 29.09% faster | 77.52% faster | Kept |
| 3 | Increase attention group from four to eight queries | 18.7705 s | 0.40% slower | 77.43% faster | Kept for its larger 30-step UNet benefit |
| 4 | Final exact-prompt validation run | 18.6588 s | run variation | 77.56% faster | Final measurement |

## End-to-end comparison

| Implementation | CLIP | DDIM UNet (30 steps) | VAE decode | Total | Relative to Python |
|---|---:|---:|---:|---:|---:|
| Python/ONNX Runtime reference | 0.177 s | 160.777 s | 6.500 s | 167.454 s | 1.00x |
| Klartraum before shader cycles | 0.301 s | 596.800 s | 83.148 s | 680.249 s | 4.06x slower |
| Klartraum after shader cycles | 0.213 s | 251.683 s | 18.659 s | 270.555 s | 1.62x slower |

The retained shader cycles reduced Klartraum's total time by **60.23%** and
made the full pipeline **2.51x faster**. Numerical validation remained within
the reference tolerances:

| Validation value | Maximum absolute error |
|---|---:|
| CLIP text embedding | 4.86374e-05 |
| First UNet prediction | 7.86781e-06 |
| Final latent after 30 steps | 1.21355e-04 |
| Decoded image | 8.28959e-04 |

## Adding another cycle

For each new optimization:

1. Run the same profiled one-step command at least once before and after the
   change.
2. Record the exact wall time and relevant aggregate GPU operation time.
3. Run the exact exported prompt to check the first UNet prediction error.
4. Mark slower variants as rejected instead of removing their rows.
5. After retaining a set of changes, run all 30 steps and record the final
   latent error, decoded image error, and stage timings.

Useful quick benchmark:

```powershell
.\build\examples\Release\sd15_denoiser_example.exe `
  --model-dir .\data\onnx\sd15_denoiser_512 `
  --size 512 `
  --max-denoise-steps 1 `
  --skip-decoder `
  --profile
```
