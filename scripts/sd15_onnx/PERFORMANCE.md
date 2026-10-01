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
| 11 | Device-local weights and transient tensors | 3.48118 s | 59.06% faster | 82.55% | Kept |

The device-local profiled run took **3.53021 s** wall time and **3467.05 ms**
of aggregate GPU operation time. Its largest operation families were fused
attention at 1425.17 ms, convolution at 1451.70 ms, and MatMul at 459.37 ms.
MatMul had previously consumed roughly 2.4 seconds by itself.

The final full run averaged **3.39325 s per UNet step**, compared with
**19.8933 s per step** before these cycles: an **82.94% reduction** or
**5.86x speedup**.

## VAE decoder improvement runs

| Cycle | Configuration | VAE decode | Change from preceding measurement | Change from baseline | Decision |
|---:|---|---:|---:|---:|---|
| 0 | Initial implementation | 83.1483 s | baseline | baseline | Baseline |
| 1 | Tiled 3x3 convolution, MatMul, and parallel normalization | 26.3624 s | 68.29% faster | 68.29% faster | Kept |
| 2 | Add tiled 1x1 convolution | 18.6949 s | 29.09% faster | 77.52% faster | Kept |
| 3 | Increase attention group from four to eight queries | 18.7705 s | 0.40% slower | 77.43% faster | Kept for its larger 30-step UNet benefit |
| 4 | Final exact-prompt validation run | 18.6588 s | run variation | 77.56% faster | Final measurement |
| 5 | Device-local weights and transient tensors | 4.01977 s | 78.46% faster | 95.17% faster | Kept |

## End-to-end comparison

| Implementation | CLIP | DDIM UNet (30 steps) | VAE decode | Total | Relative to Python |
|---|---:|---:|---:|---:|---:|
| Python/ONNX Runtime reference | 0.177 s | 160.777 s | 6.500 s | 167.454 s | 1.00x |
| Klartraum before shader cycles | 0.301 s | 596.800 s | 83.148 s | 680.249 s | 4.06x slower |
| Klartraum after shader cycles | 0.213 s | 251.683 s | 18.659 s | 270.555 s | 1.62x slower |
| Klartraum with device-local ONNX storage | 0.050 s | 101.797 s | 4.020 s | 105.868 s | 1.58x faster |

The retained shader cycles reduced Klartraum's original total time by
**60.23%** and made the full pipeline **2.51x faster**. Device-local storage
then reduced the retained shader result by a further **60.87%**. Overall,
Klartraum uses **84.44% less time than its initial implementation** (6.43x
speedup) and takes **63.22% of the Python reference time**. Numerical validation
remains within the reference tolerances:

| Validation value | Maximum absolute error |
|---|---:|
| CLIP text embedding | 4.86374e-05 |
| First UNet prediction | 7.86781e-06 |
| Final latent after 30 steps | 1.21355e-04 |
| Decoded image | 8.28959e-04 |

## Theoretical speed-of-light bounds

The roofline estimate uses the RTX 2070 SUPER's 2,560 CUDA cores at its
1.770 GHz boost clock, or **9.0624 FP32 TFLOP/s**, and its **448 GB/s** memory
bandwidth. These hardware inputs come from
[NVIDIA's RTX 2070 SUPER specifications](https://www.nvidia.com/ro-ro/geforce/graphics-cards/rtx-2070-super/).

`analyze_roofline.py` reads the fixed tensor shapes from the exported ONNX
graphs. It counts two FLOPs per multiply-accumulate in Conv, ConvTranspose,
MatMul, and Gemm. Minimum operator traffic assumes that every dispatched
operator reads each input and writes each output exactly once; fused attention
does not materialize the score or softmax tensors. The roofline floor is the
larger of `FLOPs / peak FP32 throughput` and `bytes / peak bandwidth`.

```powershell
cd scripts/sd15_onnx
.\.venv\Scripts\python.exe analyze_roofline.py
```

| Stage | Dominant FLOPs | Minimum operator traffic | Compute floor | Bandwidth floor | Roofline floor | Measured efficiency | Remaining theoretical headroom |
|---|---:|---:|---:|---:|---:|---:|---:|
| CLIP | 26.597 GFLOP | 1.047 GB | 2.935 ms | 2.338 ms | 2.935 ms | 5.83% | 17.17x |
| UNet average step | 1,606.547 GFLOP | 25.577 GB | 177.276 ms | 57.092 ms | 177.276 ms | 5.22% | 19.14x |
| UNet, 30 steps | 48,196.406 GFLOP | 767.322 GB | 5,318.283 ms | 1,712.772 ms | 5,318.283 ms | 5.22% | 19.14x |
| VAE decode | 2,514.519 GFLOP | 28.320 GB | 277.467 ms | 63.214 ms | 277.467 ms | 6.90% | 14.49x |
| Full pipeline | 50,737.522 GFLOP | 796.689 GB | 5,598.685 ms | 1,778.324 ms | 5,598.685 ms | 5.29% | 18.91x |

These are intentionally optimistic bounds, not expected execution times. They
assume simultaneous 100% boost-clock FP32 utilization and ideal peak memory
bandwidth, and omit transcendental cost, reductions, dispatches, barriers,
dependencies, cache conflicts, and layout overhead. Their useful purpose is to
show remaining headroom and whether an ideal stage is compute- or
bandwidth-limited. At present every stage is compute-bound by this model, which
supports prioritizing attention/convolution arithmetic efficiency and later
FP16 or cooperative-matrix paths.

## Apple M4 (MoltenVK)

### Benchmark configuration

- Date: 2026-09-29
- Machine: Mac mini (Mac16,10), Apple M4 with a 10-core GPU, 24 GB unified memory
- Vulkan driver: MoltenVK (Vulkan SDK 1.4.357.1), API 1.4.357
- Build: Release (`build/examples/Release/`, Ninja Multi-Config, validation layers off)
- Image size, scheduler, batch, prompt, and fixtures: as above
  (`data/onnx/sd15_denoiser_512`, 30 DDIM steps, batch 2)
- Klartraum memory plan: 336.875 MiB allocated for the UNet, identical to the
  RTX run

Before these cycles the GPU profile reported 0 ms for every UNet dispatch:
MoltenVK backs a timestamp query pool with one `MTLCounterSampleBuffer`,
limited to 4096 samples, and the UNet needs 10192. Timestamps are now split
across pools of at most 2048 elements (commit `19b7d9a`).

### UNet improvement runs

One profiled UNet step, `--profile --max-denoise-steps 1 --skip-decoder`.

| Cycle | Variant | One UNet step | GPU time | FusedAttention | Incremental change | Decision |
|---:|---|---:|---:|---:|---:|---|
| 0 | Starting point (RTX-tuned shaders) | 10.3467 s | 9196.46 ms | 6625.63 ms | baseline | Baseline |
| 1 | Key-tiled attention for head width 40 | 3.8572 s | 3094.06 ms | 514.06 ms | 62.72% faster | Kept (`a9da76f`) |
| 2 | Key-tiled attention for head width 80 | 3.5096 s | 2578.95 ms | 97.73 ms | 9.01% faster | Kept (`02b7392`) |

The starting-point attention kernel handles one key per iteration: it stages a
single key and value row, synchronizes the workgroup twice, and reduces a
40-wide dot product across a subgroup, all per key. The five 64x64-latent
self-attention dispatches (4096 keys) took 1.08-1.28 s each, about 35
GFLOP/s. The tiled kernel keeps one query row and its accumulators in
registers per invocation, stages 32 keys per barrier pair for 64 queries,
and rescales the online softmax once per 8 keys.

After cycle 2 the profiled step still spends about 930 ms outside GPU
operations (3509.6 ms wall versus 2579.0 ms GPU). Every graph element is
submitted as its own command buffer with semaphore waits and signals, which
MoltenVK maps to one `MTLCommandBuffer` and `MTLEvent` synchronization per
dispatch. The largest GPU families are now convolution at 1340.21 ms and
MatMul at 685.54 ms.

### End-to-end comparison

| Implementation | CLIP | DDIM UNet (30 steps) | VAE decode | Total |
|---|---:|---:|---:|---:|
| Python/PyTorch reference on the M4 CPU | 0.520 s | 78.682 s | 9.923 s | 89.125 s |
| Klartraum, starting point | 0.075 s | 306.866 s (10.229 s/step) | 5.483 s | 312.425 s |
| Klartraum after cycles 1-2 | 0.076 s | 103.995 s (3.467 s/step) | 5.580 s | 109.651 s |
| Klartraum on the RTX 2070 SUPER (above) | 0.050 s | 101.797 s (3.393 s/step) | 4.020 s | 105.868 s |

The two attention cycles reduced the M4 total by **64.90%** (2.85x speedup)
and the UNet step by **66.11%**. The M4 now runs the full pipeline within 4%
of the RTX 2070 SUPER, whose FP32 peak is about twice as high. The Python
reference on the M4 CPU remains faster (89.1 s).

| Validation value | Starting point | After cycles 1-2 |
|---|---:|---:|
| CLIP text embedding | 2.40326e-04 | 2.40326e-04 |
| First UNet prediction | 6.91414e-06 | 7.15256e-06 |
| Final latent after 30 steps | 7.72528e-04 | 2.67658e-04 |
| Decoded image | 9.05514e-04 | 4.79594e-04 |

### Theoretical speed-of-light bounds

Apple does not publish FP32 throughput for the M4 GPU. The estimate below
uses **4.26 TFLOP/s** (10 cores x 128 FP32 lanes x 2 FLOPs x about 1.66 GHz)
and the specified **120 GB/s** unified-memory bandwidth, with the same
operator model as the RTX table:

```bash
cd scripts/sd15_onnx
python analyze_roofline.py --fp32-tflops 4.26 --memory-bandwidth-gbps 120 \
  --clip-seconds 0.0757297 --unet-seconds 103.995 --vae-seconds 5.57983
```

| Stage | Compute floor | Bandwidth floor | Roofline floor | Efficiency before | Efficiency after cycles 1-2 | Remaining headroom |
|---|---:|---:|---:|---:|---:|---:|
| CLIP | 6.243 ms | 8.728 ms | 8.728 ms | 11.67% | 11.53% | 8.68x |
| UNet average step | 377.124 ms | 213.145 ms | 377.124 ms | 3.69% | 10.88% | 9.19x |
| VAE decode | 590.263 ms | 235.999 ms | 590.263 ms | 10.76% | 10.58% | 9.45x |
| Full pipeline | 11,910.217 ms | 6,639.078 ms | 11,910.217 ms | 3.81% | 10.86% | 9.21x |

### Single command buffer per graph path

SD1.5 used to hang on KosmicKrisp in the first UNet submission: the fence of
the batched `vkQueueSubmit` (one command buffer per graph element, chained by
about 5,000 binary semaphores) never signaled, while submitting the same
elements one at a time worked. Each graph path is now recorded into one
command buffer with a full memory barrier between consecutive elements, and
only external waits and the graph-finished semaphore remain.

The 64-query, 32-key tiled attention kernel for head width 80 also exceeded
the 32 KiB threadgroup-memory limit of Apple GPUs (Metal reports 40 KiB:
20 KiB of key/value tiles plus the per-invocation query and accumulator
arrays). It now stages 16 keys per tile, which measured no slower on either
driver.

| Driver | DDIM UNet (30 steps) | VAE decode | Total | Final latent error | Decoded image error |
|---|---:|---:|---:|---:|---:|
| MoltenVK (SDK 1.4.357.1) | 101.657 s (3.389 s/step) | 5.450 s | 107.179 s | 2.67658e-04 | 4.79594e-04 |
| KosmicKrisp (API 1.4.359) | 123.941 s (4.131 s/step) | 6.877 s | 130.902 s | 2.67658e-04 | 4.79594e-04 |

On MoltenVK, the step time is unchanged from cycle 2 (3.38 s), so its earlier
per-dispatch overhead was not the semaphores: the main thread spends the step
in `vkWaitForFences`, and the per-dispatch timestamps under-report GPU time.
Emitting barriers only before elements with predecessors (3,237 of the 5,096
UNet elements have none) changed neither driver's step time. KosmicKrisp is
about 22% slower per UNet step on the same kernels. Its timestamp queries
return zero, so `--profile` reports no per-operation GPU time on it.

Select the driver with the loader, for example:

```bash
VK_DRIVER_FILES=/usr/local/share/vulkan/icd.d/libkosmickrisp_icd.json \
  ./build/examples/sd15_denoiser_example --model-dir data/onnx/sd15_denoiser_512 --size 512
```

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
