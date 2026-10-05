# Cosmos3 Performance Log

Cosmos3-Edge image-to-video on Klartraum, tracked as the Vulkan kernels are
improved. Every benchmarked variant is listed, including rejected ones, so
later work does not repeat them (see also `scripts/sd15_onnx/PERFORMANCE.md`).

## Benchmark configuration

- Machine: Mac mini (Mac16,10), Apple M4 with a 10-core GPU, 24 GB unified memory
- Vulkan driver: MoltenVK (Vulkan SDK 1.4.357.1); no cooperative-matrix extension
- Model directory: `data/onnx/cosmos3_256` (256x256, 33 frames, 20 UniPC steps,
  batch of two guidance branches, 576 video and 3520 text tokens)
- Profiled step: `./build/examples/cosmos3_example --stage step --max-steps 1 --profile`
- MatMul micro-benchmark: `klartraum_tests --gtest_filter=LayersTest.matMulCosmos3Projection`
  ([1152, 2048] x [2048, 2048], 20 timed repeats)

One denoiser step is about 4.6 TFLOP (2 FLOP x 2e9 parameters x 1152 tokens),
the text tower about 28 TFLOP; the M4's FP32 peak is about 4.3 TFLOP/s.

## MatMul kernel

| Cycle | Variant | Micro-benchmark | Denoiser step (wall) | Decision |
|---:|---|---:|---:|---|
| 0 | 32x16 tile, one output per invocation, 16-step staging | 494 GFLOP/s | 9.59 s (GPU 6950 ms, MatMul 5002 ms) | Baseline |
| 1 | 64x64 tile, 4x4 outputs per invocation, vec4 shared reads, 16 steps | ~1455 GFLOP/s | | Superseded by 3 |
| 2 | 128x128 tile, 8x8 outputs per invocation | 850 GFLOP/s | | Rejected (register pressure) |
| 3 | 64x64 tile, 4x4 outputs, 32-step staging | ~1495 GFLOP/s | 3.65 s (GPU 3102 ms, MatMul 2385 ms) | Superseded by 4 |
| 4 | 64x128 tile, 4x8 outputs, 32-step staging | ~1505 GFLOP/s | 3.62 s (GPU 3033 ms, MatMul 2275 ms) | Kept |
| 5 | Cycle 4 plus vec4 global loads on aligned in-bounds tiles | ~1495 GFLOP/s | | Rejected (no gain, more code) |
| 6 | Cycle 4 with 16-step staging (12 KiB shared, two workgroups per core) | ~1530 GFLOP/s | | Rejected (within noise) |
| 7 | 64x128 tile with 128 invocations, 8x8 outputs each | 970 GFLOP/s | | Rejected (register pressure) |
| 8 | Cycle 4 with bank-conflict-free left-tile stores | ~1515 GFLOP/s | | Rejected (no gain) |

The global loads are not the limit (cycle 5); the inner product loop is. Without
cooperative-matrix (simdgroup) access through MoltenVK, larger register blocks
lose to register pressure (cycle 2).

## Conv3d kernel (Wan VAE)

The 256x256 decoder performs about 40 TFLOP of convolution (largest: 512 and 256
output channels with 3x3x3 kernels at 64x64 and 128x128).
Measured with `cosmos3_example --stage vae --profile`.

| Cycle | Variant | VAE decode (wall) | GPU Conv | Decision |
|---:|---|---:|---:|---|
| 0 | 64x64 implicit-GEMM tile, 4x4 per invocation, per-element tap/position division | 62.9 s | 39.8 s | Baseline |
| 1 | Positions decomposed once per invocation, shared tap table, vec4 shared reads | 35.3 s | 28.8 s | Superseded by 2 |
| 2 | Cycle 1 with 64x128 tiles, 4x8 per invocation | 30.6 s | 28.3 s (~1.4 TFLOP/s) | Kept |

## Attention with bias (text padding, causal mask)

Cosmos3 attention is MatMul -> Add(bias) -> Softmax -> MatMul with 128-wide
heads, which the unbiased SD1.5 fusion does not match, so the score tensor was
materialized: [2, 8, 1152, 4096] (302 MB) per denoiser layer at 256x256,
[2, 8, 4608, 5824] (1.7 GB) at 512x512, and [2, 8, 7040, 3520] in the text tower.
`FusedAttentionBias` replaces the four nodes. Times below with the MatMul cycle 4
kernel; the layer benchmark is `LayersTest.fusedAttentionCosmos3Denoiser`
(one 256x256 denoiser layer).

| Cycle | Variant | Layer benchmark | Denoiser step | Text tower | Decision |
|---:|---|---:|---:|---:|---|
| 0 | Unfused MatMul, Add, Softmax, MatMul | 45-60 ms | 3.55 s | 19.5 s | Baseline |
| 1 | One query per four lanes (32 components each), 16-key tiles, shuffle-summed scores | | ~5.9 s (measured 6.17 s vs 3.53 s) | 34.0 s | Rejected |
| 2 | FlashAttention-style: 64x64 score tiles register-blocked like MatMul, online softmax with row shuffles, P x V from shared memory | | | | Superseded by 3 |
| 3 | Cycle 2, skipping key tiles whose bias masks every pair (exactly zero weight) | 15.6 ms | 2.70 s | 14.8 s | Kept |

At 512x512 a denoiser step went from 23.4 s (unfused) to about 17 s with cycle 2.

## Fusions of memory-bound chains

| Variant | Denoiser step 256 | Denoiser step 512 | Text tower | Decision |
|---|---:|---:|---:|---|
| Before | 2.70 s | 11.8 s | 14.8 s | Baseline |
| MatMul epilogues: ReLU² (MatMul -> Relu -> Mul) and residual Add | 2.64 s | | 14.6 s | Kept |
| + last-axis RMS normalization in one pass (6 nodes -> 1) | 2.57 s | 11.2 s | 14.2 s | Kept |

Also measured and rejected for the MatMul kernel: register prefetch of the next
tile (~1440 GFLOP/s), double-buffered shared tiles (~1470), explicit fma() (~1510),
128x128 tiles with 512 invocations (~1290), no shared memory at all (~1250), and
Conv3d with 32-step staging (31.6 s vs 30.6 s decode). A pure-register FMA probe
reaches 2.24 TFLOP/s FP32 (2.62 TFLOP/s FP16) on this M4 through MoltenVK, so
the MatMul kernel runs at about 67% of the measured peak; KosmicKrisp runs it at
656 GFLOP/s.

## End-to-end, 256x256

| Variant | VAE encode | Text tower | Denoising (20 steps) | VAE decode | Total | Final latents | Video (mean) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Before (cycle 0) | 0.5 s | 54 s | 190 s (9.5 s/step) | 63 s | 317 s | 6.7e-4 | 1.7e-5 |
| MatMul cycle 4 | 0.5 s | 19.5 s | 70.6 s (3.53 s/step) | 63 s | 163 s | 6.7e-4 | 1.7e-5 |
| + Conv3d cycle 2 | 0.3 s | 19.4 s | 70.6 s (3.53 s/step) | 30.6 s | 130 s | 6.7e-4 | 1.7e-5 |
| + Attention cycle 3 | 0.3 s | 14.8 s | 53.9 s (2.70 s/step) | 30.6 s | 108 s | 1.0e-3 | 1.9e-5 |
| + fusions | 0.3 s | 14.2 s | 51.2 s (2.56 s/step) | 30.5 s | 105 s | 4.1e-4 | 7.7e-6 |

For comparison: the float32 PyTorch/MPS wrappers take about 2 s per step and
31 s for the decode; the diffusers bf16 pipeline 124 s in total.

SD1.5 512 (30 DDIM steps) with MatMul cycle 4: 2.78 s per UNet step (from 3.39 s),
89.1 s in total, with unchanged errors (final latent 2.68e-4, decoded image 4.80e-4).

## End-to-end, 512x512 (dashcam, `data/onnx/cosmos3_512`, 2304 video tokens)

| Variant | Text tower | Denoising (20 steps) | Tiled VAE decode | Total | Final latents | Video (mean) |
|---|---:|---:|---:|---:|---:|---:|
| Before | 54 s | 893 s (44.6 s/step) | 578 s | 1536 s | 9.0e-3 | 4.6e-5 |
| MatMul 4 + Conv3d 2 + Attention 3 | 15 s | 238 s (11.9 s/step) | 280 s | 541 s | 4.8e-3 | 3.8e-5 |
| + 2x2 tiles of a 320 decoder (stride 12, 128-pixel overlap) | 15 s | 239 s (12.0 s/step) | 197 s | 459 s | 4.8e-3 | 3.8e-5 |
| + chunked decode (one latent frame at a time, exact) | 15 s | 236 s (11.8 s/step) | 129 s | 388 s | 4.8e-3 | 3.8e-5 |

The diffusers bf16 pipeline takes 274 s at 512x512 (its VAE decodes in cached
temporal chunks); the float32 PyTorch/MPS wrappers about 10 s per step and 42 s
per decoder tile.

At 256x256 the chunked decoder takes 33.9 s against 30.6 s for the whole-clip
graph, with 1.1 GB instead of 4.2 GB of transient storage, so the 256 export
keeps the whole-clip decoder.
