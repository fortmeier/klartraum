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

## End-to-end, 256x256

| Variant | VAE encode | Text tower | Denoising (20 steps) | VAE decode | Total | Final latents | Video (mean) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Before (cycle 0) | 0.5 s | 54 s | 190 s (9.5 s/step) | 63 s | 317 s | 6.7e-4 | 1.7e-5 |
| MatMul cycle 4 | 0.5 s | 19.5 s | 70.6 s (3.53 s/step) | 63 s | 163 s | 6.7e-4 | 1.7e-5 |
| + Conv3d cycle 2 | 0.3 s | 19.4 s | 70.6 s (3.53 s/step) | 30.6 s | 130 s | 6.7e-4 | 1.7e-5 |

For comparison: the float32 PyTorch/MPS wrappers take about 2 s per step and
31 s for the decode; the diffusers bf16 pipeline 124 s in total.

SD1.5 512 (30 DDIM steps) with MatMul cycle 4: 2.78 s per UNet step (from 3.39 s),
89.1 s in total, with unchanged errors (final latent 2.68e-4, decoded image 4.80e-4).
