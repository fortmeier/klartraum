# SD1.5 Vulkan Kernel Analysis

This analysis records the bottlenecks observed in the 512x512 SD1.5 profile
before selecting another optimization. It is intentionally hypothesis-driven:
each proposal states why it may help, the likely risk, and how it should be
validated.

## Profile summary

The retained 512x512 one-step profile is approximately:

| Kernel family | UNet time per step | VAE decode time | Main concern |
|---|---:|---:|---|
| Convolution | 2.9-3.0 s | 15.5 s | NCHW traffic and repeated channel/tile loads |
| MatMul | 2.4 s | small | Occupancy and lack of tensor-core/FP16 use |
| Fused attention | 1.85 s | 1.77 s | Barrier and exponential work for every key row |
| Layer normalization | 0.56 s | small | Each row is reduced serially by one invocation |
| Elementwise Add/Mul | 0.52 s | 0.76 s | Many full-buffer passes and dispatches |
| Instance normalization | 0.10 s | 0.50 s | Already parallel, but still makes several passes |

The times above overlap slightly with wall-clock accounting and should be used
for prioritization, not as an exact sum.

## Cross-kernel memory placement

All `VulkanBuffer` allocations currently request the first
`HOST_VISIBLE | HOST_COHERENT` memory type. On the benchmark RTX 2070 SUPER,
that selects system memory rather than the 7.8 GiB device-local heap. As a
result, weights and intermediate tensors for every operation can be fetched
over PCIe.

Hypothesis: place ONNX weights and planned transient tensors in
`DEVICE_LOCAL` memory, and use host-visible staging buffers only for model
upload and result readback. This should improve every bandwidth-sensitive
kernel at once. The memory planner keeps the largest live UNet transient set
near 337 MiB; its approximately 3.4 GiB of weights also fit in the available
device heap. The VAE runs as a separate network and is smaller.

Risk: model setup becomes slower if every initializer uses its own synchronous
staging transfer. That is acceptable for the first inference benchmark; a
batched upload arena is a follow-up startup optimization. Correctness is
checked with a device-local upload/readback test and the SD reference fixtures.

## Per-kernel hypotheses

### 3x3 convolution

The retained shader shares a 10x10 input tile across four output channels, but
it still synchronizes for every input channel and reads weights separately for
each output. The VAE profile makes this the largest individual target.

Promising changes, in order:

1. Device-local inputs, weights, and outputs. The current system-memory
   placement can dominate all shader-level tuning.
2. Winograd F(2x2, 3x3), reducing multiplications for the overwhelmingly common
   stride-one 3x3 case at the cost of transform kernels and extra workspace.
3. FP16 storage/arithmetic or cooperative matrices where supported. This can
   halve traffic and access tensor-core throughput, but needs numerical and
   feature fallbacks.
4. Register-block several output pixels/channels. The earlier eight-channel
   workgroup experiment regressed, so simply increasing shared output channels
   is not sufficient.

### 1x1 convolution

This is already expressed as a tiled matrix product. Device-local storage is
the immediate opportunity. Later work should compare smaller workgroups for
occupancy, register-blocked outputs, and a shared MatMul implementation so the
same FP16/cooperative-matrix path serves both operators.

### MatMul

The 32x16 tile uses a 512-thread workgroup, which can constrain occupancy and
scheduling. Likely improvements are device-local storage first, then an
autotuned 16x16/32x8 family, per-thread register blocking, vectorized aligned
loads, and finally FP16/cooperative matrices. The tile alternatives must be
measured by matrix shape; a single shape-independent winner is unlikely.

### Fused attention

The current online softmax avoids materializing the score matrix and shares
key/value rows across eight query subgroups. However, it performs a workgroup
barrier and exponential update for every key and keeps 16 accumulators per
invocation, which can spill registers.

After device-local storage, process blocks of keys per synchronization step,
perform a blockwise online-softmax merge, and specialize common head sizes.
Profiling register occupancy should determine whether fewer simultaneous query
rows beat the saved key/value loads. FP16 key/value storage is another likely
win once accuracy is characterized.

### Layer normalization

This shader assigns one invocation to a row and scans the normalized axis
serially for mean, variance, and output. It leaves most of a 64-thread
workgroup idle for the long CLIP/UNet rows. Rework it like InstanceNorm: one
workgroup per row, parallel sum and squared-sum reductions, then a strided
parallel output pass. Fuse affine normalization into adjacent operations where
the ONNX graph permits it.

### Instance normalization

The retained shader already uses workgroup reductions. Remaining opportunities
are vectorized loads, combined sum/squared-sum accumulation, and fusing the
following scale/bias or activation. Its lower measured cost puts it behind
LayerNorm and convolution.

### Elementwise, transpose, slice, and resize

Add, Mul, activation, and scheduler-style expressions repeatedly stream whole
tensors and incur dispatch barriers. Graph-level expression fusion is more
promising than tuning each simple shader. Metadata-only reshape/cast/slice
aliases should remain zero-copy; physical transpose and resize are lower
priority until profiles show otherwise.

## Ranked implementation plan

| Rank | Change | Expected scope | Validation |
|---:|---|---|---|
| 1 | Device-local ONNX tensor and weight storage | All kernels | Buffer roundtrip, one-step profile, fixture errors |
| 2 | Workgroup-parallel LayerNorm | CLIP and UNet | Operator tests and aggregate LayerNorm time |
| 3 | Blockwise fused attention | UNet and VAE | Attention unit tests, register/occupancy profile |
| 4 | Winograd or FP16 convolution path | Mostly VAE, also UNet | Shape suite, image/latent error, full timing |
| 5 | Graph elementwise fusion | Whole graph | Graph tests, dispatch count and wall time |

The first implementation cycle therefore changes memory placement. It is a
runtime-level optimization derived from kernel behavior, rather than another
local shader edit, because it attacks the common bottleneck beneath every
kernel in the profile.

## Device-local implementation result

The hypothesis was confirmed on the same RTX 2070 SUPER:

| Measurement | Before | Device-local | Reduction |
|---|---:|---:|---:|
| Exact-prompt UNet step | 8.50328 s | 3.48118 s | 59.06% |
| Full 30-step UNet | 251.683 s | 101.797 s | 59.55% |
| VAE decode | 18.6588 s | 4.01977 s | 78.46% |
| Full pipeline | 270.555 s | 105.868 s | 60.87% |

The first UNet prediction error remained 7.86781e-06, final latent error
remained 1.21355e-04, and decoded-image error remained 8.28959e-04.

The new one-step GPU profile is led by convolution (1451.70 ms), fused
attention (1425.17 ms), and MatMul (459.37 ms). LayerNorm fell to 17.96 ms in
the UNet profile once its data was device-local, so its serial implementation
is no longer the second priority for end-to-end SD. The measured next order is:

1. blockwise fused attention, especially the five 64x64 self-attention nodes;
2. convolution algorithm/storage improvements, with the VAE as the main
   beneficiary;
3. MatMul register blocking and FP16/cooperative-matrix paths;
4. graph-level elementwise fusion;
5. parallel LayerNorm, primarily for CLIP and for broader model coverage.
