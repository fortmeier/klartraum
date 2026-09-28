# Stable Diffusion 1.5 VAE ONNX experiment

This isolated uv project downloads the Stable Diffusion 1.5 VAE, runs its
encoder and decoder on `data/lantern.jpg`, and exports the two halves for the
Klartraum ONNX runtime. It deliberately pins ONNX 1.18.0, matching
`scripts/onnx`.

From this directory:

```powershell
uv sync
uv run python run_reference.py
uv run python export_onnx.py
uv run python export_denoiser.py
```

Reference artifacts are written below `build/TestingOutput/sd15_onnx/`.
The exports are written to `data/onnx/sd15/`. Both are fixed-shape float32
models. The checked-in smoke model maps `[1,3,64,64]` to `[1,4,8,8]` and back;
this keeps routine tests fast. Pass the same `--size` to both commands to export
a different fully-convolutional image size.

Klartraum derives tensor lifetimes from the ONNX producer/consumer graph,
reuses physical Vulkan buffers for non-overlapping tensors, and aliases eligible
`Reshape` outputs as zero-copy views. For canonical 512x512 validation, keep the
large generated files separate from the smoke fixtures:

```powershell
uv run python export_onnx.py --size 512 `
  --onnx-dir ../../data/onnx/sd15_512 `
  --output-dir ../../build/TestingOutput/sd15_onnx_512
```

After building Klartraum, run the headless Vulkan example from the repository
root:

```powershell
.\build\examples\Debug\sd15_vae_example.exe

.\build\examples\Debug\sd15_vae_example.exe --size 512 `
  --model-dir .\data\onnx\sd15_512 `
  --output build\TestingOutput\sd15_vae_klartraum_512.ppm
```

It reads the normalized `lantern_input_f32.bin` generated beside the ONNX
models and writes `build/TestingOutput/sd15_vae_klartraum.ppm`. Encoder and
decoder execute sequentially, so their transient arenas are not resident at the
same time. Use `--load-only` to parse both graphs without allocating or running
their Vulkan buffers.

`export_denoiser.py` exports a fixed 256x256, batch-two UNet and matching VAE
decoder. Its default reference pipeline runs eight deterministic DDIM steps
with classifier-free guidance and seed 12. The prompt produces a recognizable
traditional Japanese granite garden lantern rather than merely exercising the
network numerically. It writes prompt
embeddings, scheduler coefficients, latent/image references, and an operator
coverage report. These generated model and tensor files remain development
artifacts and must not be committed.

After changing only the prompt, seed, or step count, reuse an existing export
of the same fixed size and regenerate just the fixtures with
`uv run python export_denoiser.py --reuse-models`.

Run the matching Vulkan pipeline with:

```powershell
.\build\examples\Debug\sd15_denoiser_example.exe
```

The example executes the complete SD1.5 UNet once per generated scheduler step
with a batch of two (negative and positive prompt conditioning), applies guidance
scale 7.5 and the DDIM update on the CPU, and feeds the final latent into the
Klartraum VAE decoder.
It writes `build/TestingOutput/sd15_pipeline_klartraum.ppm` and checks the final
latent and decoded pixels against the generated Python/ONNX Runtime reference.
The current experiment deliberately stays at 256x256. The example also accepts
`--size` and `--model-dir` for separately generated fixed-size exports. Prompt
tokenization and CLIP text encoding are still performed by the export script;
moving those into the native runtime is the remaining step for accepting
arbitrary prompt text.
