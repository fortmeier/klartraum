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
```

Reference artifacts are written below `build/TestingOutput/sd15_onnx/`.
The exports are written to `data/onnx/sd15/`. Both are fixed-shape float32
models. The checked-in smoke model maps `[1,3,64,64]` to `[1,4,8,8]` and back;
this keeps Klartraum's current retain-all-intermediates allocator below 8 GB.
Pass the same `--size` to both commands (for example, `--size 512`) to export a
different fully-convolutional image size. A canonical 512x512 model loads in
Klartraum, but needs a future liveness-aware allocator to execute on an 8 GB GPU.

After building Klartraum, run the headless Vulkan example from the repository
root:

```powershell
.\build\examples\Debug\sd15_vae_example.exe
```

It reads the normalized `lantern_input_f32.bin` generated beside the ONNX
models and writes `build/TestingOutput/sd15_vae_klartraum.ppm`. Use
`--load-only` to parse and connect both graphs without allocating or
executing all intermediate tensors.
