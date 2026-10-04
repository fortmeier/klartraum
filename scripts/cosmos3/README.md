# NVIDIA Cosmos 3 Edge image-to-video experiment

> **Warning:** NVIDIA Cosmos 3 is not part of Klartraum. These scripts download
> and process the third-party `nvidia/Cosmos3-Edge` model weights, which NVIDIA
> distributes under its own license (OpenMDW 1.1). Klartraum neither ships nor
> endorses these models. You are solely responsible for complying with the
> model license and for how you use the models and anything they generate.

This isolated uv project runs the diffusers reference pipeline
(`Cosmos3OmniPipeline`) of [Cosmos3-Edge](https://huggingface.co/nvidia/Cosmos3-Edge)
to turn `data/lantern.jpg` into a short video, and exports the model as
fixed-shape float32 ONNX graphs that `examples/cosmos3_example.cpp` runs on
Klartraum's Vulkan ONNX runtime, like `scripts/sd15_onnx` for Stable
Diffusion 1.5.

Cosmos3-Edge is a 4B Mixture-of-Transformers model: a 2B understanding tower
(text, causal attention) and a 2B generation tower (video latents, full
attention over both). Both share one 28-layer, Llama-style layout
(hidden 2048, 16 query / 8 KV heads, ReLU² MLP, 3D mRoPE). Video latents
come from the Wan2.2 VAE (48 channels, 16x spatial and 4x temporal
compression) and are patchified 2x2, so a 256x256 frame group becomes 64
tokens. Sampling uses UniPC with flow shift 12 and classifier-free guidance
(two transformer passes per step). The weights are bf16, about 9 GB.

From this directory:

```bash
uv sync
uv run python run_reference.py
```

The first run downloads the weights into the Hugging Face cache. If the
download stalls ("connection struggling" in `~/.cache/huggingface/xet/logs`),
fall back to plain HTTPS with
`HF_HUB_DISABLE_XET=1 uv run hf download nvidia/Cosmos3-Edge`.

Outputs are written to `build/TestingOutput/cosmos3/`: `<name>.mp4`, a
first/middle/last frame strip `<name>_strip.png`, and the exact prompt used,
`<name>_prompt.json`. The default prompt is "camera flight orbiting around the
lantern". Cosmos 3 was trained on structured JSON captions, so the script wraps a
plain-text `--prompt` into a minimal caption of that shape; pass `--raw-prompt` to
send it unchanged. The negative prompt is the checkpoint's
`assets/negative_prompt.json`.

Useful options: `--size W H` (multiples of 16), `--num-frames` (4k+1),
`--fps`, `--steps`, `--guidance-scale`, `--flow-shift`, `--seed`, `--name`.
`--num-frames 9 --steps 2 --name smoke` is a quick end-to-end check.

The Cosmos guardrail (`cosmos_guardrail`) is not installed. It brings
several additional large models, and the checkpoint's `model_index.json`
disables it by default.

## Results on a Mac mini M4 (24 GB)

PyTorch MPS, bf16, diffusers at the pinned commit:

| Run | Size | Frames | Steps | Time per step | Total |
|-----|------|--------|-------|---------------|-------|
| smoke | 256x256 | 9 | 2 | ~5 s | 17 s |
| default | 256x256 | 33 (16 fps) | 20 | ~5.2 s | 124 s |

Peak process memory footprint for the default run was about 11.6 GB.
The default run yields a coherent clip in which the camera moves steadily around
the lantern; the mean absolute difference from frame 0 grows from 15 to 33
(8-bit) over the clip.

## ONNX export for Klartraum

`export_onnx.py` runs in stages so that only one large model is resident at a
time. From this directory:

```bash
uv run python export_onnx.py prepare     # tokens, rotary tables, latents, schedule
uv run python export_onnx.py reference   # unmodified diffusers transformer, step 0
uv run python export_onnx.py text        # text_kv.onnx
uv run python export_onnx.py denoiser    # denoiser.onnx
uv run python export_onnx.py vae         # vae_encoder.onnx, vae_decoder.onnx
uv run python export_onnx.py pipeline    # float32 20-step reference and decode
```

Everything is written to `data/onnx/cosmos3_256/` (about 14 GB of float32
weights and 2 GB of fixtures). These are generated development artifacts and must not
be committed. `--size`, `--num-frames`, `--fps`, `--steps`, `--prompt`, and
`--seed` select another fixed configuration; pass the matching `--onnx-dir`.

The graphs:

- **text_kv.onnx** runs the text ("understanding") tower once for both
  guidance prompts and returns every layer's generation-side keys and values.
  The text stream never attends to video tokens, so these are independent of
  the denoising step. Both prompts are right-padded to one length (3520 tokens
  for the long negative prompt); padding cannot affect the causal text tower.
- **denoiser.onnx** runs the video ("generation") tower for one step and both
  guidance branches as a batch of two. Text padding is excluded by an additive
  key bias. Rotary tables are inputs because video positions depend on each
  prompt's length. Grouped-query attention is written as a reshape of the
  queries, so keys and values are never repeated.
- **vae_encoder.onnx** / **vae_decoder.onnx** are a fixed-shape, whole-clip
  rewrite of the chunked, feature-cached Wan VAE in diffusers. Causal temporal
  padding becomes asymmetric Conv pads, the latent normalization is folded into
  the quant convolutions, and the encoder handles the single conditioning frame
  (the VAE is temporally causal, so this equals the pipeline's full-clip encode).

Each stage checks its wrapper against diffusers and ONNX Runtime against
PyTorch; the measured errors are in the table below.

## Klartraum example

After exporting, build Klartraum and run from the repository root:

```bash
./build/examples/cosmos3_example
```

It encodes the conditioning frame, runs the text tower, denoises for 20 steps
(classifier-free guidance and the UniPC scheduler, `klartraum::UniPCFlowScheduler`,
on the host), and decodes the clip. Every stage is compared with the Python
fixtures, and the run fails if a stage leaves its tolerance. Frames are written
to `build/TestingOutput/cosmos3/` as `frame_###.ppm` and as
`lantern_orbit_klartraum.y4m`, which ffmpeg converts directly:

```bash
ffmpeg -i build/TestingOutput/cosmos3/lantern_orbit_klartraum.y4m -pix_fmt yuv420p lantern_orbit.mp4
```

`--stage text|step|vae` stops after the text tower or the first denoising
pass, or checks only the VAE decoder; `--max-steps N` shortens the loop,
`--profile` prints per-operation GPU times. `--image file.ppm` preprocesses the
conditioning frame in C++ (`klartraum::preprocessConditioningImage`, checked
against `image_f32.bin`), and `--name` sets the output video name.

Token ids, the initial noise, and the rotary tables still come from the Python
fixtures; a native Cosmos3 tokenizer and host-side mRoPE are not ported yet.

### Accuracy and timing (Mac mini M4, MoltenVK)

| Stage | Max error vs float32 PyTorch | Klartraum time |
|-------|------------------------------|----------------|
| VAE encoder (1 frame) | 3.4e-6 (latents up to 2.7) | 0.5 s |
| Text tower (28 layers, 2 x 3520 tokens) | 4.4e-5 relative | 54 s |
| Denoiser step (both guidance branches) | 9.3e-6 (velocity up to 5.7) | 9.5 s |
| 20 steps, final latents | 6.7e-4 (latents up to 4.6) | 190 s |
| VAE decoder (9 -> 33 frames) | mean 1.7e-5, max 9.5e-4 | 63 s |
| Total | | 317 s |

For comparison, the float32 PyTorch/MPS reference (`export_onnx.py pipeline`)
takes about 2 s per step and 31 s for the decode, and the bf16 diffusers
pipeline 124 s in total. Klartraum's float32 matrix product and attention
kernels are not yet tuned for this model.

## Dashcam example: Stable Diffusion 1.5 -> Cosmos3

`dashcam.sh` chains two models: SD1.5 generates a road scene from text, and
Cosmos3 animates it into a 3 s forward-driving clip at 512x512 (33 frames at
11 fps; the frame rate only enters through the caption and the host-side rotary
tables). The caption is `captions/dashcam.json`, a full structured caption
passed unchanged (`--raw-prompt` / `--prompt-file`) that describes the camera
motion and the scenery passing by.

```bash
scripts/cosmos3/dashcam.sh reference   # diffusers: sd15_text_to_image.py, then run_reference.py
scripts/cosmos3/dashcam.sh klartraum   # both models on Klartraum
```

At 256x256 the same image and caption give no forward motion: the clip either
stays still or, with a higher guidance scale, cuts to a different road. At
512x512 the lane markings stream under the car and the trees pass by.

The `klartraum` mode runs `sd15_denoiser_example` at 512x512 (needs
`data/onnx/sd15_denoiser_512`) and Cosmos3 from `data/onnx/cosmos3_512`. The
text tower does not depend on the video size and is shared with
`cosmos3_256`. A whole-clip float32 VAE decode at 512x512 does not fit in 24 GB,
so `prepare --decoder-tile 16 --decoder-stride 8` selects a tiled decode: the
256 decoder runs on 3x3 overlapping 16x16-latent tiles that are blended with
linear ramps over the 128-pixel overlaps (`klartraum::TileBlender`, mirrored by
`tiled_decode` in `export_onnx.py`). The decoder graph is shared with
`cosmos3_256` as well. The 512 denoiser and VAE encoder are exported once
(`reference`, `denoiser`, `vae`), then `cosmos3_example --image` runs on the SD1.5
output with every stage check active. Outputs go to `build/TestingOutput/dashcam/`.

Mac mini M4, MoltenVK:

| Stage | Klartraum | Check vs float32 PyTorch |
|-------|-----------|--------------------------|
| SD1.5 512, 30 DDIM steps | 88 s | (validated graph; no fixture for this prompt) |
| Cosmos3 conditioning image (C++ preprocessing) | | max error 0 |
| Cosmos3 text tower | 15 s | 8.9e-5 relative |
| Cosmos3 denoiser, 20 steps (2304 video tokens) | 238 s (11.9 s/step) | final latents 4.8e-3 (up to 5.8) |
| Cosmos3 tiled VAE decode, 2x2 tiles of 320x320 | 197 s | video mean 3.8e-5, max 9.7e-3 |
| Cosmos3 total | 459 s | |

With only the 256 decoder (3x3 tiles of 256x256) the decode takes 280 s and the
total 541 s. `dashcam.sh` uses the 320 decoder when `data/onnx/cosmos3_320`
exists (`export_onnx.py prepare` and `vae --skip-ort` with `--size 320`).

Kernel improvements are logged in `PERFORMANCE.md`; before them the same run
took 1536 s (44.6 s per denoising step, 578 s for the decode).

The diffusers references take 40 s (SD1.5, 30 steps) and 274 s (Cosmos3 512,
bf16); the float32 PyTorch/MPS wrappers take about 10 s per denoising step and
42 s per decoder tile.
