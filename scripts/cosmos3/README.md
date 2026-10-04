# NVIDIA Cosmos 3 Edge image-to-video experiment

> **Warning:** NVIDIA Cosmos 3 is not part of Klartraum. These scripts download
> and process the third-party `nvidia/Cosmos3-Edge` model weights, which NVIDIA
> distributes under its own license (OpenMDW 1.1). Klartraum neither ships nor
> endorses these models. You are solely responsible for complying with the
> model license and for how you use the models and anything they generate.

This isolated uv project runs the diffusers reference pipeline
(`Cosmos3OmniPipeline`) of [Cosmos3-Edge](https://huggingface.co/nvidia/Cosmos3-Edge)
to turn `data/lantern.jpg` into a short video. It is the Python baseline for a
later Klartraum port, like `scripts/sd15_onnx` for Stable Diffusion 1.5.

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
