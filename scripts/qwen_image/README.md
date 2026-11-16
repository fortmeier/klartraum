# Qwen-Image

> **Warning:** Qwen-Image is not part of Klartraum. These scripts download and
> process third-party model weights: [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image)
> and [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
> by Alibaba Cloud's Qwen team, the GGUF conversions by
> [city96](https://huggingface.co/city96/Qwen-Image-gguf) and
> [Unsloth](https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF), and the
> [Lightning](https://huggingface.co/lightx2v/Qwen-Image-Lightning) LoRA by
> lightx2v, all published under the Apache 2.0 license. Klartraum neither ships
> nor endorses these models. You are solely responsible for complying with their
> licenses and for how you use the models and anything they generate.

Klartraum runs the Qwen-Image text-to-image model on Vulkan: the text encoder
(the language model of Qwen2.5-VL-7B, `QwenTextEncoder`) and the 20B-parameter
MMDiT transformer (`QwenImageTransformer`) from GGUF files with their weights
quantized on the GPU, the Lightning LoRA (8 steps, no classifier-free guidance)
added at run time, and the VAE (the Wan 2.1 VAE, one frame) as ONNX graphs.
`examples/qwen_image_example.cpp` puts them together.

## Setup

From the repository root:

```bash
uv run --project scripts/qwen_image python scripts/qwen_image/download.py      # ~18 GB into build/models/qwen-image/
uv run --project scripts/qwen_image python scripts/qwen_image/convert_lora.py  # Lightning LoRA -> GGUF
uv run --project scripts/qwen_image python scripts/qwen_image/export_vae.py --size 512 512
```

`export_vae.py` writes fixed-shape encoder and decoder graphs for one image
size; export every size you want to generate. It checks them against diffusers
and ONNX Runtime.

## References for the tests

The tests compare Klartraum with diffusers and transformers on the same
dequantized weights; they skip when the files are missing:

```bash
uv run --project scripts/qwen_image python scripts/qwen_image/export_vae.py --size 256 256   # test_qwen_image_vae
uv run --project scripts/qwen_image python scripts/qwen_image/text_reference.py              # test_qwen_text_encoder
uv run --project scripts/qwen_image python scripts/qwen_image/dit_reference.py               # test_qwen_image_transformer
uv run --project scripts/qwen_image python scripts/qwen_image/dit_reference.py --lora
```

`dit_reference.py` reads the prompt embedding written by `text_reference.py`.
