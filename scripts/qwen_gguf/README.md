# Qwen3.6-27B (GGUF)

> **Warning:** Qwen3.6 is not part of Klartraum. These scripts download and
> process third-party Qwen model weights, which Alibaba Cloud's Qwen team
> distributes under the Apache 2.0 license; the GGUF conversions are made by
> Unsloth. Klartraum neither ships nor endorses these models. You are solely
> responsible for complying with the model license and for how you use the
> models and anything they generate.

Klartraum runs GGUF language models (the llama.cpp file format) with
`GgufNetwork`, the GGUF counterpart of `OnnxNetwork`: a compute-graph group
whose layers are Vulkan compute shaders (`shaders/gguf/`). The weights stay in
their GGUF encoding on the GPU and are decoded inside the matrix-vector
kernels. Supported architecture: `qwen35` (Qwen3.5 / Qwen3.6 dense), such as
[Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B), a hybrid of 48
Gated-DeltaNet (linear attention) layers and 16 gated full-attention layers.
Supported weight types: F32, F16, BF16, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, which
covers the Q3_K_S … Q8_0 files of
[unsloth/Qwen3.6-27B-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF)
(not the IQ* and Q4_0/Q4_1 files). Text only; the vision tower (`mmproj`) is
not used.

## Download

From this directory:

```bash
uv run python download.py            # Q3_K_M (13.6 GB) with <= 32 GB RAM, else Q4_K_M (16.8 GB)
uv run python download.py --quant Q4_K_M
uv run python download.py --list     # all files and whether Klartraum supports them
```

The file lands in `data/gguf/qwen3.6-27b/` (ignored by git). The whole model
stays resident in GPU memory, so on a 24 GB Mac Q3_K_M is the largest
practical choice. If the download stalls, use `HF_HUB_DISABLE_XET=1`.

## Chat

From the repository root:

```bash
./build/examples/qwen_chat_example                  # interactive
./build/examples/qwen_chat_example --think          # with thinking (shown dimmed)
./build/examples/qwen_chat_example --prompt "Explain Vulkan in two sentences."
```

See `examples/README.md` for the options and commands.

## Reference tooling

| Script | Purpose |
|--------|---------|
| `download.py` | Downloads a supported quantization. |
| `reference.py` | NumPy forward pass over a `qwen35` GGUF file (dequantized with gguf-py); prints and optionally saves the logits for a token list. |
| `make_tiny_model.py` | Random 4-layer Qwen3.5 (transformers) in the `qwen35` GGUF layout plus its reference logits, for `tests/test_gguf_qwen35.cpp`. Needs `uv sync --group reference`. |
| `make_test_fixtures.py` | Random quantized blocks and their gguf-py decoding, for `tests/test_gguf_quants.cpp`. |
| `make_tokenizer_cases.py` | Token ids of test strings from the official tokenizer, for `tests/test_gguf_tokenizer.cpp`. |
| `make_unicode_tables.py` | Regenerates `src/unicode_categories.inc` (letter/mark/number ranges for the pre-tokenizer). |

`reference.py` checks the tiny model against transformers
(`max |logits - reference| = 5.7e-06`) and can cross-check Klartraum on the
real model, e.g.
`uv run python reference.py ../../data/gguf/qwen3.6-27b/Qwen3.6-27B-Q3_K_M.gguf --tokens 760 6511 314 9338 369 --save-gguf ../../build/TestingOutput/qwen_reference.gguf`.

## Tensor layout notes

The `qwen35` GGUF layout differs from the Hugging Face checkpoint:

- RMSNorm weights are stored as `1 + w` (Qwen3.5 normalizes with `1 + w`),
  except `ssm_norm`, the gated norm of the linear attention.
- `ssm_a` holds `-exp(A_log)`.
- Linear-attention value heads are in *tiled* order: value head `j` reads key
  head `j % key_heads` (the checkpoint groups the value heads of each key head).
- `attn_q` produces, per head, the query followed by its output gate.
- `ssm_conv1d` is `[channels][4]`, oldest tap first.

Rotary embeddings: Qwen3.5 uses interleaved multimodal RoPE; for text, all
three position components are equal, so it reduces to NEOX RoPE on the first
64 of 256 head dimensions (base 1e7).
