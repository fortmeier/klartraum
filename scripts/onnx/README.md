# ONNX fixture generation

The ONNX fixtures are generated from a uv-managed Python 3.12 environment. The
dependency versions are recorded in `pyproject.toml` and `uv.lock`; Conda is not
required. On Windows and Linux, PyTorch comes from its CUDA 12.8 package index.
Training uses an NVIDIA GPU when one is available and falls back to the CPU.

## Set up the environment

From this directory:

```powershell
uv sync
```

To work interactively, start the notebook in the same environment:

```powershell
uv run jupyter lab train_simple_autoencoder.ipynb
```

## Regenerate all models headlessly

Run the checked-in notebook and save its executed copy under the ignored build
output directory:

```powershell
uv run jupyter nbconvert `
    --to notebook `
    --execute train_simple_autoencoder.ipynb `
    --output train_simple_autoencoder.executed.ipynb `
    --output-dir ../../build/TestingOutput `
    --ExecutePreprocessor.timeout=-1
```

The generator uses a fixed seed and trains on the complete Caltech-101 training
split by default. Set `KLARTRAUM_ONNX_TRAIN_SAMPLES` to a positive number to
train on only that many leading images for a quicker smoke run.

Published fixtures are written to `data/onnx/`:

- `simple_decoder.onnx`
- `simple_encoder_with_onnx_frozen_intermediates.onnx`
- `simple_decoder_with_onnx_frozen_intermediates.onnx`

Regenerable export intermediates are written to the ignored `data/onnx/tmp/`
directory. The frozen-intermediate models intentionally contain captured tensor
values as initializers so the C++ tests can compare Vulkan operator results with
ONNX Runtime reference results. They use the protobuf `float_data` representation
required by Klartraum's current loader. Because those captured initializers reuse
graph value names, the frozen fixtures are test annotations rather than
SSA-valid models for general ONNX Runtime execution.

## Verify the C++ consumer

From the repository root:

```powershell
.\build\Debug\klartraum_tests.exe --gtest_filter=OnnxNetworkTest.*
```

