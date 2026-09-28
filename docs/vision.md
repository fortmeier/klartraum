# Vision and roadmap

Klartraum aims to be an execution environment for pretrained deep learning
models that runs on a wide range of hardware wherever Vulkan is available:
from single-board computers such as the Raspberry Pi to virtual-reality
headsets.

## Model types

| Model type | Status |
|---|---|
| Gaussian splatting | implemented (compute and raster backends) |
| Convolutional neural networks | first ONNX operators implemented (Conv, ConvTranspose, ReLU, Reshape, Transpose) |
| Diffusion networks | not implemented yet |
| Transformers and LLMs | not implemented yet |

## Neural rendering pipelines

Because neural networks and classic real-time rendering run in the same
Vulkan compute graph, they can be combined into one pipeline. An example of
what this should make possible:

1. render Gaussian splats into a learned embedding space instead of colours,
2. decode that embedding into an image with a CNN decoder,
3. combine the result with classic rasterization or ray tracing,
4. increase resolution and fidelity with an upscaler such as DLSS.

## Inference on edge devices

Without any rendering, Klartraum can serve as a hardware-independent inference
engine for deploying deep learning models on edge devices, using the headless
frontend (see {doc}`concepts/frontends`).

## Integration with other engines

The compute graph is designed to be usable without Klartraum's own rendering
engine, so that it can run alongside other Vulkan-based engines. This is not
implemented yet: it needs another implementation of
{cpp:class}`klartraum::VulkanContext` that uses the host engine's Vulkan
instance and device.
