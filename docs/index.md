# Klartraum Engine

Klartraum Engine is a real-time neural rendering and
inference engine built on top of Vulkan.

![A stone lantern rendered with Gaussian splatting by Klartraum, turning once around its axis](../site/images/engine-lantern.gif)

*Gaussian splatting scene rendered with Klartraum*

## Goals

- **Combine neural and classical rendering.** Gaussian splatting, CNN decoders,
  rasterization and ray tracing can be chained in one pipeline.
- **Run neural networks on any Vulkan device.** From single-board computers
  such as the Raspberry Pi to virtual-reality headsets and data center hardware,
  Klartraum runs
  pretrained models wherever a Vulkan driver is available.
- **Keep the CPU out of the way.** All GPU work is recorded into Vulkan command
  buffers once; each frame only submits them.
- **Work with or without a window.** The same compute graphs run in an
  interactive GLFW application or headless, as an inference engine on edge
  devices.

## Status

Klartraum is at an early stage of development. The compute graph is
implemented, Gaussian splatting runs on the GPU and a first set of ONNX
operators is supported.

```{toctree}
:hidden:
:caption: Getting started

self
vision
getting-started/building
getting-started/first-application
getting-started/examples
```

```{toctree}
:hidden:
:caption: Concepts

concepts/prerecorded-command-buffers
concepts/compute-graph
concepts/frontends
```

```{toctree}
:hidden:
:caption: Guides

guides/gaussian-splatting
guides/onnx
```

```{toctree}
:hidden:
:caption: Reference

api/index
requirements/index
```

```{toctree}
:hidden:
:caption: Contributing

contributing/development
contributing/code-style
contributing/documentation
```
