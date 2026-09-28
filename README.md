# klartraum

Klartraum (German for *lucid dream*) is a real-time neural rendering and inference engine built on top of Vulkan.

**[Documentation](https://klartraum.ai/docs/)** · **[Website](https://klartraum.ai)** · **[Klartraum Studio](https://github.com/fortmeier/klartraum-studio)**

![A stone lantern rendered with Gaussian splatting by Klartraum, turning once around its axis](site/images/engine-lantern.gif)

## What it does

- **Combines neural and classical rendering:** Gaussian splatting, neural networks, rasterization and ray tracing in one pipeline.
- **Runs neural networks on any Vulkan device,** from single-board computers to virtual-reality headsets and data center GPUs.
- **Keeps the CPU out of the way:** all GPU work is recorded into Vulkan command buffers once; each frame only submits them.
- **Works with or without a window,** as an interactive application or as a headless inference engine.

See [Vision and roadmap](https://klartraum.ai/docs/vision.html) for where it is heading.

## Status

Early development: the compute graph works, Gaussian splatting runs on the GPU with a compute and a raster backend, and a first set of ONNX operators is supported.

## Quick start

```bash
git clone https://github.com/fortmeier/klartraum.git
cd klartraum
git submodule update --init --recursive
cmake -S . -B build
cmake --build build

# from the repository root
./build/examples/gaussian_splatting_example        # Windows: .\build\examples\Debug\gaussian_splatting_example.exe
```

Requirements, platform notes, tests and a walk-through of a first application are in the [documentation](https://klartraum.ai/docs/getting-started/building.html).

## License

[MIT](LICENSE)
