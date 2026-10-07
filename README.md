# klartraum

Klartraum (German for *lucid dream*) is a real-time neural rendering and inference engine built on top of Vulkan.

**[Documentation](https://klartraum.ai/docs/)** · **[Website](https://klartraum.ai)** · **[Klartraum Studio](https://github.com/fortmeier/klartraum-studio)**

![A stone lantern rendered with Gaussian splatting by Klartraum, turning once around its axis](site/images/engine-lantern.gif)

## Goals

- **Combine neural and classical rendering:** Gaussian splatting, neural networks, rasterization and ray tracing in one pipeline.
- **Run neural networks on any Vulkan device,** from single-board computers to virtual-reality headsets and data center GPUs.
- **Keep the CPU out of the way:** all GPU work is recorded into Vulkan command buffers once; each frame only submits them.
- **Work with or without a window,** as an interactive application or as a headless inference engine.

See [Vision and roadmap](https://klartraum.ai/docs/vision.html) for where it is heading.

## Status

Early development. What works today:

- the compute graph: GPU work described as a graph of elements, recorded into
  command buffers once and submitted per frame, with GPU timings per element;
- Gaussian splatting with a compute and a raster backend (`.spz` scenes);
- ONNX models with the operators `Conv`, `ConvTranspose`, `Relu`, `Reshape`,
  `Transpose` and `Constant`;
- windowed (GLFW) and headless frontends, several viewports in one window,
  resizable windows and a debug text overlay.

Ray tracing is not implemented yet.

## Requirements

- CMake 3.24 or newer and a C++17 compiler
- the [Vulkan SDK](https://vulkan.lunarg.com/) (loader, headers and `glslc`)
- tested on Windows (Visual Studio 2022) and macOS on Apple Silicon (MoltenVK
  or KosmicKrisp); Linux is not tested yet

All other dependencies are git submodules or are downloaded by CMake.

## Quick start

```bash
git clone https://github.com/fortmeier/klartraum.git
cd klartraum
git submodule update --init --recursive
cmake -S . -B build
cmake --build build

# from the repository root
./build/examples/gaussian_splatting_example        # Windows: .\build\examples\Debug\gaussian_splatting_example.exe
./build/klartraum_tests                            # Windows: .\build\Debug\klartraum_tests.exe
```

[examples/README.md](examples/README.md) describes all example programs and
their options. Platform notes, build options and a walk-through of a first
application are in the [documentation](https://klartraum.ai/docs/getting-started/building.html).

## Contributing

The code style, the checks that run before each commit and how to build the
documentation are described under
[Contributing](https://klartraum.ai/docs/contributing/development.html) in the
documentation.

## License

Klartraum is licensed under the [MIT License](LICENSE). The images, videos and
the captured lantern scene (`data/lantern.jpg`, `data/lantern.spz`,
`site/images/`, including the animation above) are licensed under
[CC BY 4.0](LICENSES/CC-BY-4.0.txt), © Dirk Fortmeier; [REUSE.toml](REUSE.toml)
lists the license of every file. Builds include third-party software under its
own licenses, listed with their notices in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
