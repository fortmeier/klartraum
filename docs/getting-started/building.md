# Building

Klartraum is built with CMake 3.24 or newer and needs a C++17 compiler and the
[Vulkan SDK](https://vulkan.lunarg.com/) (for the loader, the headers and the
`glslc` shader compiler). All other dependencies are git submodules or are
downloaded by CMake. Their licenses are listed in
[THIRD_PARTY_NOTICES.md](https://github.com/fortmeier/klartraum/blob/main/THIRD_PARTY_NOTICES.md);
pass that file on with any build you distribute.

## Getting the sources

```bash
git clone https://github.com/fortmeier/klartraum.git
cd klartraum
git submodule update --init --recursive
```

## Windows (Visual Studio 2022, x64)

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

The binaries end up in `build\Debug\` (tests) and `build\examples\Debug\`
(examples).

## macOS (Apple Silicon)

Klartraum runs on macOS through a Vulkan portability driver such as MoltenVK
or KosmicKrisp. Install the Vulkan SDK and make sure `VULKAN_SDK` points to it;
CMake uses it to find the Vulkan loader and the matching `glslc`.

```bash
mkdir build
cd build
cmake ..
cmake --build . -j
```

The binaries end up in `build/` (tests) and `build/examples/` (examples).

## Linux

Linux is not tested yet. With the Vulkan SDK installed, the same CMake steps
as on macOS are expected to work.

## Build options

| Option | Default | Meaning |
|---|---|---|
| `KLARTRAUM_BUILD_TESTS` | `ON` when built as the top-level project | Build the GoogleTest suite `klartraum_tests` |
| `KLARTRAUM_BUILD_EXAMPLES` | `ON` when built as the top-level project | Build the example applications |

When Klartraum is consumed through `add_subdirectory()` or `FetchContent`,
both default to `OFF`; link against the `klartraum_lib` target to get the
library and its headers.

## Shaders

Compute and graphics shaders are written in GLSL and compiled to SPIR-V by
`glslc` as part of the build. The compiled `.spv` files are written to
`shaders/` in the build directory and loaded at runtime by relative path (for
example `shaders/gsplat/gsplat_projection.comp.spv`). Such a path is looked up
in the working directory first and then in the asset root:

1. the directory set with `klartraum::setAssetRoot()`,
2. otherwise the `KLARTRAUM_ASSET_DIR` environment variable,
3. otherwise the build directory of Klartraum.

Applications therefore find the shaders from any working directory. To run a
binary on another machine or after moving the build, copy the `shaders/`
directory from the build directory and point `KLARTRAUM_ASSET_DIR` (or
`setAssetRoot()`) to the directory that contains it.

## Running the tests

The tests read their scenes and models from `data/` relative to the working
directory, so they must be run from the **repository root**:

```bash
# Windows
.\build\Debug\klartraum_tests.exe

# macOS
./build/klartraum_tests

# a single test
./build/klartraum_tests --gtest_filter=GaussianSplattingTest.classWithLanternScene
```

Alternatively, run `ctest` from the build directory; it runs the tests from
the repository root.

## Running an example

```bash
# from the repository root
./build/examples/gaussian_splatting_example
./build/examples/gaussian_splatting_example --backend raster
./build/examples/gaussian_splatting_example --file data/lantern.spz --camera-position 0.55 0.48 0.69 --flip-y
```

{doc}`examples` describes all example programs with their options.
