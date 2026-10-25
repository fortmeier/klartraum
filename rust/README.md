# Klartraum in Rust

A parallel implementation of the Klartraum engine in Rust, next to the C++ code.
It ports the framework itself — VulkanContext, the ComputeGraph and its
elements, KlartraumEngine and the frontends — on top of
[ash](https://github.com/ash-rs/ash), and runs the GLSL shaders in
[`shaders/`](../shaders) unchanged.

The first feature set is the Gaussian splatting example with the raster
backend (`examples/gaussian_splatting_example.cpp --backend raster`).

## Building and testing

Requires a Rust toolchain (rustup), the Vulkan SDK (loader, validation layers)
and `glslc` on the `PATH` (or in `GLSLC`); `build.rs` compiles the shaders.

```bash
cd rust
cargo build
cargo test            # run from anywhere; tests switch to the repository root
cargo run --example gaussian_splatting_example -- --frames 100
```

The tests compare the Rust engine against the C++ engine. The reference data
(Gaussians as loaded, camera matrices, the raster render of the lantern) is
written by the C++ tests `RustReference.*` (`tests/test_rust_reference.cpp`) to
`build/TestingOutput/rust_reference/`; run them once after building the C++
project:

```bash
./build/klartraum_tests --gtest_filter='RustReference.*'
```

Tests whose reference data is missing return early and print a "skipped" note (Rust has no runtime test skip).
