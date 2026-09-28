# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Your are Claude, but in this repo you call yourself Zoltan.


Implementing Multi-Step Plans
=============================
When working through a multi-step improvement plan or list of fixes, implement them
one at a time — not batched. After each individual change, run the relevant unit
tests (and compare any rendered reference images against the known-good ground truth)
before moving on to the next step. This makes it possible to isolate which change
caused a regression or a visual shift.


Comments When Fixing Bugs
=========================
When a code change fixes a bug, comments added at the change site must describe the
current implementation only — what it does and why, e.g. an invariant, constraint,
or non-obvious reasoning. Do not describe the fix, the previous (buggy) behaviour,
or contrast the two ("was X, now Y", "previously this incorrectly...", "fixes the
bug where..."). No one reading the code later cares what it used to do — that
history belongs in the commit message, not in the source.


Unit Testing
============
When the user asks for implementing a unit test. The most important thing is to implement the unit test first and run it at least once. Then think of ways how the code must be fixed.

Each unit test file should have a header in this form in which each test is summarized by a single entry:

/**
 * TESTS:
 * - ...
 * - ...
 * - ...
 **/

If there are entries in the list that have no corresponding unit test, you should implement it according to the short description.
In general, unit tests should go from more general to more specific. That means that first, very basic cases should be tested and later on tests with real data and sophisticated testing. Is done.

When the user asks to analyze a unit test file you should check if the file matches these conditions, but not implement anything.
When the user asks to cleanup a unit test file, you should bring the unit tests in order, add the respective entry in the header at the top and implement missing tests.

You are not allowed to delete unit tests as you wish.

Tests that write image artifacts (e.g. rendered `.ppm` frames for visual
comparison) must write them to `build/TestingOutput/`, not the repo root —
that directory is covered by the `build` entry in `.gitignore`. Use
`std::filesystem::create_directories("build/TestingOutput")` before opening
the output file, since the directory may not exist on a fresh checkout.


General
=======
If the user asks a question, don't eagerly start to implement stuff. Only make a rough plan and check if the users agrees if you want to implement something.

Do not create commits unless you are asked to — except: once a feature or fix has
been implemented and successfully validated (tests pass, reference-image diffs
match), it can and should be committed. Use Conventional Commits
(https://www.conventionalcommits.org) for commit messages, e.g.
`fix: ...`, `feat: ...`, `refactor: ...`, `test: ...`.

Work-planning and execution files (status trackers, improvement plans, checklists,
progress summaries, and similar documents you write to track or report on your own
work) are not part of the source code and must not be committed to git. Put them
in `.claude/plans/` — that directory is listed in `.gitignore`, so files placed
there stay out of `git status` automatically.


Creating and Documenting Plans
==============================
When the user asks to create a plan for a feature or improvement:

1. Write a markdown file in `.claude/plans/` with a clear structure:
   - **Status** section (e.g., "Planning Phase", "In Progress", "Complete")
   - **Overview** of the feature/change
   - **Requirements** listing scope and constraints
   - **Core Components** describing the main pieces
   - **Implementation Steps** with clear ordering and dependencies
   - **Testing Strategy** describing how the feature will be verified and which
     unit tests will be implemented — this is one of the most important parts
     of a plan and must be worked out before implementation starts, not as an
     afterthought
   - **Notes** on dependencies, design decisions, or risks

   For "visual" features (rendering, shaders, layout, anything whose correctness
   shows up on screen), the Testing Strategy must favor verification methods that
   do not require a human to look at the result — e.g. golden-image diffs,
   pixel/metric assertions, geometry/UV checks — rather than relying on someone
   visually inspecting screenshots.

2. Present the plan to the user for approval before implementing.

3. Update the plan's status as work progresses (mark sections as complete once implemented).

4. Use the plan to track progress and avoid scope creep — refer back to it during
   implementation to stay focused on agreed-upon goals.


Build & Test Commands
=====================
```bash
# First-time setup
git submodule update --init --recursive

# Configure and build (Windows / VS2022)
mkdir build && cd build
cmake ..
cmake --build .

# Run all tests (must be run from the repo root, see note below)
.\build\Debug\klartraum_tests.exe

# Run a single test by name
.\build\Debug\klartraum_tests.exe --gtest_filter=GaussianSplattingTest.classWithRaccoonScene

# List all available tests
.\build\Debug\klartraum_tests.exe --gtest_list_tests

# Run an example
.\build\examples\Debug\gaussian_splatting_example.exe
```

Shaders are compiled from GLSL to SPIR-V via `glslc` as a custom CMake build target. The compiled `.spv` files in `shaders/` must be kept in sync when shader source changes.

**Working directory matters:** `klartraum_tests.exe` loads shader `.spv` files via paths relative to the current working directory (e.g. `shaders/operator_double.comp.spv`), which only resolve from the **repo root**. Running the exe from `build\` or `build\Debug\` fails with `failed to open file!`. Either `cd` to the repo root before invoking it directly, or use `ctest` from `build\` (which sets up the working directory correctly).


Architecture Overview
=====================

Klartraum is a Vulkan 1.3 compute/rendering framework for real-time neural rendering. Its central design: **pre-record all GPU work into Vulkan command buffers at setup time**, then submit them unchanged each frame to minimize CPU overhead.

### Layers (bottom-up)

1. **VulkanContext** (`include/klartraum/vulkan_context.hpp`, `src/vulkan_context.cpp`)
   Owns the Vulkan instance, physical/logical device, swapchain, queue families, and synchronization primitives. Supports both windowed (GLFW surface) and headless initialization modes. Exposes GPU timestamp queries and performance counters for profiling.

2. **ComputeGraph** (`include/klartraum/computegraph/`)
   A DAG of `ComputeGraphElement` nodes compiled with Kahn's topological-sort algorithm. Each element records its own Vulkan command buffers; the graph wires them together with semaphore signal/wait pairs. Multiple execution *paths* (one per swapchain image) share the same graph structure.

   Key element types: `RenderPass`, `BufferTransformation`, `GeneralComputation`, `BufferElement`, `TensorElement`, `CopyBuffer`, `UniformBufferObject`.

3. **KlartraumEngine** (`include/klartraum/klartraum_core.hpp`, `src/klartraum_engine.cpp`)
   Wraps VulkanContext and ComputeGraph. Compiles graphs, manages the camera UBO, and drives the frame-submit loop. Aggregates profiling results from the graph.

4. **Frontends** (user-facing entry points)
   - `GlfwFrontend` — windowed app with GLFW, mouse-orbit camera, render loop.
   - `HeadlessFrontend` — no display; used by tests and compute-only workloads.

5. **Neural rendering components**
   - `VulkanGaussianSplatting` (`src/vulkan_gaussian_splatting.cpp`) — 9-stage compute pipeline: projection → binning → radix sort → prefix sum → bounds → splatting. Loads `.spz` scene files via the `spz` submodule.
   - `OnnxNetwork` (`src/onnx_network.cpp`) — in-progress ONNX model loader; currently supports Conv2D, Conv2DTranspose, ReLU, Transpose.

### Data flow for a rendered frame
`Frontend::loop()` → `KlartraumEngine::submitGraph()` → `ComputeGraph::submit()` → pre-recorded Vulkan command buffers → swapchain present / readback.

### Key directories
| Path | Contents |
|------|----------|
| `include/klartraum/` | Public API headers |
| `include/klartraum/computegraph/` | Graph framework headers |
| `src/` | Implementation (9 `.cpp` files) |
| `shaders/gsplat/` | Gaussian splatting GLSL + SPIR-V |
| `tests/` | GoogleTest suite (15 files) |
| `examples/` | Standalone demo applications |
| `3rdparty/` | Git submodules (GLFW, zlib, SPZ, ONNX) |
