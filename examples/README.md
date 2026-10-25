# Klartraum examples

Small programs that show how to use the Klartraum library. Each one is a single
`.cpp` file in this directory.

<!-- docs-start -->

## Building and running

The examples are built together with the library when Klartraum is the
top-level CMake project (option `KLARTRAUM_BUILD_EXAMPLES`, on by default):

```bash
cmake -S . -B build
cmake --build build
```

The programs end up in `build/examples/`; with a multi-config generator such as
Visual Studio in `build/examples/<Config>/`, e.g.
`.\build\examples\Debug\gaussian_splatting_example.exe`. Run them from the
repository root, because they load the scene and the models from `data/`
relative to the working directory.

| Example | Shows | Window |
|---|---|---|
| `gaussian_splatting_example` | A Gaussian splatting scene with an orbit camera, using the compute or the raster backend | yes |
| `multi_viewports_single_camera_example` | Two viewports in one window, the compute backend next to the raster backend, driven by one camera | yes |
| `gaussian_autoencoder_example` | Gaussian splatting feeding an ONNX encoder and decoder, all in one compute graph | yes |
| `draw_basics_example` | Coordinate axes drawn with `DrawBasics` in a render pass | yes |
| `onnx_example` | Loading an ONNX model, printing its structure and compiling it into a compute graph | opens one, exits right away |
| `turntable_example` | Headless rendering: one image per frame from a camera circling the scene | no |

### Camera controls

The windowed examples use the orbit camera (`InterfaceCameraOrbit`):

| Input | Action |
|---|---|
| Left mouse button + drag | Orbit around the pivot |
| Mouse wheel | Move closer or further away |
| W / S / A / D | Move the pivot forward, back, left, right |
| Space | Reset the camera |
| Escape | Quit |

## Examples

### `gaussian_splatting_example`

Loads a Gaussian splat scene (`.spz`) and renders it in a window. The window
can be resized: the graph is rebuilt for the new swapchain through
`KlartraumEngine::setGraphBuilder()`, while the scene stays loaded.

| Option | Meaning |
|---|---|
| `--backend compute\|raster` | Rendering backend (default `compute`) |
| `--file PATH` | Scene to load instead of `data/lantern.spz` |
| `--flip-y` | Mirror the scene given with `--file` across the Y axis (the lantern scene is always flipped) |
| `--camera-position X Y Z` | Start position of the camera in world space, looking at the origin (default `0.55 0.48 0.69`) |
| `--frames N` | Close after `N` frames and print the mean GPU time of each graph element |

```bash
./build/examples/gaussian_splatting_example
./build/examples/gaussian_splatting_example --backend raster
./build/examples/gaussian_splatting_example --file path/to/scene.spz --flip-y --camera-position 0.55 0.48 0.69
```

### `multi_viewports_single_camera_example`

Renders the lantern scene twice in one window: the raster backend in the left
half, the compute backend in the right half. `Window::makeViewport()` returns a
render target for each half; the engine composites both into the swapchain
image and presents once per frame. Both backends read the same camera uniform
buffer, so one orbit camera moves both views. No options.

```bash
./build/examples/multi_viewports_single_camera_example
```

### `gaussian_autoencoder_example`

Chains Gaussian splatting and two neural networks in one compute graph: the
lantern scene is rendered at 128×128 pixels with the compute backend, converted
into a tensor, encoded and decoded by the ONNX models
`data/onnx/simple_encoder.onnx` and `data/onnx/simple_decoder.onnx`, and the
decoded image is shown across the window. The models are small test fixtures
trained on a photo of the lantern; how they are made is described in
[`scripts/onnx/`](https://github.com/fortmeier/klartraum/tree/main/scripts/onnx).

| Option | Meaning |
|---|---|
| `--spz PATH` | Scene to render instead of `data/lantern.spz` (loaded without flipping the Y axis) |
| `--frames N` | Close after `N` frames and print the value range of the decoded image |

```bash
./build/examples/gaussian_autoencoder_example
```

### `draw_basics_example`

The smallest windowed program: a render pass with one draw component,
`DrawBasics`, that draws the coordinate axes, and an orbit camera. No options.

```bash
./build/examples/draw_basics_example
```

### `onnx_example`

Loads `data/onnx/simple_encoder.onnx`, prints its inputs, outputs and nodes, and
compiles it into a compute graph. It opens a window only to create the Vulkan
context and exits after compiling. No options.

```bash
./build/examples/onnx_example
```

### `turntable_example`

Renders a scene without a window, from a camera that circles the scene's
vertical axis exactly once, and writes one image per frame
(`frame_0000.ppm`, `frame_0001.ppm`, …). The camera circles the centre of the
scene: the median of the Gaussian positions, so a few stray splats do not pull
it off the object. After the last frame, the program renders the view after a
full turn and compares it with the first frame; if they differ, the animation
would not loop, and the program exits with an error.
[`scripts/site/make_lantern_animation.sh`](https://github.com/fortmeier/klartraum/blob/main/scripts/site/make_lantern_animation.sh)
uses it for the animation at the top of the project README.

| Option | Meaning |
|---|---|
| `--file PATH` | Scene to render (default `data/lantern.spz`) |
| `--flip-y` | Mirror the scene across the Y axis; needed for the lantern scene |
| `--frames N` | Number of frames in one turn (default 72) |
| `--width W`, `--height H` | Image size in pixels (default 480×360) |
| `--camera-position X Y Z` | Start position of the camera relative to the scene centre (default `0.55 0.48 0.69`) |
| `--distance D` | Distance of the camera from the scene centre; overrides the length of `--camera-position` |
| `--backend compute\|raster` | Rendering backend (default `compute`) |
| `--out-dir DIR` | Output directory (default `build/TestingOutput/turntable`) |

```bash
./build/examples/turntable_example --file data/lantern.spz --flip-y --frames 60 \
    --width 400 --height 300 --camera-position 0.55 0.48 0.69 --distance 0.65 \
    --out-dir build/TestingOutput/turntable
```
