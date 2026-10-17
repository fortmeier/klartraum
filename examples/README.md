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
| `title_font_example` | The promo title set in several fonts and turned into Gaussians, rendered front-on for comparison | no |
| `title_morph_example` | The promo opening shot: the title, made of the scene's Gaussians, morphs into the turning lantern | no |

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

### `title_font_example`

Sets the promo title ("klartraum" over "tech demo 000") in each of several
fonts with FreeType, places white Gaussians inside the letters
(`text_gaussians.hpp`) and renders them front-on, white on black. Writes one
image per font to `frames/<Font>.ppm` and all of them, half size and labelled,
to `contact_sheet.ppm`. The fonts are downloaded by the build to `build/fonts/`
(see `cmake/KlartraumFonts.cmake`); configure with
`-DKLARTRAUM_DOWNLOAD_FONTS=OFF` to skip that.

| Option | Meaning |
|---|---|
| `--fonts A,B,...` | Fonts for the title: names from `build/fonts/` or paths to `.ttf`/`.otf` files (default: SpaceGrotesk, Geist, Unbounded, Syne, IBMPlexSans, Fraunces) |
| `--title TEXT`, `--subtitle TEXT` | The two lines (default `klartraum`, `tech demo 000`; an empty subtitle leaves it out) |
| `--subtitle-font NAME` | Font of the subtitle (default `JetBrainsMono`) |
| `--weight W`, `--subtitle-weight W` | Value of the fonts' `wght` axis (default 500 and 400) |
| `--size S`, `--subtitle-size S` | Em size in world units (default 0.5 and 0.2) |
| `--letter-spacing E`, `--subtitle-letter-spacing E` | Extra space between letters, in em (default 0 and 0.05) |
| `--line-spacing F` | Distance between the baselines, in em of the larger line (default 1.2) |
| `--spacing D` | Distance between neighbouring splats in world units (default 0.003) |
| `--splat-scale F` | Splat size relative to `--spacing` (default 0.6) |
| `--jitter F` | Random offset of each splat, relative to `--spacing` (default 0.35) |
| `--inset F` | How far splats keep inside the outline, relative to `--spacing` (default 0.7), so the soft splat edges do not make the letters bolder |
| `--depth-jitter D` | Random depth offset of each splat in world units (default 0) |
| `--seed N` | Seed of the jitter (default 1) |
| `--width W`, `--height H` | Image size in pixels (default 1920×1080) |
| `--distance D` | Camera distance from the text (default 3) |
| `--out-dir DIR` | Output directory (default `build/TestingOutput/title_fonts`) |

```bash
./build/examples/title_font_example --weight 600 --out-dir build/TestingOutput/title_fonts
ffmpeg -i build/TestingOutput/title_fonts/contact_sheet.ppm title_fonts.png
```

### `title_morph_example`

Renders the opening shot of the promo video without a window. The title
("klartraum" in Unbounded over "tech demo 000" in JetBrains Mono) is sampled
with exactly as many splats as the scene has Gaussians and placed facing the
camera at the scene centre. After `--title-seconds` it morphs into the scene
(`GaussianMorph`, each text splat flies to the Gaussian it is matched with by
screen position), while the camera speeds up evenly into a turntable orbit that
completes `--turns` turns by the end of the shot. The morph runs on the GPU in a
graph recorded once; only the progress changes per frame.

Frames go to `<out-dir>/<variant>/frames/frame_XXXX.ppm`. Afterwards the program
checks each variant: the title frames must be identical and the last frame must
equal the scene rendered directly from the same camera; otherwise it exits with
an error.

| Option | Meaning |
|---|---|
| `--variant NAME` | `straight`, `swirl` (spirals around the vertical axis), `scatter` (out into a cloud and back), `wave` (letter by letter from the left), `explode` (the letters burst apart, come to rest, then fly into place), `explode-swirl` (burst, then spiral in), `shatter` (burst towards the camera, then into place), `swirl-scatter` (the scatter while circling the vertical axis), `swirl-wave` (letter by letter, each on a spiral) or `all` (default) |
| `--fps F` | Frame rate (default 30) |
| `--title-seconds S`, `--morph-seconds S`, `--spin-seconds S` | Length of the still title, the morph and the turntable after it (default 2, 1.5, 5) |
| `--turns N` | Camera turns from the start of the morph to the end of the shot (default 1) |
| `--swirl-turns N` | Extra turns of each splat around the axis in the swirls, rounded to whole turns (default 1) |
| `--scatter-radius R` | How far splats fly out in the scatter, in world units (default 0.25) |
| `--wave-spread F` | Part of the morph over which the starts of the wave are spread (default 0.6) |
| `--explode-radius R` | How far splats burst out in the explosions, in world units (default 0.3) |
| `--explode-share F` | Part of the morph the burst takes before the splats fly into place (default 0.35) |
| `--file PATH`, `--no-flip-y` | Scene (default `data/lantern.spz`, mirrored across Y as the lantern needs) |
| `--camera-position X Y Z`, `--distance D` | Start position of the camera relative to the scene centre and its distance (default `0.55 0.48 0.69`, 0.65) |
| `--width W`, `--height H` | Image size in pixels (default 1920×1080) |
| `--text-width F` | Width of the wordmark as a fraction of the image width (default 0.7) |
| `--font NAME`, `--weight W`, `--subtitle-font NAME` | Fonts as in `title_font_example` (default Unbounded at 500, JetBrains Mono) |
| `--title TEXT`, `--subtitle TEXT` | The two lines (default `klartraum`, `tech demo 000`) |
| `--out-dir DIR` | Output directory (default `build/TestingOutput/promo_shot1`) |

```bash
./build/examples/title_morph_example --variant all
cd build/TestingOutput/promo_shot1
for v in straight swirl scatter wave explode explode-swirl shatter swirl-scatter swirl-wave; do
    ffmpeg -framerate 30 -i $v/frames/frame_%04d.ppm -c:v libx264 -crf 16 -pix_fmt yuv420p $v/$v.mp4
done
ffmpeg -i straight/straight.mp4 -i swirl/swirl.mp4 -i scatter/scatter.mp4 -i wave/wave.mp4 -filter_complex \
    "[0]scale=960:540[a];[1]scale=960:540[b];[2]scale=960:540[c];[3]scale=960:540[d];[a][b]hstack[t];[c][d]hstack[u];[t][u]vstack" \
    -c:v libx264 -crf 18 -pix_fmt yuv420p variants_2x2.mp4
```
