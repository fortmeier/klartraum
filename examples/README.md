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
| `title_font_example` | The promo title set in several fonts and turned into Gaussians, rendered front-on for comparison (needs `KLARTRAUM_WITH_FREETYPE`) | no |
| `title_morph_example` | The promo opening shot: the title, made of the scene's Gaussians, morphs into the turning lantern (needs `KLARTRAUM_WITH_FREETYPE`) | no |
| `autodiff_training_example` | Training a small network on the GPU, with gradients from Slang's automatic differentiation | no |
| `qwen_chat_example` | A chat in the terminal with the Qwen3.6-27B language model, run from a GGUF file | no |
| `qwen_image_example` | Text-to-image and image-to-image with the Qwen-Image diffusion model (GGUF transformer and text encoder, ONNX VAE) | no |
| `landscape_gbuffer_example` | Metric depth and semantic classes of images, estimated by two networks (ONNX) | no |
| `photo_enhancer_example` | The bonsai landscape's forest fly-in made photo-realistic: G-buffers, a learnable hash-grid surface texture and a GAN generator (ONNX) per frame | no |
| `crate_gbuffer_example` | A toy scene for the photo enhancer: a crate on a floor from random cameras, with all G-buffers (`GBufferMeshRenderer`) | no |
| `crate_training_example` | Trains the crate's photo enhancer (texture, generator, appearance codes) entirely in Klartraum | no |

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
to `contact_sheet.ppm`. Built only with `-DKLARTRAUM_WITH_FREETYPE=ON`; the
fonts are then downloaded by the build to `build/fonts/` (see
`cmake/KlartraumFonts.cmake`).

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

Built only with `-DKLARTRAUM_WITH_FREETYPE=ON`. Renders the opening shot of the promo video without a window. The title
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


### `techdemo001_example`

Renders *klartraum tech demo 001*, a promo video of the differentiable rendering
and the procedural trees, without a window. Needs `KLARTRAUM_WITH_FREETYPE`. The
chapters run in one continuous camera move:

1. **Title and morph:** on white, the black "klartraum" logo (Gaussians in
   Unbounded) with a black sphere above it. The sphere morphs into the
   jack-o'-lantern by differentiable mesh optimization on the masked photographs
   of the pumpkin (`raster::MeshMorph`), starting with the first frame. Each frame
   runs optimization steps in the frame's compute graph, so the gradients flow
   from the photographs into the mesh's shape and texture. Beside it, the
   photograph in use and the training's rendering of the mesh from its camera.
   The pumpkin is lit by the scene's light from the start.
2. **Drop:** the pumpkin falls onto the splatted bonsai pot with its soil and
   the lantern, casting its shadow, and the camera follows.
3. **Growth:** the bonsai pine grows from a seedling, branching out step by
   step.
4. **Style transfer:** a summer afternoon holding while a painted style fades
   in (with `--style` or `--field`); with `--panels`, a training sample of the
   style beside it (rendering, classes, depth, watercolor target).
5. **Days:** the afternoon turning into night, then four days, one season
   each, stormy in fall and winter. The background is white by day and black
   by night. With `--panels`, the season's watercolor targets by day and by
   night beside them.

Each chapter shows a caption naming the work it builds on. Frames go to
`<out-dir>/frames/frame_XXXX.ppm`. The trained pumpkin goes to
`<out-dir>/pumpkin.obj`, so later chapters can be re-rendered alone.

| Option | Meaning |
|---|---|
| `--pot FILE` | The bonsai pot's `object.spz` (default `build/TestingOutput/splat_training/bonsai/object.spz`) |
| `--photos DIR` | The pumpkin's photographs as prepared by `scripts/mesh_fitting/prepare_data.py`, the morph's targets (default `build/TestingOutput/real_data/pumpkin_closed_1_512`) |
| `--width W`, `--height H` | Frame size (default 1280 x 720) |
| `--fps F` | Frames per second (default 30) |
| `--from N`, `--to M` | Render chapters N to M (0 to 4) |
| `--step S` | Write every S-th frame, for previews |
| `--out-dir DIR` | Output directory (default `build/TestingOutput/techdemo001`) |
| `--video FILE.mp4` | Encode the frames straight into an H.264 video with ffmpeg (through a pipe) instead of writing them as PPM files |
| `--dataset N` | Instead of the video: N renderings of the final scene from random cameras, seasons and times of day as `<out-dir>/dataset/image_XXXX.png`, each with its G-buffers (`class_XXXX.png`: the surfaces' classes; `depth_XXXX.png`: their distance from the camera; `coverage_XXXX.png` and `field_depth_XXXX.png`: how much of each pixel they cover and their distance as `--field` reads them), and `dataset.yaml` giving each one's camera, light, season and time of day (needs the pumpkin of an earlier run) |
| `--style MODEL.onnx` | The days in a painted style: an image-to-image network of `scripts/neural/train_watercolor.py` (at half the frame's size, on the rendering and, if it takes 11 input channels, its classes and depth from two more renders of the frame) restyles every frame, fading in over two seconds, one second into the days |
| `--field STEM` | The days in a painted style anchored in the world: the style of `scripts/neural/train_watercolor_field.py` (`STEM.onnx` and `STEM.bin`) restyles every pixel from its colour, class, the world grids at the surface point it shows (classes, coverage and depth from three more renders of the frame) and the time of day, fading in over two seconds, one second into the days |
| `--panels DIR` | The style's training data beside the days, as `scripts/neural/prepare_style_panels.py` prepares it: a training sample with its G-buffers and target while the style fades in, then the targets of the season shown by day and by night |
| `--sequences N` | With `--dataset`: also N short sequences of 8 frames (a slow orbit step and a few minutes of the day per frame) with G-buffers, in `<out-dir>/dataset/sequences/`, e.g. for temporal stability |
| `--seed S` | Seed of the dataset's random choices (default 1) |

```bash
./build/examples/techdemo001_example
ffmpeg -framerate 30 -i build/TestingOutput/techdemo001/frames/frame_%04d.ppm -pix_fmt yuv420p techdemo001.mp4
```
### `autodiff_training_example`

Trains a fully connected network (Fourier features of the pixel coordinates,
two hidden layers of 64 ReLU units, a sigmoid output) to reproduce a
procedural grayscale image, and writes the target and snapshots of the
prediction to `build/TestingOutput/autodiff_training/`. The network is built
from the existing layers (`layers::matMul`, a bias `layers::binary` Add,
`layers::relu`) by `training::MlpTraining`. The gradients of the activations and
of the loss come from Slang's automatic differentiation (`bwd_diff` of the
`[Differentiable]` functions in `shaders/training/differentiable.slang`), the
linear layers' gradients are products of transposes, and Adam updates the
parameters. Each training step is one pre-recorded compute graph.

| Option | Meaning |
|---|---|
| `--steps N` | Training steps over the whole image (default 2000) |
| `--size S` | Image width and height in pixels (default 64) |

```bash
./build/examples/autodiff_training_example
```

### `mesh_texture_fitting_example`

Recovers the shape and texture of a synthetic pumpkin from rendered, masked
views with differentiable rasterization (in the style of nvdiffrast). Training
starts from an icosphere with a noise texture. Each training step renders one
view: the hardware rasterizer decides which triangle covers each pixel, and
Slang compute shaders recompute the shading and antialias the silhouettes as
`[Differentiable]` functions. Their gradients (from `bwd_diff`), plus a
Laplacian smoothness term, move the vertices and texels through Adam. Each step
is one pre-recorded command buffer and one queue submit, with the view chosen
on the GPU (`raster::MeshTraining`). The example prints the silhouette IoU and
PSNR of a held-out view. It writes the following to
`build/TestingOutput/mesh_texture_fitting/`: three target views, a progress
frame per epoch in `frames/` (the held-out target beside the current fit), and
the fitted mesh with its texture (`fitted.obj`, `.mtl`, `fitted_texture.ppm`)
next to the true mesh. `object.obj` holds the fitted mesh in the frame it was
trained in: its enclosing sphere is the unit sphere and up is +y. This is the
form `tree_bonsai_example` places objects in.

| Option | Meaning |
|---|---|
| `--epochs N` | Passes over all views (default 60, or 150 with `--scene`) |
| `--size S` | Image width and height in pixels (default 128) |
| `--views V` | Training views, on three rings of V/3 (default 24) |
| `--scene DIR` | Fit photographs prepared by `scripts/mesh_fitting/prepare_data.py` instead of the synthetic pumpkin |
| `--texture-width W` | Texture width in texels, height W/2 (default 64, or 128 with `--scene`) |
| `--laplacian L` | Weight of the Laplacian smoothness term (default 100) |
| `--subdivisions S` | Subdivisions of the start icosphere (default 4, or 5 with `--scene`) |
| `--view-colors` | Learn a colour gain and offset per training view (for photographs with varying exposure) |
| `--view-color-weight W` | Pull of the per-view colour corrections towards identity (default 0.01) |
| `--final-lr F` | Learning rates decay exponentially over the run to F times their start (default 0.1) |
| `--display-view N` | With `--scene`: progress frames show photo N, turned upright, beside the fit seen upright from that side (default: the first validation view) |

```bash
./build/examples/mesh_texture_fitting_example
```

With `--scene`, the example fits real photographs with masks and COLMAP
camera poses instead. It trains on the scene's training views and evaluates on
its validation views. The scene's enclosing sphere is mapped to the unit
sphere, so the default step sizes apply at any scene scale. Output goes to a
subdirectory named after the scene and includes `residuals.ppm`: four training
views as photo, fit and four times the colour error. Prepare a scene (images/, masks/ and
`splat_poses.json` with OpenGL camera-to-world poses, pinhole intrinsics and
OpenCV distortion) first:

```bash
uv run scripts/mesh_fitting/prepare_data.py SCENE_DIR build/TestingOutput/real_data/SCENE
./build/examples/mesh_texture_fitting_example --scene build/TestingOutput/real_data/SCENE
```


### `splat_training_example`

Trains 3D Gaussian splats (Kerbl et al. 2023) on photographs with masks and
COLMAP camera poses. Prepare the scene with
`scripts/mesh_fitting/prepare_data.py`, which also writes the surface points of
the visual hull carved from the masks. The Gaussians start at those points. Each
training step is one pre-recorded command buffer and one queue submit:
- project the Gaussians (with an EWA covariance);
- emit tile instances, radix sort them by tile and depth, and blend each tile
  front to back;
- compute the image loss (colour inside the mask, coverage everywhere);
- run the backward pass (back to front per pixel, then through the projection
  via Slang's `bwd_diff`);
- take an Adam step per parameter group.

Densification (clone, split, prune) runs between steps
(`splatting::SplatTraining`). The example prints silhouette IoU and PSNR on the
scene's training and validation views. It writes the following to
`build/TestingOutput/splat_training/<scene>/`: progress frames (a photo turned
upright beside the splats seen upright from its side), residuals of four
training views, and the Gaussians as `splats.ply` in scene coordinates, readable
by common splat viewers. `object.spz` holds the Gaussians in the frame they were
trained in (enclosing sphere the unit sphere, up +y), the form
`tree_bonsai_example` places objects in.

| Option | Meaning |
|---|---|
| `--scene DIR` | Scene prepared by `prepare_data.py` (required) |
| `--steps N` | Training steps (default 6000); densification runs during the first half |
| `--display-view N` | Photo the progress frames show (default: the first validation view) |

```bash
uv run scripts/mesh_fitting/prepare_data.py SCENE_DIR build/TestingOutput/real_data/SCENE
./build/examples/splat_training_example --scene build/TestingOutput/real_data/SCENE

### `tree_bonsai_example`

Renders a bonsai forest planting without a window: by default 12 trees in an oval pot,
with the captured stone lantern standing on the moss among them.
The trees follow Kuth et al., *Real-Time GPU Tree Generation* (HPG 2025): a
compute graph recorded once generates them on the GPU every frame from a few
hundred bytes of parameters per tree type. They are drawn either classically
(tessellated spline stems and Bézier leaves, with the lantern splatted and
composited by depth) or entirely as Gaussians, sorted and splatted together
with the pot and the lantern. The camera circles the pot; view *i* is seen from
the start azimuth plus 2π·*i*/*views*. Images go to
`<out-dir>/frames/<renderer>_<view>.ppm`.

| Option | Meaning |
|---|---|
| `--renderer classical\|gaussian\|both` | How the scene is drawn (default `both`) |
| `--backend raster\|compute` | Gaussian splatting backend (default `raster`) |
| `--views N` | Number of views around the pot (default 4) |
| `--scale S` | How much of the world the pot holds, from 0 (one bonsai, the lantern a 35 cm ornament) to 1 (8 x 8 km); up to about 0.3 a planting, beyond the trees follow the land cover: the nearest 24 generated in full, the others as proxies (default 0.21, a planting of 12 trees with a real lantern) |
| `--relief F` | Factor on the landscape's relief, which rises from scale 0.25 on: 1 for true proportions (1,500 m from valleys to peaks at scale 1), more for a more dramatic landscape (default 1.5) |
| `--trees N` | A planting of this many trees instead of the scale's (at most 60) |
| `--seed S` | Seed of the tree placement and the trees (default 7) |
| `--width W`, `--height H` | Image size in pixels (default 1280×720) |
| `--distance D` | Distance of the camera from the scene's focus (default 4.2) |
| `--elevation E` | Elevation of the camera in radians (default 0.32) |
| `--azimuth A` | Azimuth of the first view in radians (default 0.35) |
| `--season S` | Season from 0 (winter) over 1 (spring), 2 (summer) to 3 (fall) (default 2) |
| `--ground X,Z,M,H` | Instead of orbiting the pot, stand in the landscape at (X, Z) (units), M metres above the ground, looking along heading H (degrees from -z towards +x); views turn around that place |
| `--no-grass` | Draw no blades of grass near the camera (classical renderer, where the trees follow the placement) |
| `--budget MS` | In a video, keep the frame time near MS milliseconds by generating fewer trees in full (the others as impostors or Gaussian copies); only where the trees follow the placement |
| `--profile` | Print the GPU time of the view's frame and of its 12 longest graph elements |
| `--video` | Render the days of a video instead of views: one season per day, changing in the night, with sun, moon and wind |
| `--days N` | Number of days of the video (default 4: spring, summer, fall, winter) |
| `--seconds-per-day S` | Length of a day in the video (default 10) |
| `--fps F` | Frames per second of the video (default 30) |
| `--wind W` | Strength of the wind (default 0.5) |
| `--turns T` | Turns of the camera around the pot during the video (default 0.5) |
| `--enhancer M` | The Gaussian renderer also runs the neural enhancer of distant trees (`TreeEnhancer`, an ONNX model of `scripts/neural/train_tree_enhancer.py` at the image's size) and writes its images as `enhanced_NNNN.ppm` beside the Gaussian ones |
| `--fly-in S` | Render a flight of S seconds instead of views or days: from the first view outside the pot down to the `--ground` place at noon of the season, ending with a slow pan; takes precedence over `--video`. Without `--ground` (scale 0.6 or more), the flight goes low over a field into a dense forest and ends at 1.7 m among the trunks |
| `--gbuffers` | With `--fly-in`: also write every frame's G-buffers, `<renderer>_class_NNNN.ppm` (the surfaces' classes) and `<renderer>_depth_NNNN.ppm` (their distance from the camera as grey from 0 to a far plane that grows with the camera's height), the surface coordinates `<renderer>_u_NNNN` and `<renderer>_v_NNNN` (16 bits in red and green, a byte of the object id in blue) and the 16-bit depth `<renderer>_depth16_NNNN` (red and green, over the same window), and `<out-dir>/cameras_<renderer>.csv` with each frame's camera and depth window |
| `--from N`, `--to M` | Render only the video frames N to M - 1 |
| `--frame-stride N` | With `--fly-in`: simulate every frame but write only every N-th (e.g. for training data) |
| `--image-format ppm\|png` | Format of the fly-in's frames and G-buffers (default ppm; png is lossless and several times smaller) |
| `--moss-patches N` | Patches of the captured moss laid on the soil (default 1500) |
| `--out-dir DIR` | Output directory (default `build/TestingOutput/tree_bonsai`) |
| `--pot FILE` | A captured pot in place of the procedural one: `object.spz` of `splat_training_example` or `object.obj` of `mesh_texture_fitting_example`. It is scaled to the procedural pot's length at its rim, and the soil fills it up to just below the rim |
| `--style S` | How bonsai-like the trees are, 0 (natural) to 1 (thick, flared, S-curved trunk with surface roots, a wide crown; pines with dense pads of dark needles); default: from `--scale`, 1 at 0 fading to 0 at 0.21 |
| `--species P` | `mixed` (default: every fourth tree a pine), `pine` or `broadleaf` |
| `--tree-size F` | Factor on the trees' sizes (default 1) |
| `--tree-spread F` | Share of the pot around its middle the trees stand in (default 1) |
| `--soil-mound M` | How much the soil rises towards the middle of the pot (default 0.07) |
| `--object FILE,X,Z,HEIGHT[,YAW]` | Place a captured object (`object.spz` or `object.obj`) on the soil at (X, Z), HEIGHT units tall, turned by YAW radians; repeatable |

```bash
./build/examples/tree_bonsai_example --views 8 --width 1920 --height 1080
scripts/tree/make_bonsai_video.sh   # both renderers' videos in build/TestingOutput/tree_bonsai_video/
scripts/tree/make_bonsai_video.sh build/TestingOutput/tree_bonsai_flyin --scale 1 --fly-in 16 --azimuth -2.09 \
    --ground -1.04,0.15,1.7,120   # from outside the pot down onto a meadow
scripts/tree/make_bonsai_video.sh build/TestingOutput/tree_bonsai_forest --scale 1 --fly-in 24 \
    --azimuth -2.09               # over a field into the forest, among the trunks
```

The fitting examples' objects can join the scene. For example, with the bonsai
pot and the pumpkin trained on photographs (`splat_training_example`,
`mesh_texture_fitting_example`), this command sets the splatted pot with two
trees, a splatted pumpkin and a mesh pumpkin. Pass the mesh pot's `object.obj`
to `--pot` for the mesh version:

```bash
A=build/TestingOutput
./build/examples/tree_bonsai_example --trees 2 --species pine --style 1 --soil-mound 0.4 \
    --tree-size 1.25 --tree-spread 0.65 --distance 6.3 --elevation 0.38 \
    --pot $A/splat_training/bonsai/object.spz \
    --object $A/splat_training/pumpkin_closed_1/object.spz,-0.9,0.45,0.28,0.5 \
    --object $A/mesh_texture_fitting/pumpkin_closed_1/object.obj,0.95,-0.35,0.28,2.0
```

### `tree_bonsai_explore_example`

Walks or flies through the bonsai landscape of `tree_bonsai_example` in a window, with an ego camera. By
default it starts where that example's forest fly-in ends, standing 1.7 m above the ground deep in a forest
(scale 1).

| Input | Does |
|---|---|
| left mouse button + drag | look around |
| Tab | capture the mouse and look without a button (again: release) |
| W, A, S, D | move forwards, left, back, right |
| Shift | move ten times faster |
| scroll wheel | scale the speed |
| F | switch between walking (1.7 m above the ground) and flying |
| Space or E, Q | rise, sink (flying) |
| R | back to the start |
| Esc | release the mouse; quit when it is free |

The window's title shows the frame rate, the mode, the speed and the height above the ground. Resizing the
window rebuilds the renderer, which takes a few seconds.

On macOS, run it with the KosmicKrisp driver. With MoltenVK the window flickers black once the swapchain is
recreated, about a second after the start:

```bash
VK_DRIVER_FILES=/usr/local/share/vulkan/icd.d/libkosmickrisp_icd.json ./build/examples/tree_bonsai_explore_example
```

| Option | Meaning |
|---|---|
| `--renderer gaussian\|classical` | Which renderer draws the landscape (default `gaussian`) |
| `--scale S` | Scale of the scene as in `tree_bonsai_example` (default 1, a landscape) |
| `--seed N` | Seed of the scene (default 7) |
| `--season S` | Season in [0, 4) (default 2, summer) |
| `--width W`, `--height H` | Size of the window (default 1280 x 720) |
| `--render-scale F` | Render at F times the window's pixels and scale up (default 0.5; a high-density display has twice as many pixels as the window's size) |
| `--start forest\|pot` | Start in the forest (default at scales from 0.6 on) or outside the pot |
| `--speed M` | Speed in metres per second (default 4 in the forest, 400 outside the pot) |
| `--frames N` | Close the window after N frames |

```bash
./build/examples/tree_bonsai_explore_example                      # walk in the forest
./build/examples/tree_bonsai_explore_example --start pot          # fly in from outside the pot
```

### `tree_lod_metrics_example`

Measures a single Gaussian tree's levels of detail while the camera moves away from it and back, as the Gaussian
renderer of `tree_bonsai_example` draws it: generated every frame while it is among the near trees, beyond that as
its library copy with the copy's levels of detail by size on screen. It runs headless and renders four runs over
the same flight: a static tree (no wind) and a dynamic one (in the wind), each with the levels of detail and
without them (generated every frame at every distance without any reduction by size). With a large
`--near-radius` the tree is generated at every distance, which measures the generated trees' own levels of detail
(`TreeRenderSettings::gaussianLod`, `leafLod`).

The camera stays level with the tree's middle and moves from `--start` to `--far` metres from the trunk and back.
The logarithm of the distance follows a cosine, so the motion is smooth everywhere and slow at both ends. Two
metrics from `klartraum/lod_metrics.hpp` are taken every frame:

- **Popping** (`poppingResidual()`), for all four runs: over the lit 16 px blocks, the mean of how far a block's
  mean light lies from the mean of the same block in the two neighbouring frames. A smooth motion keeps it small,
  also where leaves slide by within a block; a sudden change of coverage or brightness puts half of its size into
  the frames on both sides. The runs without levels of detail show what the motion alone leaves.
- **Fidelity** (`fidelityError()`), of each run with levels of detail against the same run without: the mean
  absolute difference of the RGB means of 4 px blocks, over the blocks lit in either image.
- **Detail** (`detailError()`), of the same pairs: per 8 px block the energy of the luminance's Laplacian, compared
  as |log2(ratio)|. 0 means as much fine structure as the reference, 1 twice or half as much; blur and noise both
  count, colour shifts do not.

Each frame also records the GPU time of every run, from the tree's generation to its splatting, and the number of
Gaussians it draws (those opaque enough for the splatting's cull). The program prints
the popping of each run (largest, median, and where what draws the tree changes, against the run's median, the
frames around it and the run without levels of detail), and the fidelity and GPU time per representation.

| Option | Meaning |
|---|---|
| `--species broadleaf\|conifer` | Which tree (default `broadleaf`) |
| `--yaw Y` | The tree's turn about its trunk in radians (default 0.3); also picks its seed |
| `--width W`, `--height H` | Size of the measured images (default 1280 x 720, the explorer's) |
| `--fov F` | Vertical field of view in degrees (default 50, the explorer's) |
| `--start M`, `--far M` | Nearest and farthest distance in metres (default 5 and 300) |
| `--frames N` | Frames of the flight out and back (default 800) |
| `--near-radius M` | Generated nearer than this; by default the renderer's near radius at the explorer's start in the forest (0.8 times the distance of the 24th nearest tree) |
| `--wind S` | Strength of the wind in the dynamic run (default 0.5) |
| `--leaf-lod X,Y,Z,W` | `TreeRenderSettings::leafLod` of the tree with levels of detail (default the renderer's) |
| `--enhancer M` | Also run the neural enhancer of distant trees (`TreeEnhancer`, an ONNX model of `scripts/neural/train_tree_enhancer.py` at the image's size) and judge it: the four runs become the tree with levels of detail, enhanced, the classical reference (stems and leaves at twice the resolution, box-filtered) and without levels of detail, all without wind, with the fidelity against the classical reference |
| `--stem-round P`, `--stem-fade P` | `gaussianLod.x` and `.y` of the tree with levels of detail: the stem radius in pixels below which a tube is drawn round, and below which stems fade out (default the renderer's) |
| `--out-dir D` | Where `metrics.csv` and `frames/` go (default `build/TestingOutput/tree_lod_metrics`) |

`scripts/tree/make_lod_metrics_video.sh` runs it and encodes `tree_lod_metrics.mp4`: the four runs (with levels of
detail above, without below; static left, dynamic right), labelled, with the popping and fidelity of every frame
plotted below:

```bash
scripts/tree/make_lod_metrics_video.sh                     # build/TestingOutput/tree_lod_metrics/
```

### `tree_neural_dataset_example`

Renders a database of view pairs of the bonsai landscape (scale 1) for training a network that enhances the
Gaussian renderer's distant trees. For every view it writes the Gaussian renderer's image as it runs (low
fidelity) and the same view by `TreeReferenceRenderer` (high fidelity): the classical renderer with every far tree
in view generated in full, batch by batch over one depth buffer, at `--supersample` times the resolution and
box-filtered down, with the mask of the far trees' pixels. The far trees are seeded as the Gaussian renderer's,
so both images show the same trees. Views are drawn from `--seed`: at eye height on open land looking towards
forest, and 30, 120 and 400 m above the soil looking down onto forest.

| Option | Meaning |
|---|---|
| `--views N`, `--first F` | Render N views, starting at view F (to continue a database) |
| `--width W`, `--height H` | Size of the images (default 1280 x 720, the explorer's) |
| `--supersample S` | The reference's resolution factor before the box filter (default 2) |
| `--batch N` | Far trees generated per batch (default 256) |
| `--range M` | Far trees beyond M metres are left out (default 1500) |
| `--season S` | Season (default 2, summer) |
| `--seed N` | Seed of the views (default 1) |
| `--out-dir D` | Where `lr_NNNN.ppm`, its features `fa_NNNN.ppm` (coverage, tree, level of detail) and `fb_NNNN.ppm` (Gaussian size, depth), `hf_NNNN.ppm`, `mask_NNNN.pgm` and `views.csv` go (default `build/TestingOutput/neural_dataset`) |

A view takes about 2-3 s for the Gaussian image and 12-25 s for the reference.

```bash
./build/examples/tree_neural_dataset_example --views 200
### `qwen_chat_example`

An interactive chat in the terminal with Qwen3.6-27B, run by `GgufNetwork` from
a GGUF file. Download the model with `scripts/qwen_gguf/download.py`; see
[`scripts/qwen_gguf/`](https://github.com/fortmeier/klartraum/tree/main/scripts/qwen_gguf).
Qwen3.6 is third-party software under its own license and not part of
Klartraum. Each turn is formatted with the Qwen chat template and only the new
tokens are run; the conversation stays in the network's key/value caches and
linear-attention states. Replies are streamed as they are generated, thinking
(when enabled) in dim text.

Commands during the chat: `/reset` starts a new conversation, `/think on` and
`/think off` switch thinking for the next turns, `/stats` prints the context
use and the speed of the last turn, `/quit` exits.

| Option | Meaning |
|---|---|
| `--model FILE` | GGUF model (default: the `.gguf` file in `data/gguf/qwen3.6-27b/`) |
| `--system TEXT` | System prompt |
| `--prompt TEXT` | Answer one prompt and exit |
| `--think` | Start with thinking enabled |
| `--context N` | Context length in tokens (default 4096) |
| `--chunk N` | Prompt tokens per submission, 1 to 16 (default 16) |
| `--max-reply N` | Most tokens per reply (default 2048) |
| `--temperature T`, `--top-p P`, `--top-k K` | Sampling (defaults follow Qwen's recommendations: 0.7 / 0.8 / 20 without thinking, 1.0 / 0.95 / 20 with it) |
| `--greedy` | Always take the most likely token |
| `--seed N` | Random seed (default: the time) |
| `--quiet` | No loading and speed messages |
| `--profile` | Print the mean GPU time per layer kind after each reply |

```bash
./build/examples/qwen_chat_example
./build/examples/qwen_chat_example --think --system "You are a concise assistant."
./build/examples/qwen_chat_example --greedy --prompt "What is a Vulkan compute shader?"
```

### `qwen_image_example`

Text-to-image and image-to-image with [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image):
`QwenTextEncoder` turns the prompt into an embedding, `QwenImageTransformer`
denoises the latent in 8 flow-matching steps with the Lightning LoRA, and the
VAE graphs decode it. The stages run one after another, so only one large
model is on the GPU at a time (the transformer takes about 13 GB). Set up the
models with the scripts in
[`scripts/qwen_image/`](https://github.com/fortmeier/klartraum/tree/main/scripts/qwen_image);
Qwen-Image is third-party software under its own license and not part of
Klartraum. With `--control`, the depth ControlNet Union
([InstantX](https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union),
converted by `scripts/qwen_image/convert_controlnet.py`) makes the image
follow a depth map. With `--image`, the input is encoded and noised to the schedule's
noise level after skipping the first steps, so `--strength` sets how much of
it is repainted. Images are binary PPM files.

| Option | Meaning |
|---|---|
| `--prompt TEXT` | What to paint |
| `--size H W` | Image size, multiples of 16 (default 512 512); needs the VAE graphs of that size |
| `--steps N` | Denoising steps (default 8, the Lightning LoRA's) |
| `--seed N` | Seed of the initial noise (default 0) |
| `--image FILE` | Input image (PPM of the image size) for image-to-image |
| `--batch FILE` | Many images with one prompt: every line of FILE names an input (`-` for none), an output and optionally a control PPM; the models load once and every image starts from the same noise (see `--noise-per-image`) |
| `--noise-per-image` | With `--batch`, every image gets its own noise, from `--seed` and its output path |
| `--control FILE` | Control image for the ControlNet (a depth map: near white, far black), PPM of the image size |
| `--control-scale S` | The ControlNet's conditioning scale (default 1) |
| `--controlnet FILE` | The ControlNet (default: the converted ControlNet Union in `build/models/qwen-image/controlnet/`) |
| `--strength S` | Share of the steps run on the input, 0 to 1 (default 0.6) |
| `--out FILE` | Output PPM (default `build/TestingOutput/qwen_image/image.ppm`) |
| `--transformer FILE`, `--text-encoder FILE`, `--lora FILE` | Model files (defaults in `build/models/qwen-image/`) |

```bash
./build/examples/qwen_image_example --prompt "a bonsai tree in a clay pot, watercolor painting"
./build/examples/qwen_image_example --prompt "an ink drawing" --image rendering.ppm --strength 0.5 --out ink.ppm
```

### `landscape_gbuffer_example`

Estimates G-buffers of photographs or generated images with two networks run in
Klartraum: the metric depth of Depth Anything V2 Metric Outdoor Small (Apache-2.0)
and the 150 ADE20K classes of UperNet ConvNeXt-Base (MIT). Export the ONNX graphs
first with `scripts/landscape/export_depth.py` and `scripts/landscape/export_semseg.py`;
the models are third-party software under their own licenses and not part of
Klartraum. Every image (binary PPM of any size) is resized to each network's input
and its outputs are scaled back. Per image it writes `<name>_depth.f32` (metres,
float32), `<name>_depth.pgm` (centimetres, 16 bit), `<name>_class.pgm` (class
indices) and `<name>_class.ppm` (the classes in colour).

| Option | Meaning |
|---|---|
| `--out-dir DIR` | Output directory (default `build/TestingOutput/landscape_gbuffers`) |
| `--depth-model FILE` | Depth graph (default: the 518 x 924 export in `build/models/depth/onnx/`) |
| `--segmentation-model FILE` | Segmentation graph (default: the 512 x 896 export in `build/models/semseg/onnx/`) |

```bash
./build/examples/landscape_gbuffer_example --out-dir build/TestingOutput/landscape_gbuffers photo1.ppm photo2.ppm
```

### `photo_enhancer_example`

Renders the forest fly-in of `tree_bonsai_example` (`--fly-in`, the classical
renderer at 1280 x 720) and turns every frame into a photograph-like image on the
GPU. The frame's G-buffers (colour, classes, 16-bit depth and the surface
coordinates u, v and object id) are downsampled to the 832 x 464 sample size of
training. `neural::hashTexture()` looks up a learnable surface texture there: one
multi-resolution 3D hash grid (Instant NGP) over (u, v, id) per surface class
(soil, stem and leaf). The photo enhancer's generator, an ONNX graph, then turns the
G-buffers and texture features into the image. Train the model with
`scripts/neural/train_photo_enhancer.py` (with its texture option) on Qwen-Image photographs of
the flights, which also writes the texture next to the ONNX graph
(`<stem>.onnx`, `<stem>_texture.txt`, `<stem>_texture_f32.bin`). Frames go to
`<out-dir>/frames/frame_NNNN.png` (enhanced) and `render_NNNN.png` (the
downsampled rendering), as PNG. A model trained with the season conditioning
(`<stem>_season_f32.bin` next to the graph) also gets the season of `--season`.

| Option | Meaning |
|---|---|
| `--model STEM` | The exported model (default `build/TestingOutput/gan_runs/v6/photo_enhancer`) |
| `--out-dir DIR` | Output directory (default `build/TestingOutput/photo_enhancer`) |
| `--seed N`, `--azimuth A`, `--season S` | The scene and the flight's start, as in `tree_bonsai_example` (defaults 7, -2.09, 2) |
| `--fly-in SECONDS` | Length of the flight at 30 frames per second (default 24) |
| `--every N`, `--from F`, `--to T` | Enhance every N-th frame of frames F to T (all frames are simulated) |

```bash
./build/examples/photo_enhancer_example --out-dir build/TestingOutput/photo_enhancer
ffmpeg -framerate 30 -i build/TestingOutput/photo_enhancer/frames/frame_%04d.png -pix_fmt yuv420p enhanced.mp4
```

### `crate_gbuffer_example`

A toy scene for the photo enhancer pipeline: a crate (a box whose six faces each
have their own surface coordinates and object id) standing on a floor, drawn by
`GBufferMeshRenderer` from random cameras around it (any direction, 8 to 40
degrees above, 2.2 to 3.8 away). Every view is written with all G-buffers like a
frame of the `tree_bonsai_example` fly-ins (`frames/classical_*.png`, the class,
depth, 16-bit depth and u and v images, and `cameras_classical.csv`), so that
`scripts/neural/prepare_photo_pairs.py` and `add_surface_coordinates.py` (with
their sample size option) turn the views into training pairs. With `--model`, a
trained photo enhancer renders every view, entirely in Klartraum.

| Option | Meaning |
|---|---|
| `--views N` | Number of views (default 180) |
| `--size S` | Width and height of the images (default 512) |
| `--seed N` | Seed of the cameras (default 1) |
| `--out-dir DIR` | Output directory (default `build/TestingOutput/crate/flights/crate_a`) |
| `--orbit` | Circle the crate once instead (22 degrees above, 3 away) |
| `--model STEM` | Also enhance every view with the photo enhancer `STEM` (`neural::PhotoEnhancer`) into `frames/enhanced_NNNN.png` |
| `--appearance K` | The appearance code of training image K (default: the codes' mean) |
| `--sample-size S` | The enhancer's sample size (default 256) |

```bash
./build/examples/crate_gbuffer_example --views 180 --seed 1 --out-dir build/TestingOutput/crate/flights/crate_a
./build/examples/crate_gbuffer_example --orbit --views 240 --out-dir build/TestingOutput/crate/orbit \
    --model build/TestingOutput/crate/runs/c1/photo_enhancer_best --appearance 3
```

### `crate_training_example`

Trains the photo enhancer of the crate toy scene entirely in Klartraum with
`neural::PhotoEnhancerTraining`: the views of the crate flights of
`crate_gbuffer_example` are rendered again from their cameras and prepared as
samples (`neural::photoEnhancerInputs()`), their Qwen-Image photos
(`<crate>/data/<split>/photo/<view>.ppm`) are the targets, and the learnable
surface texture, the local U-Net generator (3 levels, ChannelNorm, ELU) and one
appearance code per view are trained with the photometric losses (L1 and L1 of
8 x 8 block means), optionally the VGG perceptual loss and a PatchGAN, on random
crops, with Adam and an exponential moving average.
Views with an alignment score below the threshold (`alignment.json` of
`scripts/neural/check_photo_alignment.py`) or more than 60 % background are left
out. Every `--every` steps the averaged parameters go to `<out>/parameters/` and
the held-out views are rendered with the codes' mean
(`<out>/validation_NNNNNN.png`: rendering, photo, enhanced; the L1 in
`<out>/log.txt`).

| Option | Meaning |
|---|---|
| `--crate DIR` | The crate data (default `build/TestingOutput/crate`: `flights/crate_a`, `flights/crate_v`, `data/train`, `data/validation`) |
| `--out DIR` | Output directory (default `build/TestingOutput/crate/runs/k1`) |
| `--steps N` | Training steps (default 30000) |
| `--batch N`, `--crop S` | Crops per step and their size (default 16 and 128) |
| `--appearance D` | Size of the appearance codes (default 8, 0 for none) |
| `--every N` | Steps between checkpoints and validations (default 1000) |
| `--min-alignment A` | Leave out views aligned worse (default 0.7) |
| `--profile` | Print the GPU time of a step per element kind every 100 steps |
| `--perceptual W` | Weight of the VGG-16 perceptual loss (default 0; needs `scripts/neural/export_vgg16.py`) |
| `--adversarial W` | Weight of the adversarial loss of a 2-scale PatchGAN with DiffAugment (default 0: no GAN) |
| `--feature-matching W` | Weight of the discriminator's feature matching (default 2) |
| `--init-from DIR` | Start generator, texture and codes from an earlier run's `parameters/` |

```bash
./build/examples/crate_training_example --crate build/TestingOutput/crate --out build/TestingOutput/crate/runs/k1
```
