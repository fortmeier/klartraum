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
next to the true mesh.

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
by common splat viewers.

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
| `--fly-in S` | Render a flight of S seconds instead of views or days: from the first view outside the pot down to the `--ground` place at noon of the season, ending with a slow pan; takes precedence over `--video`. Without `--ground` (scale 0.6 or more), the flight goes low over a field into a dense forest and ends at 1.7 m among the trunks |
| `--from N`, `--to M` | Render only the video frames N to M - 1 |
| `--moss-patches N` | Patches of the captured moss laid on the soil (default 1500) |
| `--out-dir DIR` | Output directory (default `build/TestingOutput/tree_bonsai`) |

```bash
./build/examples/tree_bonsai_example --views 8 --width 1920 --height 1080
scripts/tree/make_bonsai_video.sh   # both renderers' videos in build/TestingOutput/tree_bonsai_video/
scripts/tree/make_bonsai_video.sh build/TestingOutput/tree_bonsai_flyin --scale 1 --fly-in 16 --azimuth -2.09 \
    --ground -1.04,0.15,1.7,120   # from outside the pot down onto a meadow
scripts/tree/make_bonsai_video.sh build/TestingOutput/tree_bonsai_forest --scale 1 --fly-in 24 \
    --azimuth -2.09               # over a field into the forest, among the trunks
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

### `tree_lod_popping_example`

Measures how much a single Gaussian tree pops while the camera moves away from it and back, as the Gaussian
renderer of `tree_bonsai_example` draws it: generated every frame while it is among the near trees, beyond that as
its library copy with the copy's levels of detail by size on screen. It runs headless and renders two runs over the
same flight side by side: a static tree (no wind) and a dynamic one (in the wind).

The camera stays level with the tree's middle and moves from `--start` to `--far` metres from the trunk and back.
The logarithm of the distance follows a cosine, so the motion is smooth everywhere and slow at both ends. Every
frame gets a residual (`poppingResidual()` in `klartraum/popping_metric.hpp`): over the lit 16 px blocks, the mean
of how far a block's mean light lies from the mean of the same block in the two neighbouring frames. A smooth
motion keeps it small, also where leaves slide by within a block; a sudden change of coverage or brightness puts
half of its size into the frames on both sides. The program prints the largest, median and mean residual of each
run, and the residual where what draws the tree changes, against the run's median and against the frames around it.

| Option | Meaning |
|---|---|
| `--species broadleaf\|conifer` | Which tree (default `broadleaf`) |
| `--yaw Y` | The tree's turn about its trunk in radians (default 0.3); also picks its seed |
| `--width W`, `--height H` | Size of the measured images (default 1280 x 720, the explorer's) |
| `--fov F` | Vertical field of view in degrees (default 50, the explorer's) |
| `--start M`, `--far M` | Nearest and farthest distance in metres (default 5 and 300) |
| `--frames N` | Frames of the flight out and back (default 800) |
| `--near-radius M` | Generated nearer than this; by default the distance of the 24th nearest tree at the explorer's start in the forest |
| `--wind S` | Strength of the wind in the dynamic run (default 0.5) |
| `--out-dir D` | Where `popping.csv` and `frames/` go (default `build/TestingOutput/tree_lod_popping`) |

`scripts/tree/make_lod_popping_video.sh` runs it and encodes `tree_lod_popping.mp4` with labels and the residuals
of both runs below the frames:

```bash
scripts/tree/make_lod_popping_video.sh                     # build/TestingOutput/tree_lod_popping/
```
