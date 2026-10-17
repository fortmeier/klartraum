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
| `sd15_vae_example` | The Stable Diffusion 1.5 VAE encoder and decoder run as ONNX networks | no |
| `sd15_denoiser_example` | The complete Stable Diffusion 1.5 text-to-image pipeline run as ONNX networks | no |
| `openpose_skeleton_example` | A 3D skeleton drawn as ControlNet OpenPose pose maps from a camera circling it | no |
| `pose_to_video_example` | Neural character animation turned into a photorealistic video with ControlNet-OpenPose and SD1.5 | no |
| `lantern_walk_example` | A neural animated character walking around the lantern scene, as an OpenPose skeleton with inpainting masks | no |
| `techdemo000_example` | "klartraum tech demo 000": the promo video in one continuous shot around the lantern, chapter by chapter with captions | no |

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

### `sd15_vae_example`

Encodes `lantern_input_f32.bin` with the Stable Diffusion 1.5 VAE encoder and
decodes the latent again, both run as ONNX networks without a window, and
writes the decoded image. The models are third-party and are not part of
Klartraum: export them first with `scripts/sd15_onnx` (see its README).

| Option | Meaning |
|---|---|
| `--size N` | Image size the models were exported for (default 64) |
| `--model-dir DIR` | Directory of the exported models (default `data/onnx/sd15`) |
| `--output PATH` | Output image (default `build/TestingOutput/sd15_vae_klartraum.ppm`) |
| `--load-only` | Only parse both graphs, without allocating or running them |

```bash
./build/examples/sd15_vae_example --size 64 --model-dir data/onnx/sd15
```

### `sd15_denoiser_example`

Runs the complete Stable Diffusion 1.5 pipeline without a window: CLIP text
encoder, UNet denoising with classifier-free guidance and DDIM steps, and the
VAE decoder, all as ONNX networks. For the prompt the models were exported
with, it checks the results against the Python reference. The models are
third-party and are not part of Klartraum: export them first with
`scripts/sd15_onnx` (see its README).

| Option | Meaning |
|---|---|
| `--size N` | Image size the models were exported for (default 256) |
| `--model-dir DIR` | Directory of the exported models (default `data/onnx/sd15_denoiser_256`) |
| `--prompt TEXT`, `--negative-prompt TEXT` | Prompts; others than the exported ones skip the reference checks |
| `--output PATH` | Output image (default `build/TestingOutput/sd15_pipeline_klartraum.ppm`) |
| `--max-denoise-steps N` | Stop after N scheduler steps |
| `--skip-decoder` | Do not run the VAE decoder |
| `--profile` | Print GPU timings per operation |

```bash
./build/examples/sd15_denoiser_example --prompt "a red sports car parked on a city street"
### `openpose_skeleton_example`

Draws a fixed 3D standing pose with `OpenPoseSkeletonRenderer` without a
window, from a camera that circles it once, and writes one pose map per frame
(`frame_0000.ppm`, `frame_0001.ppm`, …). The pose maps follow the drawing
convention of ControlNet's OpenPose annotator, so they can condition a
ControlNet-OpenPose model. The first frame looks at the front of the skeleton.

| Option | Meaning |
|---|---|
| `--frames N` | Number of frames in one turn (default 36) |
| `--width W`, `--height H` | Image size in pixels (default 512×512) |
| `--distance D` | Distance of the camera from the hips in metres (default 3.2) |
| `--out-dir DIR` | Output directory (default `build/TestingOutput/openpose_skeleton`) |

```bash
./build/examples/openpose_skeleton_example --frames 36 --out-dir build/TestingOutput/openpose_skeleton
```

### `pose_to_video_example`

Walks a character with the neural character controller (`CharacterController`)
along a scripted path, draws each frame as an OpenPose pose map, and turns the
frames into a photorealistic video with SD1.5, ControlNet-OpenPose and four
LCM steps, all without a window (`PoseVideoPipeline`). The character stands at
first, walks forward and then turns right; the camera follows it. Frames are
sampled as img2img variations of one keyframe with shared noise
(`TemporalLcmSampler`), which keeps appearance and background stable from frame
to frame and needs three instead of four steps per frame. It writes
`pose_XXXX.ppm` and `frame_XXXX.ppm` per frame and prints the mean difference
between consecutive frames.

The SD1.5, ControlNet and LCM-LoRA models are third-party and not part of
Klartraum: export them first with `scripts/sd15_onnx` (`export_denoiser.py` and
`export_controlnet.py` with the same `--size`, see its README). Use 512 pixels:
SD1.5 and the ControlNet were trained at that size, and at 256 pixels faces and
backgrounds fall apart (the tests use 256 only to stay fast). The controller in
`data/character` comes with the repository.

| Option | Meaning |
|---|---|
| `--size N` | Image size the models were exported for (default 512) |
| `--frames N` | Number of frames (default 50) |
| `--controller-steps N` | Controller steps (at 30 per second) per frame (default 2) |
| `--prompt TEXT` | What the video shows |
| `--style NAME` | Walking style of the controller, e.g. `Neutral`, `Proud`, `Old` (default `Neutral`) |
| `--speed V` | Walking speed in m/s (default 1.3) |
| `--start-step K` | LCM step at which each frame starts from the noised keyframe (default 1 of 4) |
| `--anchor W` | Weight of the keyframe against the previous frame for a frame's start (default 1) |
| `--independent` | Sample every frame from noise instead, for comparison |
| `--character-dir DIR`, `--sd-dir DIR`, `--control-dir DIR` | Model directories (defaults `data/character`, `data/onnx/sd15_denoiser_<size>`, `data/onnx/sd15_controlnet_<size>`) |
| `--out-dir DIR` | Output directory (default `build/TestingOutput/pose_to_video`) |

```bash
./build/examples/pose_to_video_example --size 512 --frames 50 --out-dir build/TestingOutput/pose_to_video
```

### `lantern_walk_example`

Places the neural character controller (`CharacterController`) in the lantern
scene and lets it walk once around the lantern, without a window. The lantern
is measured from its Gaussians: the character is scaled to about half the
lantern's height and walks on the moss, between the lantern's footprint and the
edge of the moss (`measureUprightObject`, `CirclePathFollower`). The splatting
writes its depth (`GsplatConfig::outputDepth`), so the skeleton and the
character mask are cut where the lantern is in front of the character.

Per frame it writes `frame_XXXX.ppm` (scene with skeleton), `pose_XXXX.ppm`
(the OpenPose pose map), `character_mask_XXXX.ppm` (the region of the body,
`CharacterMaskRenderer`), `scene_mask_XXXX.ppm` (where the scene has a surface,
`createSceneMask`) and `panels_XXXX.ppm`: these four in a 2 × 2 grid, with both
masks in the last panel (scene blue-grey, character white).

| Option | Meaning |
|---|---|
| `--file PATH` | Scene (default `data/lantern.spz`, read with a flipped y axis) |
| `--size N` | Image size in pixels (default 512) |
| `--controller-steps N` | Controller steps (at 30 per second) per written frame (default 2) |
| `--relative-height F` | Character height as a fraction of the lantern's (default 0.5) |
| `--path-position F` | Path between the lantern's footprint (0) and the edge of the moss (1) (default 0.5) |
| `--speed V` | Walking speed in m/s (default 1.2) |
| `--style NAME` | Walking style of the controller (default `Neutral`) |
| `--camera-position X Y Z`, `--distance D` | Camera direction and distance from the lantern at the start |
| `--turns F` | How many times the character walks around the lantern (default 1) |
| `--orbit-degrees A` | How far the camera orbits around the lantern over the walk (default 20) |
| `--push-in F` | Fraction of its distance the camera moves closer over the walk (default 0.12) |
| `--character-dir DIR` | Controller model (default `data/character`) |
| `--out-dir DIR` | Output directory (default `build/TestingOutput/lantern_walk`) |
| `--generate` | Also render the person with SD1.5, ControlNet-OpenPose and LCM-LoRA into the scene (see below) |
| `--prompt TEXT` | What the generated person looks like (default: a woman in a dark blue kimono in a moss garden) |
| `--seed N` | Seed of the noise the person is generated from (default 5) |
| `--paste-margin F` | Margin of the mask the close-up is pasted with, around the body (default 0) |
| `--ring-weight F` | How freely SD may repaint the margin around the body (1: freely, 0: keep the scene; default 0.5) |
| `--latent-composite` | Combine scene and person only in latent space: the close-up is the decoded latent as a whole, not pasted into the scene render |
| `--sd-dir DIR`, `--vae-dir DIR`, `--control-dir DIR` | Model directories (defaults `data/onnx/sd15_denoiser_<size>`, `data/onnx/sd15_<size>`, `data/onnx/sd15_controlnet_<size>`) |

With `--generate`, every written frame also gets `generated_XXXX.ppm`, the scene
with a generated person, `closeup_XXXX.ppm`, the close-up the person was
generated in, and `generated_panels_XXXX.ppm`: the generated frame, the
close-up, its pose map and its masks (pasted region white, repainted margin
grey) in a 2 × 2 grid. The person is too small in the full view for SD1.5, so a crop
camera (`cropProjection`) renders the scene, its depth, the pose map and the
character mask of a window around the character at full resolution. The
close-up is inpainted inside the character mask by latent blending
(`sampleInpainted`: every step keeps the scene's latent outside the mask), and
pasted into the scene render with a feathered mask, so the lantern stays exactly
as rendered, also where it is in front of the person. All frames use the same
noise; the character stays centred in the close-up, so it meets the same noise
every frame. Generation takes about 8 s per frame at 512 pixels on an Apple M4.
It needs the third-party models exported by `scripts/sd15_onnx` for the same
size: `export_denoiser.py` (CLIP), `export_onnx.py` (VAE encoder and decoder,
into `data/onnx/sd15_<size>`) and `export_controlnet.py`.

```bash
./build/examples/lantern_walk_example --out-dir build/TestingOutput/lantern_walk
ffmpeg -framerate 15 -i build/TestingOutput/lantern_walk/panels_%04d.ppm -pix_fmt yuv420p lantern_walk_panels.mp4

./build/examples/lantern_walk_example --generate --out-dir build/TestingOutput/lantern_generate
ffmpeg -framerate 15 -i build/TestingOutput/lantern_generate/generated_%04d.ppm -pix_fmt yuv420p lantern_person.mp4
```

### `techdemo000_example`

Renders the promo video "klartraum tech demo 000" without a window: one
continuous shot that circles the lantern, in chapters that each add a feature.
So far: (1) the title made of the lantern's Gaussians holds still and morphs
into the scene (`MorphVariant::SwirlScatter`), (2) the captured scene with a
slow push-in, (3) dynamic lighting (`GaussianLighting`): dusk falls, a
candle flickers and glows in the lantern's fire box, then a red, a green and a
blue light circle the scene, (4) the neural character controller (`CharacterController`)
walking a plain skeleton in from the side and around the lantern
(`WalkInPathFollower`), drawn as Gaussians merged with the scene
(`SkeletonGaussians`, `GaussianMerge`), so the lantern hides it (with
`--mesh-skeleton` as rasterized cylinders and spheres over the scene instead,
`SkeletonMeshRenderer`, hidden by the splatting's depth); it runs in and
slows to a walk, and the camera turns four times slower from here on. (5) The
skeleton turns into an OpenPose pose map over the scene. (6) SD1.5 with
ControlNet-OpenPose and LCM-LoRA inpaints a person over the skeleton, as
`lantern_walk_example` does when it generates, in a square close-up cut from the wide
frame (about 8 s per frame); the pose map ControlNet sees is shown picture in
picture. The whole frame runs on the GPU as one `GraphChain`: the close-up window and
cameras, the VAE encoder, four times ControlNet + UNet each followed by an LCM step,
the VAE decoder and the paste into the wide frame, without the CPU in between. With `--placeholder` the character mask stands in for the person, to
work on the timing quickly. (7) The scene's Gaussians explode and the picture
fades to black.
The character starts walking with chapter 4; when only later chapters are
rendered, its steps since then still run, without writing frames. Each chapter
shows a caption with the papers it builds on, lower left over a darkened corner
(Unbounded and JetBrains Mono, `blendCaption`). The timing is a `DemoTimeline`:
the camera starts orbiting with the morph and keeps one turn per
`--seconds-per-turn` until the character appears.

Frames go to `<out-dir>/frames/frame_XXXX.ppm`, numbered through the whole video,
so `--from`/`--to` re-render chapters into the same sequence. The program checks
that the title frames are identical and that the first frame after the morph
equals the scene rendered directly; otherwise it exits with an error.

| Option | Meaning |
|---|---|
| `--from N`, `--to N` | First and last chapter to render (default: all) |
| `--step N` | Render only every Nth frame, for a quick preview (default 1) |
| `--placeholder` | Show the character mask instead of generating the person in chapter 6 |
| `--mesh-skeleton` | Draw the skeleton in chapters 4 and 5 as rasterized cylinders and spheres instead of Gaussians |
| `--prompt TEXT`, `--seed N` | What the generated person looks like, and the seed of its noise (default: a woman in a dark blue kimono, 5) |
| `--sd-path gpu\|cpu\|both` | How chapter 6 generates the person: entirely on the GPU, one `GraphChain` per frame (default), with the CPU between the graphs, or both, writing the GPU path's frames to `<out-dir>/frames_gpu` and the times and differences of both paths to `<out-dir>/sd_compare.csv` |
| `--backend raster\|compute` | The Gaussian splatting backend of every graph (default: raster) |
| `--dump-graphs DIR` | Record GPU timestamps in the graphs of the SD1.5 chain and write them, with each element's mean time, as JSON to `DIR` at the end (for the graph pictures of `techdemo000_graphs`) |
| `--width W`, `--height H` | Image size in pixels (default 1920×1080); captions scale with the height |
| `--fps F` | Frame rate (default 30) |
| `--seconds-per-turn S` | Time for one camera turn around the lantern (default 15) |
| `--file PATH` | Scene (default `data/lantern.spz`, read with a flipped y axis) |
| `--out-dir DIR` | Output directory (default `build/TestingOutput/techdemo000`) |

```bash
./build/examples/techdemo000_example
ffmpeg -framerate 30 -i build/TestingOutput/techdemo000/frames/frame_%04d.ppm -c:v libx264 -crf 16 \
    -pix_fmt yuv420p build/TestingOutput/techdemo000/techdemo000.mp4
```

### `techdemo000_graphs`

Builds the compute graphs of the tech demo's chapters 2 to 4 as `techdemo000_example`
does (the splatting, the relighting with three lights, the controller walking a
skeleton of Gaussians next to the lantern), runs each with GPU timestamps for a few
warm-up frames and then `--frames` frames while the camera circles the scene, and
writes it with each element's mean GPU time as JSON (`klartraum::writeGraphJson()`).
The SD1.5 chain of chapter 6 comes from `techdemo000_example` with its option to dump graphs.
`scripts/techdemo000/graph_dispatches.py` counts the commands every element records
(with the api_dump layer), `scripts/techdemo000/draw_graphs.py` draws each graph as a
screen of at most twelve blocks with their dispatches, draws and GPU times, and
`scripts/techdemo000/compose_chapter8.py` turns the screens into the demo's chapter 8.

| Option | Meaning |
|---|---|
| `--graph NAME` | Only `gaussian_splatting`, `dynamic_lighting` or `neural_animation` (default: all three) |
| `--backend raster\|compute` | The Gaussian splatting backend (default: raster) |
| `--frames N` | Measured frames after the warm-up (default 50) |
| `--file PATH` | Scene (default `data/lantern.spz`) |
| `--out-dir DIR` | Output directory (default `build/TestingOutput/techdemo000_graphs`) |

```bash
./build/examples/techdemo000_graphs
python3 scripts/techdemo000/graph_dispatches.py gaussian_splatting
```

The SD1.5 chain and chapter 8 are described in the docstrings of the scripts.
