# Latent neural textures

Train a renderer that turns a toy scene's G-buffers into photos, with the
appearance of each object held in a learned texture and selected by a latent
code. Everything trains in Klartraum (`neural::PhotoEnhancerTraining`); the
PyTorch scripts in `scripts/neural/` are only the references of the unit tests.

The setup is the crate toy scene: one box geometry, many appearances.

- [A. Make consistent photos](#a-make-consistent-photos) of many crates.
- [B. Train one model for all crates](#b-train-one-model-for-all-crates).
- [C. Render and evaluate](#c-render-and-evaluate) turntables and code morphs.
- [Results](#results) of the experiments so far.

## A. Make consistent photos

A photo-realistic target needs the same crate in every view. `crate_paint_example`
paints a UV atlas of the crate's six faces photo by photo, as TEXTure paints
meshes: each view's start image for Qwen-Image image-to-image shows the atlas
where it is painted, and the finished photo is baked back into the atlas.

Per crate the driver paints six views:

1. A diagonal view, freely from the rendering (strength 0.75).
2. The opposite diagonal (0.65), which paints the two other sides.
3. Four random views from the atlas (0.4), each baked in.
4. View 2 again from the finished atlas (0.4), so that its lid agrees.

Bake with `--margin 0.2`: a view takes over a painted texel only if it sees it
clearly more frontally. Two views that see the lid at the same angle would
otherwise split it along its diagonal.

`crate_paint_example --consistency FLIGHT PHOTOS` measures how well the photos
of a crate agree: each photo warped onto its nearest other view by the flow from
the G-buffers, mean squared difference on the crate.

## B. Train one model for all crates

```bash
./build/examples/crate_training_example --crates DIR --held-out-crates 10 \
    --texture-only-latent --latent 16 --texture-basis 8 16,32,64,128 \
    --widths 32,64,128,256,256,256 --sample-size 128 --crop 96 --batch 16 \
    --perceptual 1 --temporal 10 --no-floor --min-alignment 0 --steps 10000
```

- `--crates DIR` reads every `DIR/crates/<id>/` (cameras and photos). Every
  crate gets its own row of a learned latent table: an auto-decoder, with no
  encoder.
- `--texture-only-latent` keeps the code out of the generator. Every difference
  between crates then has to go through the texture.
- The texture is one of:
  - `--texture-mlp F W K`: a latent-conditioned MLP on the positional encoding
    of (u, v, face). Add `--texture-mlp-residual` for finer residual blocks.
  - `--texture-basis C R1,R2,...`: feature images per face and resolution,
    with a base layer plus one layer per code number, mixed by the code. This
    is Deferred Neural Rendering's neural texture made multidimensional.
  - `--basis-decay` and `--basis-smoothness` regularise the finer levels.
- After training, the codes of the held-out crates are fitted on three of their
  views with everything else fixed (`--fit-steps`). Their other three views
  measure how well a new crate is reproduced.

## C. Render and evaluate

The run writes:

| Output | Content |
|---|---|
| `validation_*.png` | Held-out views: photo, the crate's own code, the mean code |
| `crates_*.png` | 16 crates on the same view |
| `new_crates.png` | Held-out crates: photo, fitted code, mean code |
| `crate_codes.txt` | One crate's code per line |

To render a turntable of a crate with its code, run
`crate_gbuffer_example --orbit --parameters RUN/parameters --latent-code "<line>"`.
`--latent-codes FILE` renders one code per view, for a morph through the codes.

## Results

100 crates, 6 views each, painted at 256 x 256 and trained at 128 x 128. The
10 last crates are held out. After the clean-up described below, 465 training
views and 13 validation views remain. All runs: 16-number codes, 10,000 steps.
The lower an L1, the better.

| Run | Texture | Held-out views | Training views | New crates (3 views) |
|---|---|---:|---:|---:|
| b0 | MLP, 6 octaves, width 64 | 0.047 | 0.030 | 0.051 |
| tA | Feature images 16-128, 8 channels | 0.047 | 0.010 | 0.048 |
| tB | Residual MLP, 4 + 6 + 8 octaves | 0.045 | 0.030 | 0.049 |

- The code carries the crate. Every run halves the error compared with the
  training crates' mean code (about 0.080), even though the generator never sees
  the code.
- Only the feature images keep fine detail: rope handles, corner brackets, iron
  bands, slats. Both MLP textures stay smooth.
- The feature images memorise. Their training error is far below the held-out
  error, unseen views show a noisy grain, and the turntables flicker more:
  silhouette jitter 0.002-0.006 against 0.0002-0.0009 for the MLP textures.
- Regularising the finer levels (`--basis-smoothness 1e-7`) was stopped at
  step 4,000: held-out 0.051 and training 0.033. That is a smaller gap than
  tA's, but it isn't evaluated yet.

Data:

- Mean photo consistency is 0.029, against 0.027 for the single painted crate.
- 61 photos (10 %) are left out because they disagree with their own atlas
  rendering or their start image: open boxes, cut-out holes, invented symbols.
- Open boxes that were baked into an atlas are not caught: there the photo and
  the atlas agree.
