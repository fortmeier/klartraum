# Lantern-scene Gaussian Splatting experiment

This experiment creates its own uv-managed Python 3.10 environment, estimates
camera poses and a sparse point cloud with COLMAP, trains Nerfstudio's
`splatfacto` model, and exports the reconstruction as a Gaussian-splat PLY.

## Prerequisites

- `uv`
- An NVIDIA GPU and driver
- A CUDA 12.x Toolkit (including `nvcc`); PyTorch uses its CUDA 12.8 build
- COLMAP (the runner automatically finds the local portable installation at
  `C:\Users\dfort\Desktop\tools\colmap-x64-windows-cuda\COLMAP.bat`)

The runner automatically activates the installed Visual Studio 2022 Community
x64 compiler environment. Nerfstudio's CUDA extensions compile on first use.
FFmpeg is supplied inside the uv environment through `imageio-ffmpeg`.
The included COLMAP compatibility launcher maps Nerfstudio's legacy SIFT GPU
flags to their COLMAP 4.x names.

## Run

From the repository root:

```powershell
py scripts/lantern_splat/run_experiment.py
```

The command is resumable: stages whose expected output exists are reused. Run a
specific stage or repeat one explicitly with:

```powershell
py scripts/lantern_splat/run_experiment.py --stage process
py scripts/lantern_splat/run_experiment.py --stage process --colmap C:\path\to\COLMAP.bat
py scripts/lantern_splat/run_experiment.py --stage train
py scripts/lantern_splat/run_experiment.py --stage train --no-viewer
py scripts/lantern_splat/run_experiment.py --stage export
py scripts/lantern_splat/run_experiment.py --stage train --force
```

For a quick pipeline check, reduce the training length, for example with
`--iterations 100`. The production default is 30,000 iterations.

All regenerable files are written below the ignored `build/lantern_splat/`
directory:

- `processed/` contains the Nerfstudio dataset and COLMAP reconstruction.
- `outputs/` contains checkpoints and the run's `config.yml`.
- `export/splat.ply` is the exported Gaussian splat.

Training starts Nerfstudio's browser viewer by default. Pass `--no-viewer` only
for a non-interactive TensorBoard run.
