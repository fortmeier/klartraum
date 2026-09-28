"""Process, train, and export the lantern-scene Gaussian splat."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_DIR.parents[1]
SOURCE_IMAGES = REPOSITORY_ROOT / "data" / "lantern_scene"
RUN_ROOT = REPOSITORY_ROOT / "build" / "lantern_splat"
PROCESSED_DATA = RUN_ROOT / "processed"
TRAINING_OUTPUT = RUN_ROOT / "outputs"
EXPORT_OUTPUT = RUN_ROOT / "export"
CONFIG_PATH = TRAINING_OUTPUT / "lantern-scene" / "splatfacto" / "run" / "config.yml"
SPLAT_PATH = EXPORT_OUTPUT / "splat.ply"
MODEL_DIR = CONFIG_PATH.parent / "nerfstudio_models"
LOCAL_COLMAP = Path(r"C:\Users\dfort\Desktop\tools\colmap-x64-windows-cuda\COLMAP.bat")
LOCAL_VCVARS = Path(
    r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
)

# Nerfstudio's Rich output contains emoji that the legacy Windows code page cannot encode.
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an isolated uv environment and reconstruct data/lantern_scene with Nerfstudio Splatfacto."
    )
    parser.add_argument(
        "--colmap",
        type=Path,
        help="Path to COLMAP.bat or colmap.exe; defaults to PATH, then the local portable COLMAP installation.",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "setup", "process", "train", "export"),
        default="all",
        help="Run the complete pipeline or one stage (default: all).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=30_000,
        help="Number of Splatfacto training iterations (default: 30000).",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_false",
        dest="viewer",
        help="Disable Nerfstudio's web viewer and use non-interactive TensorBoard output.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run the selected stage even when its output already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without creating the environment or reconstruction.",
    )
    return parser.parse_args()


def run(command: list[str], *, dry_run: bool, environment: dict[str, str] | None = None) -> None:
    print("+", subprocess.list2cmdline(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True, env=environment)


def uv_command(*arguments: str) -> list[str]:
    return ["uv", "--project", str(EXPERIMENT_DIR), *arguments]


def uv_run(*arguments: str) -> list[str]:
    return uv_command("run", "--locked", "--no-sync", *arguments)


def require_executable(name: str, explanation: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Missing required executable '{name}'. {explanation}")


def setup_environment(*, dry_run: bool) -> None:
    if not dry_run:
        require_executable("uv", "Install uv from https://docs.astral.sh/uv/getting-started/installation/.")
    run(uv_command("sync", "--locked"), dry_run=dry_run)


def find_colmap(explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        colmap_path = explicit_path.expanduser().resolve()
    elif path_entry := shutil.which("colmap"):
        colmap_path = Path(path_entry)
    else:
        colmap_path = LOCAL_COLMAP
    if not colmap_path.is_file():
        raise SystemExit("COLMAP was not found. Pass its executable or batch launcher with --colmap.")
    return colmap_path


def cuda_build_environment() -> dict[str, str]:
    environment = os.environ.copy()
    if shutil.which("cl", path=environment.get("PATH")) is None:
        if not LOCAL_VCVARS.is_file():
            raise SystemExit(
                "The MSVC compiler is not active and vcvars64.bat was not found. "
                "Run from a Visual Studio x64 Developer PowerShell."
            )
        result = subprocess.run(
            f'call "{LOCAL_VCVARS}" >nul && set',
            check=True,
            capture_output=True,
            text=True,
            errors="replace",
            shell=True,
        )
        environment.update(
            line.split("=", 1)
            for line in result.stdout.splitlines()
            if "=" in line and not line.startswith("=")
        )

    if shutil.which("cl", path=environment.get("PATH")) is None:
        raise SystemExit("MSVC activation completed but cl.exe is still unavailable.")
    environment["TORCH_EXTENSIONS_DIR"] = str(RUN_ROOT / "torch_extensions")
    environment["TORCH_CUDA_ARCH_LIST"] = "7.5"
    environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    return environment


def process_images(*, colmap: Path | None, force: bool, dry_run: bool) -> None:
    transforms_path = PROCESSED_DATA / "transforms.json"
    if transforms_path.exists() and not force:
        print(f"Reusing processed dataset: {transforms_path}")
        return

    colmap_path = find_colmap(colmap) if not dry_run else (colmap or LOCAL_COLMAP)
    process_environment = os.environ.copy()
    process_environment["PATH"] = str(EXPERIMENT_DIR) + os.pathsep + process_environment.get("PATH", "")
    process_environment["KLARTRAUM_COLMAP"] = str(colmap_path)

    images = tuple(SOURCE_IMAGES.glob("*.jpg"))
    if not images:
        raise SystemExit(f"No JPEG images found in {SOURCE_IMAGES}")

    run(
        uv_run(
            "ns-process-data",
            "images",
            "--data",
            str(SOURCE_IMAGES),
            "--output-dir",
            str(PROCESSED_DATA),
            "--sfm-tool",
            "colmap",
            "--colmap-cmd",
            str(EXPERIMENT_DIR / "colmap.cmd"),
            "--matching-method",
            "exhaustive",
            "--camera-type",
            "perspective",
        ),
        dry_run=dry_run,
        environment=process_environment,
    )


def latest_checkpoint_step() -> int | None:
    steps = []
    for checkpoint in MODEL_DIR.glob("step-*.ckpt"):
        if match := re.fullmatch(r"step-(\d+)\.ckpt", checkpoint.name):
            steps.append(int(match.group(1)))
    return max(steps, default=None)


def train(*, iterations: int, viewer: bool, force: bool, dry_run: bool) -> None:
    if iterations <= 0:
        raise SystemExit("--iterations must be positive")
    if not (PROCESSED_DATA / "transforms.json").exists() and not dry_run:
        raise SystemExit(f"Processed dataset not found. Run --stage process first: {PROCESSED_DATA}")
    checkpoint_step = latest_checkpoint_step()
    if checkpoint_step is not None and checkpoint_step >= iterations - 1 and not force:
        print(f"Reusing completed checkpoint at step {checkpoint_step}: {MODEL_DIR}")
        return

    training_environment = None
    if not dry_run:
        require_executable("nvidia-smi", "Splatfacto training requires an NVIDIA GPU and driver.")
        require_executable("nvcc", "Install a CUDA 12.x Toolkit and add its bin directory to PATH.")
        training_environment = cuda_build_environment()
    train_arguments = [
        "ns-train",
        "splatfacto",
        "--data",
        str(PROCESSED_DATA),
        "--output-dir",
        str(TRAINING_OUTPUT),
        "--experiment-name",
        "lantern-scene",
        "--timestamp",
        "run",
        "--max-num-iterations",
        str(iterations),
        "--vis",
        "viewer" if viewer else "tensorboard",
        "--logging.steps-per-log",
        "100",
        "--logging.local-writer.max-log-size",
        "1",
    ]
    if viewer:
        train_arguments.extend(("--viewer.quit-on-train-completion", "True"))
    if checkpoint_step is not None and not force:
        train_arguments.extend(("--load-dir", str(MODEL_DIR)))

    run(
        uv_run(*train_arguments),
        dry_run=dry_run,
        environment=training_environment,
    )


def export(*, force: bool, dry_run: bool) -> None:
    if SPLAT_PATH.exists() and not force:
        print(f"Reusing exported Gaussian splat: {SPLAT_PATH}")
        return
    if not CONFIG_PATH.exists() and not dry_run:
        raise SystemExit(f"Training config not found. Run --stage train first: {CONFIG_PATH}")

    export_environment = None if dry_run else cuda_build_environment()
    run(
        uv_run(
            "ns-export",
            "gaussian-splat",
            "--load-config",
            str(CONFIG_PATH),
            "--output-dir",
            str(EXPORT_OUTPUT),
        ),
        dry_run=dry_run,
        environment=export_environment,
    )


def main() -> int:
    args = parse_args()
    setup_environment(dry_run=args.dry_run)

    if args.stage in ("all", "process"):
        process_images(colmap=args.colmap, force=args.force, dry_run=args.dry_run)
    if args.stage in ("all", "train"):
        train(
            iterations=args.iterations,
            viewer=args.viewer,
            force=args.force,
            dry_run=args.dry_run,
        )
    if args.stage in ("all", "export"):
        export(force=args.force, dry_run=args.dry_run)

    if args.stage != "setup":
        print(f"Gaussian splat: {SPLAT_PATH}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
