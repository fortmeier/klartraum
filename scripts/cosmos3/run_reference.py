# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

"""Generate a short Cosmos3-Edge image-to-video clip with the diffusers reference pipeline.

Runs on Apple Silicon (MPS), CUDA, or CPU. The model was trained on structured
JSON captions, so a plain-text prompt is wrapped into a minimal caption of that
shape unless ``--raw-prompt`` is given.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import Cosmos3OmniPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image
from huggingface_hub import hf_hub_download
from PIL import Image

from disclaimer import print_cosmos3_notice


MODEL_ID = "nvidia/Cosmos3-Edge"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE = REPO_ROOT / "data" / "lantern.jpg"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "build" / "TestingOutput" / "cosmos3"
DEFAULT_PROMPT = "camera flight orbiting around the lantern"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def structured_prompt(text: str, num_frames: int, fps: float, height: int, width: int) -> str:
    """Wrap a plain prompt into the JSON caption layout of the Cosmos 3 examples."""
    duration = num_frames / fps
    seconds = max(1, round(duration))
    caption = {
        "subjects": [
            {
                "description": "A traditional Japanese granite garden lantern standing in a garden",
                "action": "Stands still while the camera moves around it",
                "number_of_subjects": 1,
            }
        ],
        "cinematography": {
            "camera_motion": text,
            "framing": "Medium shot keeping the lantern centered",
        },
        "style_medium": "Live-action video",
        "actions": [{"time": f"0:00-0:{seconds:02d}", "description": text}],
        "temporal_caption": text,
        "resolution": {"W": width, "H": height},
        "duration": f"{seconds}s",
        "fps": fps,
    }
    return json.dumps(caption)


def main() -> None:
    print_cosmos3_notice()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--raw-prompt", action="store_true", help="pass --prompt unchanged instead of wrapping it")
    parser.add_argument("--size", type=int, nargs=2, default=(256, 256), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--num-frames", type=int, default=33, help="4k+1 frames (VAE temporal stride 4)")
    parser.add_argument("--fps", type=float, default=16.0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--flow-shift", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--name", default="lantern_orbit")
    args = parser.parse_args()

    width, height = args.size
    device = pick_device()
    print(f"device={device} size={width}x{height} frames={args.num_frames} steps={args.steps}", flush=True)

    t0 = time.perf_counter()
    # The Cosmos guardrail (cosmos_guardrail) pulls in several additional large
    # models and is not practical on a 24 GB unified-memory machine; the
    # checkpoint's model_index.json also defaults it to disabled.
    pipe = Cosmos3OmniPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, enable_safety_checker=False)
    pipe.to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=args.flow_shift, use_karras_sigmas=False
    )
    print(f"loaded pipeline in {time.perf_counter() - t0:.1f}s", flush=True)

    prompt = (
        args.prompt
        if args.raw_prompt
        else structured_prompt(args.prompt, args.num_frames, args.fps, height, width)
    )
    negative_prompt = Path(hf_hub_download(MODEL_ID, "assets/negative_prompt.json")).read_text()
    image = load_image(str(args.image))

    step_times: list[float] = []

    def on_step_end(pipeline, step, timestep, callback_kwargs):
        step_times.append(time.perf_counter())
        print(f"step {step + 1}/{args.steps} t={float(timestep):.1f} ({step_times[-1] - t1:.1f}s)", flush=True)
        return callback_kwargs

    # Noise is drawn from a CPU generator so a seed gives the same initial
    # latents on MPS, CUDA, and CPU.
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    t1 = time.perf_counter()
    result = pipe(
        prompt=prompt,
        negative_prompt=json.dumps(json.loads(negative_prompt)),
        image=image,
        num_frames=args.num_frames,
        height=height,
        width=width,
        fps=args.fps,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        add_resolution_template=False,
        add_duration_template=False,
        enable_safety_check=False,
        output_type="np",
        callback_on_step_end=on_step_end,
    )
    total = time.perf_counter() - t1
    print(f"generated in {total:.1f}s", flush=True)

    video = result.video
    if isinstance(video, list):
        video = video[0]
    video = np.asarray(video)
    if video.ndim == 5:
        video = video[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mp4 = args.output_dir / f"{args.name}.mp4"
    export_to_video(list(video), str(mp4), fps=int(args.fps), macro_block_size=1)
    frames = (np.clip(video, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(np.concatenate([frames[0], frames[len(frames) // 2], frames[-1]], axis=1)).save(
        args.output_dir / f"{args.name}_strip.png"
    )
    (args.output_dir / f"{args.name}_prompt.json").write_text(prompt)
    print(f"wrote {mp4} ({len(frames)} frames, {frames.shape[2]}x{frames.shape[1]})", flush=True)


if __name__ == "__main__":
    main()
