#!/usr/bin/env bash

# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

# Two-step dashcam example: Stable Diffusion 1.5 text-to-image, then
# Cosmos3-Edge image-to-video (33 frames at 11 fps, 3 s). See README.md.
#
#   scripts/cosmos3/dashcam.sh reference   # both models with diffusers
#   scripts/cosmos3/dashcam.sh klartraum   # both models on Klartraum
#
# The klartraum mode needs the SD1.5 512 export (data/onnx/sd15_denoiser_512),
# the Cosmos3 export (data/onnx/cosmos3_256), and a build of both examples.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
COSMOS="$REPO/scripts/cosmos3"
OUT="$REPO/build/TestingOutput/dashcam"
CAPTION="$COSMOS/captions/dashcam.json"
PROMPT="dashcam photo from inside a car driving on a suburban street, view through the windshield, road ahead with lane markings, parked cars, trees and houses on both sides, daylight, realistic photograph"
NEGATIVE="blurry, distorted, low quality, cartoon, painting, text, watermark"
mkdir -p "$OUT"

case "${1:-}" in
reference)
    cd "$COSMOS"
    uv run python sd15_text_to_image.py --prompt "$PROMPT" --negative-prompt "$NEGATIVE" \
        --output "$OUT/sd15_reference.png"
    uv run python run_reference.py --raw-prompt --prompt "$(cat "$CAPTION")" \
        --image "$OUT/sd15_reference.png" --num-frames 33 --fps 11 \
        --output-dir "$OUT" --name cosmos3_reference
    ;;
klartraum)
    FIXTURES="$REPO/data/onnx/cosmos3_dashcam"
    cd "$REPO"
    ./build/examples/sd15_denoiser_example --size 512 --model-dir data/onnx/sd15_denoiser_512 \
        --prompt "$PROMPT" --negative-prompt "$NEGATIVE" --output "$OUT/sd15_klartraum.ppm"

    # The graphs are shared with cosmos3_256 through hard links (the ONNX loader
    # rejects external data that resolves outside the model directory); the
    # prompt- and image-dependent fixtures are generated for this run.
    mkdir -p "$FIXTURES"
    for file in data/onnx/cosmos3_256/*.onnx data/onnx/cosmos3_256/*.onnx.data \
        data/onnx/cosmos3_256/decoder_reference_*_f32.bin; do
        ln -f "$file" "$FIXTURES/$(basename "$file")"
    done
    cd "$COSMOS"
    uv run python export_onnx.py prepare --fps 11 --prompt-file "$CAPTION" \
        --image "$OUT/sd15_klartraum.ppm" --onnx-dir "$FIXTURES"
    uv run python export_onnx.py text --fixtures-only --onnx-dir "$FIXTURES"
    uv run python export_onnx.py denoiser --fixtures-only --onnx-dir "$FIXTURES"
    uv run python export_onnx.py pipeline --onnx-dir "$FIXTURES" --video-dir "$OUT"

    cd "$REPO"
    ./build/examples/cosmos3_example --model-dir "$FIXTURES" --image "$OUT/sd15_klartraum.ppm" \
        --output-dir "$OUT/klartraum" --name dashcam
    ffmpeg -y -loglevel error -i "$OUT/klartraum/dashcam_klartraum.y4m" -pix_fmt yuv420p "$OUT/dashcam_klartraum.mp4"
    echo "wrote $OUT/dashcam_klartraum.mp4"
    ;;
*)
    echo "usage: $0 reference|klartraum" >&2
    exit 2
    ;;
esac
