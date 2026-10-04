#!/usr/bin/env bash

# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

# Two-step dashcam example: Stable Diffusion 1.5 text-to-image, then
# Cosmos3-Edge image-to-video (512x512, 33 frames at 11 fps, 3 s). See README.md.
#
#   scripts/cosmos3/dashcam.sh reference   # both models with diffusers
#   scripts/cosmos3/dashcam.sh klartraum   # both models on Klartraum
#
# The klartraum mode needs the SD1.5 512 export (data/onnx/sd15_denoiser_512),
# the Cosmos3 256 export (data/onnx/cosmos3_256, for its text tower and VAE
# decoder), and a build of both examples. The 512 denoiser and VAE encoder are
# exported into data/onnx/cosmos3_512 on the first run (about 15 minutes).
# If a 320x320 export exists (data/onnx/cosmos3_320, `export_onnx.py prepare` and
# `vae --skip-ort` with --size 320, about an hour), its decoder decodes the clip
# in 2x2 tiles of 20x20 latents instead of 3x3 tiles of 16x16.
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
        --image "$OUT/sd15_reference.png" --size 512 512 --num-frames 33 --fps 11 \
        --output-dir "$OUT" --name cosmos3_reference
    ;;
klartraum)
    MODELS="$REPO/data/onnx/cosmos3_512"
    cd "$REPO"
    ./build/examples/sd15_denoiser_example --size 512 --model-dir data/onnx/sd15_denoiser_512 \
        --prompt "$PROMPT" --negative-prompt "$NEGATIVE" --output "$OUT/sd15_klartraum.ppm"

    # The text tower does not depend on the video size, and the 512 clip is
    # decoded in tiles by a smaller VAE decoder, so both graphs are shared
    # through hard links (the ONNX loader rejects external data that resolves
    # outside the model directory).
    if [[ -f data/onnx/cosmos3_320/vae_decoder.onnx ]]; then
        DECODER=data/onnx/cosmos3_320 TILE=(--decoder-tile 20 --decoder-stride 12)
    else
        DECODER=data/onnx/cosmos3_256 TILE=(--decoder-tile 16 --decoder-stride 8)
    fi
    mkdir -p "$MODELS"
    for file in data/onnx/cosmos3_256/text_kv.onnx* "$DECODER"/vae_decoder.onnx* \
        "$DECODER"/decoder_reference_*_f32.bin; do
        ln -f "$file" "$MODELS/$(basename "$file")"
    done
    cd "$COSMOS"
    CONFIG=(--size 512 --fps 11 --onnx-dir "$MODELS")
    uv run python export_onnx.py prepare "${CONFIG[@]}" --prompt-file "$CAPTION" \
        --image "$OUT/sd15_klartraum.ppm" "${TILE[@]}"
    uv run python export_onnx.py text --fixtures-only "${CONFIG[@]}"
    if [[ -f "$MODELS/denoiser.onnx" && -f "$MODELS/vae_encoder.onnx" ]]; then
        uv run python export_onnx.py denoiser --fixtures-only "${CONFIG[@]}"
    else
        uv run python export_onnx.py reference "${CONFIG[@]}"
        uv run python export_onnx.py denoiser "${CONFIG[@]}"
        uv run python export_onnx.py vae "${CONFIG[@]}"
    fi
    uv run python export_onnx.py pipeline "${CONFIG[@]}" --video-dir "$OUT"

    cd "$REPO"
    ./build/examples/cosmos3_example --model-dir "$MODELS" --image "$OUT/sd15_klartraum.ppm" \
        --output-dir "$OUT/klartraum" --name dashcam
    ffmpeg -y -loglevel error -i "$OUT/klartraum/dashcam_klartraum.y4m" -pix_fmt yuv420p "$OUT/dashcam_klartraum.mp4"
    echo "wrote $OUT/dashcam_klartraum.mp4"
    ;;
*)
    echo "usage: $0 reference|klartraum" >&2
    exit 2
    ;;
esac
