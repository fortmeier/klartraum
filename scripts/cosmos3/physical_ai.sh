#!/usr/bin/env bash

# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

# Physical-AI demo scenes: SD1.5 start image, then a 512x512, 3 s Cosmos3 clip
# (33 frames at 11 fps). Scenes, SD prompts, and Cosmos3 captions are in
# physical_ai_scenes.json. See README.md.
#
#   scripts/cosmos3/physical_ai.sh reference <scene>   # SD1.5 and Cosmos3 with diffusers
#   scripts/cosmos3/physical_ai.sh klartraum <scene>   # Cosmos3 on Klartraum from the reference SD image
#
# The klartraum mode needs data/onnx/cosmos3_512 with the chunk decoders (see
# dashcam.sh) and runs without reference checks: only `export_onnx.py prepare`
# runs in Python (tokens, rotary tables, noise).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
COSMOS="$REPO/scripts/cosmos3"
OUT="$REPO/build/TestingOutput/physical_ai"
NEGATIVE="blurry, distorted, low quality, cartoon, painting, text, watermark"
MODE="${1:-}"
SCENE="${2:-}"
mkdir -p "$OUT"
cd "$COSMOS"
field() {
    uv run python -c "import json, sys; scene = json.load(open('physical_ai_scenes.json'))[sys.argv[1]]; \
print(scene['sd_prompt'] if sys.argv[2] == 'sd' else json.dumps(scene['caption']))" "$SCENE" "$1"
}
[[ -n "$SCENE" ]] && field sd > /dev/null || { echo "usage: $0 reference|klartraum <scene>" >&2; exit 2; }
field caption > "$OUT/${SCENE}_caption.json"

case "$MODE" in
reference)
    uv run python sd15_text_to_image.py --prompt "$(field sd)" --negative-prompt "$NEGATIVE" --output "$OUT/${SCENE}_sd.png"
    uv run python run_reference.py --raw-prompt --prompt "$(cat "$OUT/${SCENE}_caption.json")" \
        --image "$OUT/${SCENE}_sd.png" --size 512 512 --num-frames 33 --fps 11 --output-dir "$OUT" --name "$SCENE"
    ;;
klartraum)
    MODELS="$REPO/data/onnx/cosmos3_512"
    FIXTURES="$REPO/data/onnx/cosmos3_demo_$SCENE"
    mkdir -p "$FIXTURES"
    for file in text_kv text_kv_cond_512 denoiser vae_encoder vae_decoder_first vae_decoder_chunk; do
        ln -f "$MODELS/$file.onnx" "$FIXTURES/$file.onnx"
        ln -f "$MODELS/$file.onnx.data" "$FIXTURES/$file.onnx.data"
    done
    uv run python -c "import sys; from PIL import Image; Image.open(sys.argv[1]).convert('RGB').save(sys.argv[2])" \
        "$OUT/${SCENE}_sd.png" "$OUT/${SCENE}_sd.ppm"
    uv run python export_onnx.py prepare --size 512 --fps 11 --onnx-dir "$FIXTURES" \
        --prompt-file "$OUT/${SCENE}_caption.json" --image "$OUT/${SCENE}_sd.ppm" --decoder-chunked
    cd "$REPO"
    ./build/examples/cosmos3_example --model-dir "$FIXTURES" --image "$OUT/${SCENE}_sd.ppm" --skip-checks \
        --output-dir "$OUT/klartraum_$SCENE" --name "$SCENE"
    ffmpeg -y -loglevel error -i "$OUT/klartraum_$SCENE/${SCENE}_klartraum.y4m" -pix_fmt yuv420p \
        "$OUT/${SCENE}_klartraum.mp4"
    echo "wrote $OUT/${SCENE}_klartraum.mp4"
    ;;
*)
    echo "usage: $0 reference|klartraum <scene>" >&2
    exit 2
    ;;
esac
