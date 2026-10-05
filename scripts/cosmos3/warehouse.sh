#!/usr/bin/env bash

# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

# Warehouse scenes at Cosmos3-Edge's documented settings: 832x480, 121 frames
# at 24 fps (5 s), 20 steps. Start images come from SD1.5 on Klartraum (512x512,
# centre-cropped to 16:9 by the conditioning preprocessing), the clips from
# Cosmos3 on Klartraum without reference checks. Scenes, SD prompts, and
# captions are in warehouse_scenes.json. See README.md.
#
#   scripts/cosmos3/warehouse.sh                  # all scenes
#   scripts/cosmos3/warehouse.sh forklift conveyor
#
# Needs the 832x480 export in data/onnx/cosmos3_832x480 (README.md) and the
# SD1.5 512 export in data/onnx/sd15_denoiser_512.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
COSMOS="$REPO/scripts/cosmos3"
OUT="$REPO/build/TestingOutput/warehouse"
MODELS="$REPO/data/onnx/cosmos3_832x480"
NEGATIVE="blurry, distorted, low quality, cartoon, painting, text, watermark"
mkdir -p "$OUT"
cd "$COSMOS"
field() {
    python3 -c "import json, sys; scene = json.load(open('warehouse_scenes.json'))[sys.argv[1]]; \
print(scene['sd_prompt'] if sys.argv[2] == 'sd' else json.dumps(scene['caption']))" "$1" "$2"
}
SCENES=("$@")
[[ ${#SCENES[@]} -gt 0 ]] || SCENES=($(python3 -c "import json; print(' '.join(json.load(open('warehouse_scenes.json'))))"))

for scene in "${SCENES[@]}"; do
    echo "== $scene"
    field "$scene" caption > "$OUT/${scene}_caption.json"
    if [[ ! -f "$OUT/${scene}_sd.ppm" ]]; then
        (cd "$REPO" && ./build/examples/sd15_denoiser_example --size 512 --model-dir data/onnx/sd15_denoiser_512 \
            --prompt "$(field "$scene" sd)" --negative-prompt "$NEGATIVE" --output "$OUT/${scene}_sd.ppm")
    fi
    FIXTURES="$REPO/data/onnx/cosmos3_warehouse_$scene"
    mkdir -p "$FIXTURES"
    for file in text_kv text_kv_cond_512 denoiser vae_encoder vae_decoder_first vae_decoder_chunk; do
        ln -f "$MODELS/$file.onnx" "$FIXTURES/$file.onnx"
        ln -f "$MODELS/$file.onnx.data" "$FIXTURES/$file.onnx.data"
    done
    uv run python export_onnx.py prepare --size 832 480 --num-frames 121 --fps 24 --onnx-dir "$FIXTURES" \
        --prompt-file "$OUT/${scene}_caption.json" --image "$OUT/${scene}_sd.ppm" --decoder-chunked --skip-encode-check
    (cd "$REPO" && ./build/examples/cosmos3_example --model-dir "$FIXTURES" --image "$OUT/${scene}_sd.ppm" \
        --skip-checks --output-dir "$OUT/klartraum_$scene" --name "$scene")
    ffmpeg -y -loglevel error -i "$OUT/klartraum_$scene/${scene}_klartraum.y4m" -pix_fmt yuv420p "$OUT/${scene}.mp4"
    echo "wrote $OUT/${scene}.mp4"
done
