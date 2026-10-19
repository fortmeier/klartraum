#!/usr/bin/env bash
# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

# Renders the bonsai video with both renderers (tree_bonsai_example --video:
# four days, one season each, with sun, moon and wind) and encodes
#   <out>/bonsai_classical.mp4, <out>/bonsai_gaussian.mp4 and
#   <out>/bonsai_side_by_side.mp4 (classical left, Gaussian right).
# The frames stay in <out>/frames/.
#
#   scripts/tree/make_bonsai_video.sh [out-dir] [extra options for tree_bonsai_example]
#
# Run from the repository root after building.

set -euo pipefail

out="${1:-build/TestingOutput/tree_bonsai_video}"
shift || true
fps=30

./build/examples/tree_bonsai_example --video --width 1920 --height 1080 --fps "$fps" --out-dir "$out" "$@"

for renderer in classical gaussian; do
    ffmpeg -loglevel error -y -framerate "$fps" -i "$out/frames/${renderer}_%04d.ppm" \
        -c:v libx264 -crf 18 -pix_fmt yuv420p "$out/bonsai_${renderer}.mp4"
done
ffmpeg -loglevel error -y -i "$out/bonsai_classical.mp4" -i "$out/bonsai_gaussian.mp4" -filter_complex \
    "[0]scale=960:540[a];[1]scale=960:540[b];[a][b]hstack" \
    -c:v libx264 -crf 18 -pix_fmt yuv420p "$out/bonsai_side_by_side.mp4"
echo "videos in $out"
