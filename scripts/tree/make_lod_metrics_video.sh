#!/usr/bin/env bash
# SPDX-FileCopyrightText: Dirk Fortmeier
#
# SPDX-License-Identifier: MIT

# Measures the popping of a single Gaussian tree along a flight away from it and back
# (tree_lod_metrics_example: a static tree left, one in the wind right) and encodes
# <out>/tree_lod_metrics.mp4 with labels and a strip of the residuals per frame.
# The frames stay in <out>/frames/, the residuals in <out>/popping.csv.
#
#   scripts/tree/make_lod_metrics_video.sh [out-dir] [extra options for tree_lod_metrics_example]
#
# Run from the repository root after building; needs ffmpeg and uv.

set -euo pipefail

out="${1:-build/TestingOutput/tree_lod_metrics}"
shift || true

./build/examples/tree_lod_metrics_example --out-dir "$out" "$@"
uv run scripts/tree/lod_metrics_video.py "$out"
