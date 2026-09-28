"""Forward FFmpeg arguments to imageio-ffmpeg's uv-managed binary."""

from __future__ import annotations

import subprocess
import sys

import imageio_ffmpeg


raise SystemExit(subprocess.call([imageio_ffmpeg.get_ffmpeg_exe(), *sys.argv[1:]]))
