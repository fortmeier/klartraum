#!/usr/bin/env bash
# Renders the lantern scene with the turntable example (one full turn around
# the up axis) and assembles the frames into seamlessly looping animations:
#   site/images/engine-lantern.mp4         480 px video for the landing page
#   site/images/engine-lantern-poster.jpg  its first frame, shown before it plays
#                                          and instead of it with reduced motion
#   site/images/engine-lantern.gif         300 px GIF for the documentation
# Run from the repository root after building the examples.
# TURNTABLE selects the executable (default: build/examples/turntable_example).
set -euo pipefail

cd "$(dirname "$0")/../.."

TURNTABLE="${TURNTABLE:-build/examples/turntable_example}"
# 60 frames at 20 fps: one turn in 3 s, 6 degrees per frame.
FRAMES=60
FPS=20
# 4:3, the aspect ratio of the product cards on the landing page. Frames are
# rendered larger than they are shown and scaled down, which smooths the
# splat edges.
WIDTH=640
HEIGHT=480
VIDEO_WIDTH=480
GIF_WIDTH=300
FRAME_DIR=build/TestingOutput/lantern_turntable
OUT=site/images/engine-lantern

rm -rf "$FRAME_DIR"
"$TURNTABLE" --file data/lantern.spz --flip-y --frames "$FRAMES" --width "$WIDTH" --height "$HEIGHT" \
    --camera-position 0.55 0.48 0.69 --distance 0.65 --out-dir "$FRAME_DIR" \
    | grep -Ev '^edge:'

ffmpeg -loglevel error -y -framerate "$FPS" -i "$FRAME_DIR/frame_%04d.ppm" \
    -vf "scale=$VIDEO_WIDTH:-2:flags=lanczos" \
    -c:v libx264 -preset veryslow -crf 23 -pix_fmt yuv420p -movflags +faststart -an "$OUT.mp4"

ffmpeg -loglevel error -y -i "$FRAME_DIR/frame_0000.ppm" \
    -vf "scale=$VIDEO_WIDTH:-2:flags=lanczos" -q:v 3 "$OUT-poster.jpg"

# Every second frame, one palette for all frames (stable colours over the
# loop) and 48 colours without dithering keep the GIF below 500 KB.
ffmpeg -loglevel error -y -framerate "$FPS" -i "$FRAME_DIR/frame_%04d.ppm" \
    -vf "select='not(mod(n\,2))',setpts=N/$((FPS / 2))/TB,scale=$GIF_WIDTH:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=48:stats_mode=full[p];[b][p]paletteuse=dither=none" \
    -r "$((FPS / 2))" -loop 0 "$OUT.gif"

for f in "$OUT.mp4" "$OUT-poster.jpg" "$OUT.gif"; do
    echo "wrote $f ($(du -h "$f" | cut -f1))"
done
