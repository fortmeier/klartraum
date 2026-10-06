# Klartraum Examples

This directory contains example applications demonstrating how to use the Klartraum library.

## Examples

### Gaussian Splatting Example
**File**: `gaussian_splatting_example.cpp`

A basic example that demonstrates:
- Loading a Gaussian splat file (.spz format)
- Setting up a camera with orbit controls
- Rendering axes for reference
- Basic interaction (mouse orbit, keyboard movement)

#### Controls:
- **Mouse + Left Click**: Orbit camera around the scene
- **Mouse Wheel**: Zoom in/out
- **WASD**: Move camera position
- **Space**: Reset camera to default position
- **Escape**: Exit application

#### Usage:
```bash
# From build directory
./gaussian_splatting_example

# Or from project root
./build/gaussian_splatting_example

# The default scene is data/lantern.spz. Load another scene (--flip-y mirrors
# it across the Y axis) and set the world-space camera position; it looks at
# the origin
./build/gaussian_splatting_example --file path/to/scene.spz --camera-position 0.55 0.48 0.69 --flip-y
```

### Turntable Example
**File**: `turntable_example.cpp`

Renders a Gaussian splat scene headlessly (no window) from a camera that
circles the scene's vertical axis exactly once, and writes one PPM image per
frame. The camera orbits the scene centre (the median of the Gaussian
positions). After the frames, the view after a full turn is compared with the
first frame, and the program fails if they differ, so the frames always loop
seamlessly. `scripts/site/make_lantern_animation.sh` uses it for the animation on
the landing page.

#### Usage:
```bash
# From project root
./build/examples/turntable_example --file data/lantern.spz --flip-y --frames 60 \
    --width 400 --height 300 --camera-position 0.55 0.48 0.69 --distance 0.65 \
    --out-dir build/TestingOutput/turntable
```

### Qwen Chat Example
**File**: `qwen_chat_example.cpp`

An interactive chat in the terminal with Qwen3.6-27B, run by `GgufNetwork`
from a GGUF file (download it with `scripts/qwen_gguf/download.py`; see
`scripts/qwen_gguf/README.md`). Each turn is formatted with the Qwen chat
template and only the new tokens are run; the conversation stays in the
network's key/value caches and linear-attention states. Replies are streamed
as they are generated, thinking (when enabled) in dim text.

Commands: `/reset` starts a new conversation, `/think on` / `/think off`
switches thinking for the next turns, `/stats` prints the context use and the
last turn's speed, `/quit` exits.

Options: `--model FILE`, `--system TEXT`, `--prompt TEXT` (answer one prompt
and exit), `--think`, `--context N` (default 4096 tokens), `--chunk N` (prompt
tokens per submission, up to 16), `--max-reply N`, `--temperature T`,
`--top-p P`, `--top-k K` (defaults follow Qwen's recommendations: 0.7 / 0.8 /
20 without thinking, 1.0 / 0.95 / 20 with it), `--greedy`, `--seed N`,
`--quiet`.

#### Usage:
```bash
# From project root
./build/examples/qwen_chat_example
./build/examples/qwen_chat_example --think --system "You are a concise assistant."
./build/examples/qwen_chat_example --greedy --prompt "What is a Vulkan compute shader?"
```

## Building Examples

Examples are built automatically when you build the main project.
