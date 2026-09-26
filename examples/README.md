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

# Load a scene and set the world-space camera position; it looks at the origin
./build/gaussian_splatting_example --file data/lantern.spz --camera-position 0.55 0.48 0.69 --flip-y
```

### ImGui Gaussian Splatting Example
**File**: `imgui_gaussian_splatting_example.cpp`

Renders a Gaussian splat scene with a Dear ImGui user interface on top, using
`ImGuiFrontend`. The rendering backend (raster or compute) can be switched at
runtime from a dropdown:
- Statistics: frame rate, resolution, number of Gaussians
- Camera: azimuth, elevation, distance and target sliders (kept in sync with
  mouse orbiting), reset button
- Rendering: backend dropdown (raster/compute, switches immediately) and the
  settings the selected backend uses — raster: SH degree, alpha-cull threshold,
  mesh-shader path; compute: spread multiplier. "Apply" rebuilds the pipeline
  with the new settings
- Optional ImGui demo window

Mouse input over a GUI window goes to the GUI, not the camera. The window is
resizable.

#### Usage:
```bash
# From project root (shaders are loaded relative to it)
./build/examples/imgui_gaussian_splatting_example
./build/examples/imgui_gaussian_splatting_example --spz path/to/scene.spz
./build/examples/imgui_gaussian_splatting_example --backend compute   # start on the compute backend
./build/examples/imgui_gaussian_splatting_example --frames 120   # close after 120 frames
```

## Building Examples

Examples are built automatically when you build the main project.
