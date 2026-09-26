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
```

### ImGui Gaussian Splatting Example
**File**: `imgui_gaussian_splatting_example.cpp`

Renders a Gaussian splat scene with the raster backend and a Dear ImGui user
interface on top, using `ImGuiFrontend`:
- Statistics: frame rate, resolution, number of Gaussians
- Camera: azimuth, elevation, distance and target sliders (kept in sync with
  mouse orbiting), reset button
- Rendering: SH degree, alpha-cull threshold and the mesh-shader path; "Apply"
  rebuilds the raster pipeline with the new settings
- Optional ImGui demo window

Mouse input over a GUI window goes to the GUI, not the camera. The window is
resizable.

#### Usage:
```bash
# From project root (shaders are loaded relative to it)
./build/examples/imgui_gaussian_splatting_example
./build/examples/imgui_gaussian_splatting_example --spz path/to/scene.spz
./build/examples/imgui_gaussian_splatting_example --frames 120   # close after 120 frames
```

## Building Examples

Examples are built automatically when you build the main project.
