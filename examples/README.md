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

## Building Examples

Examples are built automatically when you build the main project.
