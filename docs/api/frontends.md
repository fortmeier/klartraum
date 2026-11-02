# Frontends

The user-facing entry points: a windowed application with GLFW, or a headless
frontend for tests and compute-only workloads.

```{doxygenclass} klartraum::GlfwFrontend
```

```{doxygenclass} klartraum::HeadlessFrontend
```

## Cameras

The camera uniform buffer has the type `CameraUboType`, a
`UniformBufferObject<CameraMVP>`.

```{doxygenclass} klartraum::InterfaceCamera
```

```{doxygenclass} klartraum::InterfaceCameraOrbit
```

```{doxygenstruct} klartraum::CameraMVP
```

`cropProjection()` creates an off-center view of a square window in the original
projection's normalized device coordinates. `centre` gives the window's center
and `halfExtent` its positive half-size; the resulting projection preserves depth.
Use it for a cropped render or a higher-resolution view of part of the scene.

```{doxygenfunction} klartraum::cropProjection
```

## Input events

```{doxygenclass} klartraum::Event
```

```{doxygenclass} klartraum::EventMouseMove
```

```{doxygenclass} klartraum::EventMouseButton
```

```{doxygenclass} klartraum::EventMouseScroll
```

```{doxygenclass} klartraum::EventKey
```
