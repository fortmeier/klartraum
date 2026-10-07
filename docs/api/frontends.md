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
