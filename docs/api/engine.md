# Engine

Wraps the Vulkan context and the compute graph, compiles graphs, manages the
camera uniform buffer and drives the frame-submit loop.

```{doxygenclass} klartraum::KlartraumEngine
```

## Windows and render targets

A window composites viewports into its swapchain image; offscreen targets are
the render targets of viewports and of rendering without a window.

```{doxygenclass} klartraum::Window
```

```{doxygenclass} klartraum::OffscreenTarget
```

## Overlays

```{doxygenclass} klartraum::FrameOverlay
```
