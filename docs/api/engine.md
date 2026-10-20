# Engine

Wraps the Vulkan context and the compute graph, compiles graphs, manages the
camera uniform buffer and drives the frame-submit loop.

```{doxygenclass} klartraum::KlartraumEngine
```

## Frame updates

`setFrameUpdate()` installs an application callback that receives the acquired
render path's index after its previous GPU use completes and before graph
submission. Pass `nullptr` to disable the callback.

If the callback throws, the engine completes the acquired frame without running
its graphs and rethrows the original exception. The frame fence is signaled and
its acquisition semaphore is consumed, so the application can catch the exception
and render another frame or shut down. A cancelled window frame discards its
contents and returns the image in presentation layout. Cancellation waits for
device completion before releasing its temporary submission resources.

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
