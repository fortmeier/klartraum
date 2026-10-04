# Frontends

A frontend creates the {cpp:class}`klartraum::VulkanContext` and the
{cpp:class}`klartraum::KlartraumEngine` and is the entry point of an
application.

## GLFW frontend

{cpp:class}`klartraum::GlfwFrontend` opens a window, creates a swapchain and
runs the frame loop with `loop()`. Mouse and keyboard events are passed to the
engine, which forwards them to the active camera (for example
{cpp:class}`klartraum::InterfaceCameraOrbit`).

Everything that depends on the swapchain (image count, size) is created in a
*graph builder* set with `engine.setGraphBuilder(...)`. The engine runs it
once at start-up and again after every window resize.

### Where a frame comes from

There are two ways to render a windowed frame. Both ultimately call
`beforeStep()` followed by `KlartraumEngine::step()`:

```text
GlfwFrontend::loop()
    |
    +-- glfwPollEvents()
    |      |
    |      +-- no resize/refresh frame
    |      |      `-- return to loop -- process input -- render frame
    |      |
    |      `-- resize/refresh callback -- render frame -- return to loop
    |                                             `-- process input -- skip duplicate
    |
    `-- stop when the window closes or the finite frame budget is exhausted
```

The normal path polls window events, translates input into Klartraum events,
and then renders one frame. This is the path used during ordinary application
operation.

The callback path exists because a native window system can keep control while
the user drags a window edge. In that situation, `glfwPollEvents()` may not
return promptly, but GLFW continues invoking framebuffer-size and window-refresh
callbacks. Rendering synchronously from those callbacks keeps the resized
window responsive instead of freezing until the drag finishes.

The callback executes on the same thread and is not a background render loop.
Once `glfwPollEvents()` returns, the normal path checks whether a callback has
already rendered. If so, it does not render a second frame for that iteration.

### Finite frame runs

`loop(maxFrames)` is also used by tests and examples that must stop
automatically. Its budget counts actual calls through the shared frame routine,
regardless of whether the frame originated from the normal path or a window
callback. For example, `loop(8)` invokes `beforeStep()` at most eight times even
if Windows emits a refresh event when the window becomes resizable.

A non-positive `maxFrames` means that the frontend runs until the window is
closed. Callback frames are still followed by duplicate suppression, but there
is no finite budget to consume.

### Errors raised by callbacks

C++ exceptions must not unwind through GLFW's C callback frames. A resize or
refresh callback therefore stores any exception raised while rendering.
`pollEvents()` rethrows it after GLFW returns, restoring normal C++ exception
handling for the application.

### ImGui frames

{cpp:class}`klartraum::ImGuiFrontend` implements `beforeStep()` to start a new
ImGui frame, invoke the application's GUI callback, and finalize the overlay.
Because both frame origins use `beforeStep()`, the GUI is updated during an
interactive resize as well as during ordinary loop frames—and exactly once for
each rendered frame.

## Headless frontend

{cpp:class}`klartraum::HeadlessFrontend` creates a Vulkan context without a
window or swapchain. Graphs are compiled and submitted directly:

```cpp
klartraum::HeadlessFrontend frontend;
auto& vc = frontend.getKlartraumEngine().getVulkanContext();

klartraum::ComputeGraph graph(vc, 1);
graph.compileFrom(output);
graph.submitAndWait(vc.getGraphicsQueue(), 0);
```

The test suite uses the headless frontend, and it is the starting point for
inference-only applications.
