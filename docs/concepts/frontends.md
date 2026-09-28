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
