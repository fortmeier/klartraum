# Pre-recorded command buffers

The central design decision of Klartraum is that **all GPU work is recorded
into Vulkan command buffers once, when a graph is compiled**. Running the
graph, for example once per frame, only submits these command buffers again;
nothing is re-recorded.

This keeps the CPU almost idle while the GPU works, which lowers latency and
leaves the CPU free for other tasks. That matters most on small devices, where
the CPU is often the bottleneck.

## Consequences

- **Structure is fixed after compilation.** Buffer sizes, pipelines and the
  order of operations are decided when the graph is compiled. Changing them
  means compiling the graph again (the engine does this, for example, after a
  window resize through its graph builder).
- **Values can still change every frame.** Data such as the camera matrices or
  numbers set from the CPU live in GPU buffers. Elements that need to copy new
  host-side values into their buffers report this with `isUpdatable()`, and
  the graph calls their `_update()` right before a path is submitted.
- **Several frames can be in flight.** The graph is recorded once per
  *path*, so while the GPU is still working on one frame, the next frame can
  already be submitted on another path with its own buffers.
