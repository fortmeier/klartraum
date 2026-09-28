# The compute graph

A {cpp:class}`klartraum::ComputeGraph` is a directed acyclic graph of
{cpp:class}`klartraum::ComputeGraphElement` nodes. Each element produces an
output (a buffer, a tensor, an image) and can take the outputs of other
elements as inputs.

```cpp
auto blur  = std::make_shared<BlurOp>();  blur->setInput(renderpass);
auto noise = std::make_shared<NoiseOp>(); noise->setInput(blur);
auto add   = std::make_shared<AddOp>();   add->setInput(blur, 0); add->setInput(noise, 1);

klartraum::ComputeGraph graph(vulkanContext, numberPaths);
graph.compileFrom(add);
graph.submitAndWait(vulkanContext.getGraphicsQueue(), 0);
```

## Compiling

`compileFrom(element)` starts at the given output element and walks its
inputs. It then:

1. **Orders the elements** with Kahn's topological-sort algorithm, so every
   element runs after all of its inputs.
2. **Sets up** each element (`_setup()`): pipelines, descriptor sets and
   buffers are created.
3. **Records** one command buffer per element and path (`_record()`).
4. **Connects** the command buffers with semaphores, so each element waits
   for the elements it depends on.

## Paths

A graph is compiled for a fixed number of *paths*. Every path has its own
command buffers and, where needed, its own output buffers. In a windowed
application there is one path per swapchain image, so several frames can be
processed at the same time without sharing buffers that are still in use.

Elements whose data is identical for all paths (a loaded model, for example)
keep a single buffer; elements whose output changes per frame keep one per
path.

## Submitting

- `submitTo(queue, pathId)` runs the host-side updates of the path, submits
  its command buffers and returns a semaphore that is signalled when the path
  has finished. The frame loop uses this before presenting.
- `submitAndWait(queue, pathId)` submits and blocks until the GPU is done,
  which is convenient for tests and one-off computations.

## Element types

| Element | Purpose |
|---|---|
| {cpp:class}`klartraum::BufferElement` | A GPU buffer of a fixed type |
| {cpp:class}`klartraum::TensorElement` | A tensor with separate dimension and data buffers |
| {cpp:class}`klartraum::UniformBufferObject` | A uniform buffer updated from the CPU, e.g. the camera |
| {cpp:class}`klartraum::GeneralComputation` | A compute shader with inputs, outputs and push constants |
| {cpp:class}`klartraum::BufferTransformation` | A compute shader mapping one buffer to another |
| {cpp:class}`klartraum::CopyBuffer` | Copies a buffer |
| {cpp:class}`klartraum::RenderPass` | Classic rasterization into an image |

## Profiling

Before compiling, `enableProfiling()` adds GPU timestamp queries around every
element, and `enablePerformanceProfiling()` adds hardware performance counters
where the driver supports `VK_KHR_performance_query` (with pipeline statistics
as a fallback). `getProfilingResults()` returns the mean time per element.
