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
3. **Records** all elements of the path into a single command buffer (`recordPath()`),
   with a full memory barrier between consecutive elements.
4. **Prepares submission info** for the path, collecting external synchronization
   points (e.g. swapchain image acquisition semaphores).

## Paths

A graph is compiled for a fixed number of *paths*. Every path has its own
command buffer and, where needed, its own output buffers. In a windowed
application there is one path per swapchain image, so several frames can be
processed at the same time without sharing buffers that are still in use.

Elements whose data is identical for all paths (a loaded model, for example)
keep a single buffer; elements whose output changes per frame keep one per
path.

See {doc}`../api/buffers` for allocation lifetime, host-visible and device-local
storage, and per-path upload examples.

## Submission model

### One command buffer per path

All elements of a graph path run on one queue, including independent
branches. Separate graphs can be submitted to the graphics or background
queue.

- All elements of a path are recorded into **one command buffer** in topological order
- A **full memory barrier** (`MEMORY_WRITE → MEMORY_READ|WRITE`) separates consecutive elements,
  ensuring each element sees all writes from earlier elements
- One `VkSubmitInfo` per path
- The submission waits for **external semaphores** (e.g. swapchain image acquisition)
  and signals the graph-finished semaphore when the path completes

### Submitting

- `submitTo(queue, pathId)` runs the host-side updates of the path, submits
  the path's single command buffer and returns a semaphore that is signalled when the path
  has finished. The frame loop uses this before presenting.
- The queue must belong to the queue family the graph was compiled for:
  `compileFrom(element)` compiles for the graphics queue,
  `compileFrom(element, family)` for the queues of another family (see
  [Queues and threads](#queues-and-threads)). Submitting to a queue of
  another family throws.
- `submitAndWait(queue, pathId)` submits and blocks until the GPU is done,
  which is convenient for tests and one-off computations.

## Queues and threads

### The problem: compute graphs on several threads

An application has one `VulkanContext`, but it may want to run several
compute graphs at the same time, each built and submitted by its own thread.
All of them use what the context provides: its queues, its command pool and
helpers such as `submitImmediate()`.

The typical case is a long computation next to a frame loop. The main
thread submits a frame graph to the graphics queue every frame, while a
worker thread runs a graph that takes up to seconds, e.g. a Stable Diffusion run. Run on the
main thread instead, it would stop the frame loop and freeze the window.

Sharing one context between threads raises two issues:

1. **Vulkan objects are not thread-safe.** Vulkan does not lock queues or
   command pools internally; the spec requires the application to *externally
   synchronize* them. Two threads calling `vkQueueSubmit()` on the same queue
   at the same time, or recording command buffers from the same command pool,
   is a data race with undefined behaviour (lost submissions, corrupted driver
   state, crashes).
2. **Graphs on one queue do not really run in parallel.** Even with correct
   locking, if both threads submit to the graphics queue, the frames'
   submissions queue up behind the worker's long submission. The CPU side of
   the frame loop keeps running, but in practice each frame waits on the GPU
   until that submission has finished.

### The solution: locked queues and a background queue

klartraum addresses the first point by how it shares objects between threads:

- **Queues** are shared, so `VulkanContext` keeps a mutex per queue. Submit,
  wait and present with `queueSubmit()`, `queueWaitIdle()` and
  `queuePresent()`, never with the `vkQueue*` functions directly.
- **Command pools** are never shared: each `ComputeGraph` owns its pool.

For the second point it provides a **background queue**: a second queue,
separate from the graphics queue, that the GPU can work on next to the frames.

### Example

The main thread compiles the frame graph for the graphics queue, with one
path per swapchain image. A worker thread compiles its graph for the
background queue's family and submits it to the background queue, while the
main thread keeps rendering on the graphics queue:

```cpp
// Main thread: the frame graph, compiled for the graphics queue.
// `frameRoot` is the output element of the frame, e.g. a render pass.
klartraum::ComputeGraph frames(vulkanContext, vulkanContext.getNumberOfSwapChainImages());
frames.compileFrom(frameRoot);

// Worker thread: the long computation, compiled for the background queue.
// `root` is its output element, e.g. a UNet.
std::atomic<bool> done{false};
std::thread worker([&] {
    klartraum::ComputeGraph graph(vulkanContext, 1);
    graph.compileFrom(root, vulkanContext.getBackgroundQueueFamily());
    graph.submitAndWait(vulkanContext.getBackgroundQueue(), 0);
    done = true;
});

// Main thread: the frame loop keeps running on the graphics queue.
while (!done) {
    auto [imageIndex, fence] = vulkanContext.beginRender();
    VkSemaphore finished = frames.submitTo(vulkanContext.getGraphicsQueue(), imageIndex, fence);
    vulkanContext.endRender(imageIndex, finished);
}
worker.join();
// the result of `root` can now be read by the frames
```

### Details

- `getBackgroundQueue()` is a queue of its own when the device has a second
  graphics and compute queue (another queue of the graphics family, or a queue
  of another family; MoltenVK has four families with one queue each).
  Otherwise, e.g. on KosmicKrisp, which has a single queue, it is the graphics
  queue itself. A worker thread can then still submit its graph safely, but it
  does not run in the background: rendering is blocked while one of its
  submissions executes. Check `hasOwnBackgroundQueue()` to tell the two cases
  apart.
- A graph is compiled for one queue family, because Vulkan command buffers
  can only be submitted to queues of the family their command pool was
  created for. One thread can compile several graphs for different families
  and submit each to its own queue.
- Immediate submissions take the queue they run on: `submitImmediate()`,
  `copyBufferImmediate()`, the staging copies of `VulkanBuffer::memcopyFrom()`
  and `memcopyTo()` for device-local buffers, and the copies of
  `SinglePathImage::copyFrom()`. Without one they use the graphics queue. On
  a worker thread, pass the background queue: the graphics queue also holds
  the frames, which wait for the swapchain (vsync), so every immediate
  submission there waits up to a frame.
- Uploads while a graph is compiled, e.g. an ONNX network's weights, go to
  the queue family the graph is compiled for: `compileFrom()` passes a queue
  of that family as the optional `setupQueue` argument to `_setup()`. An
  element that sets up other elements passes this argument on to them.
  Omitting the queue uses the graphics queue for immediate submissions.
- When the background queue belongs to another queue family, buffers made by
  `VulkanContext::createBuffer()` and offscreen images are shared by both
  families (`VK_SHARING_MODE_CONCURRENT`), so a result computed on the
  background queue can be read by the frames without an ownership transfer.
  Wait for the computation (e.g. `submitAndWait()`) before the frames read
  it.
- `vkDeviceWaitIdle()` waits for both queues and needs every queue unused:
  call it only when no background work is running, e.g. at shutdown. A
  swapchain recreation waits only for the graphics and present queues.

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
`getElementTimesMs()` returns the same means in the order of `getElements()`,
and `getGpuSpanMs()` the mean time from the first element's start to the last
element's end, which also covers the barriers between elements.
`resetProfiling()` starts the means anew, e.g. after warm-up frames.
`writeGraphJson()` exports a compiled graph with its elements, edges and times
for drawing it.

See {doc}`../guides/profiling` for examples, readback rules and the performance
impact of collecting these measurements.

## Chains of graphs

A {cpp:class}`klartraum::GraphChain` submits several compiled graphs in one
`vkQueueSubmit`, each waiting for a semaphore of the one before, so data passes
from graph to graph on the GPU without the CPU in between. A graph can appear
several times in a chain if it was compiled after `enableSimultaneousUse()`.
`BufferElement` instances shared by the graphs are allocated once, by the
first graph that compiles them, and the graphs must have the same number of
paths. For submissions the graph does not wait for itself, `readProfiling()`
reads the timestamps once the chain has finished.

### Example

With an initialized `VulkanContext vc`, this example copies four values
through two graphs. The first writes a shared device-local buffer, and the
second reads it:

```cpp
using Floats = klartraum::VulkanBuffer<float>;
using Buffer = klartraum::BufferElement<Floats>;

auto input = std::make_shared<Buffer>(vc, 4);
auto shared = std::make_shared<Buffer>(
    vc, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
auto output = std::make_shared<Buffer>(vc, 4);

auto firstCopy = std::make_shared<klartraum::CopyBuffer>(vc);
firstCopy->setInput(input, 0);
firstCopy->setInput(shared, 1);

auto secondCopy = std::make_shared<klartraum::CopyBuffer>(vc);
secondCopy->setInput(shared, 0);
secondCopy->setInput(output, 1);

klartraum::ComputeGraph first(vc, 1), second(vc, 1);
first.compileFrom(firstCopy);
second.compileFrom(secondCopy);

input->getBuffer(0).memcopyFrom(std::vector<float>{1, 2, 3, 4});

klartraum::GraphChain chain(vc);
chain.add(first, 0);
chain.add(second, 0);
chain.submitAndWait(vc.getGraphicsQueue());

std::vector<float> result(4);
output->getBuffer(0).memcopyTo(result); // {1, 2, 3, 4}
```

Both graphs have one path, selected by the `0` passed to `add()`. Compiling
the second graph reuses the shared buffer allocated by the first. The chain
orders their copies on the GPU; the CPU uploads before the chain and reads
the result after both links finish.

See {doc}`../requirements/graph-chains` for the requirements, specifications
and tests.
