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

## Submission model (as of commit b37b1c3b)

### Current: Single queue, single command buffer per path

- All elements of a path are recorded into **one command buffer** in topological order
- A **full memory barrier** (`MEMORY_WRITE → MEMORY_READ|WRITE`) separates consecutive elements,
  ensuring each element sees all writes from earlier elements
- One `VkSubmitInfo` per path (instead of one per element)
- Only **external semaphores** (e.g. swapchain image acquisition) are waited on;
  no per-element binary semaphores are needed
- Synchronization happens inside the command buffer, not via the queue

**Why this model?** The old per-element semaphore chain caused KosmicKrisp to hang
when submitting 5,000+ elements (Stable Diffusion 1.5 UNet). Batching thousands
of binary-semaphore waits on a single queue exceeded driver capabilities.
Recording everything into one command buffer removes this overhead.

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
- `submitImmediate()` (and everything built on it, such as buffer uploads
  with `memcopyFrom()`) always submits to the graphics queue, also from a
  worker thread. Large uploads from a worker therefore briefly hold up the
  frames ([#40](https://github.com/fortmeier/klartraum/issues/40)).
- When the background queue belongs to another queue family, buffers made by
  `VulkanContext::createBuffer()` and offscreen images are shared by both
  families (`VK_SHARING_MODE_CONCURRENT`), so a result computed on the
  background queue can be read by the frames without an ownership transfer.
  Wait for the computation (e.g. `submitAndWait()`) before the frames read
  it.
- `vkDeviceWaitIdle()` waits for both queues and needs every queue unused:
  call it only when no background work is running, e.g. at shutdown. A
  swapchain recreation waits only for the graphics and present queues.

## Future: Multi-queue architecture

**Not yet implemented.** When graph branches are independent (no data dependencies),
they could run in parallel on separate compute or graphics queues for higher GPU utilization.

### Motivation

Independent branches currently serialize on a single queue:
- Branch A: Shader A → Shader B (10ms)
- Branch B: Shader C → Shader D (5ms)
- They wait for each other, total time: 15ms

With multiple queues:
- Queue 0: Shader A → Shader B (10ms, runs in parallel)
- Queue 1: Shader C → Shader D (5ms, runs in parallel)
- Total time: max(10ms, 5ms) = 10ms (33% faster)

### Implementation strategy

1. **Detect independent sub-graphs**: Analyze the DAG to find branches with no data dependencies.
2. **Assign to queues**: Partition elements into disjoint sets, each bound to a queue family.
3. **Reconstruct edge semaphores**: Create binary semaphores only at merge points where
   independent branches reconverge and synchronize.
4. **Record per-queue command buffers**: Each queue has its own buffer(s) for its assigned elements.
5. **Multi-queue submission**: Submit to each queue, waiting on inter-queue synchronization semaphores.

### Changes required

- Add support for multiple queue families (compute, graphics, transfer, etc.)
- Modify topological sort to optionally partition into independent sets
- Add per-element queue family assignment
- Reconstruct `SubmitInfoWrapper` per queue (not per path)
- Track queue-to-elements mapping and inter-queue dependencies
- Update `submitTo()` to handle multiple `vkQueueSubmit()` calls

### Compatibility

The public API (`compileFrom()`, `submitTo()`, `submitAndWait()`) would remain unchanged.
The graph would automatically benefit from multi-queue execution if the topology allows it.

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
