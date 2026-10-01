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
- `submitAndWait(queue, pathId)` submits and blocks until the GPU is done,
  which is convenient for tests and one-off computations.

## Queues and threads

Vulkan queues and command pools must not be used from two threads at once.
`VulkanContext` keeps a lock per queue: submit and wait with
`queueSubmit()`, `queueWaitIdle()` and `queuePresent()` instead of the
`vkQueue*` functions.

Long computations can run next to the frames on a worker thread:

```cpp
std::thread worker([&] {
    klartraum::VulkanContext::BackgroundQueueScope scope(vulkanContext);
    klartraum::ComputeGraph graph(vulkanContext, 1);  // records for the background queue
    graph.compileFrom(root);
    graph.submitAndWait(vulkanContext.getThreadQueue(), 0);
});
```

- `getBackgroundQueue()` is a queue of its own when the device has a second
  graphics and compute queue (another queue of the graphics family, or a queue
  of another family; MoltenVK has four families with one queue each).
  Otherwise it is the graphics queue, and the frames wait while one of its
  submissions executes.
- Inside a `BackgroundQueueScope`, `getThreadQueue()` is the background
  queue; elsewhere it is the graphics queue. A `ComputeGraph` allocates its
  command buffers for the queue of the thread that creates it, and
  `submitImmediate()` (and everything built on it, such as buffer uploads)
  submits to the calling thread's queue.
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
