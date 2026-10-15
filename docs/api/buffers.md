# Buffers

{cpp:class}`klartraum::VulkanBuffer` owns a typed Vulkan allocation.
{cpp:class}`klartraum::BufferElement` makes that storage available to a compute
graph, with one allocation per {ref}`render path <compute-graph-paths>`.

| Type | Allocation lifetime | Sharing |
| --- | --- | --- |
| `VulkanBuffer<T>` | Allocated by its constructor; released by its destructor | One allocation |
| `BufferElement<VulkanBuffer<T>>` | Allocated during graph compilation | One independent allocation per path |
| `BufferElementSinglePath<VulkanBuffer<T>>` | Allocated by its constructor | One allocation shared by every path |

**`BufferElement` defers allocation because the number of paths is supplied during
graph compilation.** Its constructor stores the capacity, usage and memory
properties. When `compileFrom()` calls `_setup(context, numberPaths)`, it creates
one buffer for each path. This supports independent per-frame inputs and outputs.

**`BufferElementSinglePath` already knows it needs exactly one buffer.** Its
constructor directly constructs its `VulkanBuffer` member, which allocates
immediately. Its `_setup()` does nothing, and `getBuffer(pathId)` ignores the path
index.

The Vulkan context must outlive the allocations. Finish GPU use before destroying
them. Shared single-path buffers are useful for model data that stays unchanged
while several paths read it.

## CPU access and memory placement

The `usage` flags describe GPU operations, such as storage-buffer or indirect-draw
access. The `memory` flags select a memory type. Transfer-source and
transfer-destination usage are enabled automatically by `VulkanBuffer`.

| Requested memory flags | CPU copy helpers | Typical use |
| --- | --- | --- |
| `HOST_VISIBLE_BIT \| HOST_COHERENT_BIT` (default) | Map the allocation directly | Small, frequent CPU updates |
| `DEVICE_LOCAL_BIT`, without `HOST_VISIBLE_BIT` | Copy through temporary coherent staging buffers | GPU model data and working buffers |

These names abbreviate the `VK_MEMORY_PROPERTY_` flags. Device-local and
host-visible properties can coexist on a memory type. The helpers choose mapping
or staging from the **requested** `HOST_VISIBLE_BIT`; `getMemoryProperties()`
reports the **actual selected** memory type's flags.

Direct mapping requires coherent host-visible memory: these helpers do not flush
or invalidate non-coherent allocations. Coherence does not replace synchronization;
wait for conflicting GPU work before CPU access. For engine-driven per-frame
uploads, use `setFrameUpdate()` to update only the acquired path.

Staged uploads, readback, GPU prefix copies and device-local `zero()` wait for the
graphics queue to become idle. Use them for initialization and explicit readback.
Per-frame GPU processing stays in shaders and recorded command buffers.

With an initialized `VulkanContext vc`, both access modes use the same copy API:

```cpp
std::vector<float> data{1, 2, 3, 4};

klartraum::VulkanBuffer<float> cpuUpdates(vc, 4);
cpuUpdates.memcopyFrom(data);

klartraum::VulkanBuffer<float> gpuData(
    vc, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
gpuData.memcopyFrom(data);

std::vector<float> result(4);
gpuData.memcopyTo(result);
```

## Graph paths

See {ref}`compute-graph-paths` for path IDs and resource isolation.

Connect buffer elements before `compileFrom()` and access per-path allocations
after compilation. For example, a two-path graph can copy mapped input to
device-local output:

```cpp
using Floats = klartraum::VulkanBuffer<float>;
auto input = std::make_shared<klartraum::BufferElement<Floats>>(vc, 4);
auto output = std::make_shared<klartraum::BufferElement<Floats>>(
    vc, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
auto copy = std::make_shared<klartraum::CopyBuffer>(vc);
copy->setInput(input, 0);
copy->setInput(output, 1);

klartraum::ComputeGraph graph(vc, 2);
graph.compileFrom(copy);
input->getBuffer(0).memcopyFrom(data);
graph.submitAndWait(vc.getGraphicsQueue(), 0);
output->getBuffer(0).memcopyTo(result);
```

`BufferElement::zero()` immediately clears every path. Configure
`setRecordToZero()`, `setRecordToZeroRange()` or `setRecordToFill()` before
compilation to perform fills on the GPU each time the graph runs.

## Allocation API

```{doxygenclass} klartraum::VulkanBuffer
```

## Graph buffer API

```{doxygenclass} klartraum::BufferElement
```

```{doxygenclass} klartraum::BufferElementSinglePath
```

## Interfaces

```{doxygenclass} klartraum::BufferElementInterface
```

```{doxygenclass} klartraum::TemplatedBufferElementInterface
```
