# Compute graph

A directed acyclic graph of elements. Each element records its own Vulkan
command buffers; the graph connects them with semaphores.

(compute-graph-paths)=
## Paths

A *path* is a numbered execution instance of the same graph, with its own
recorded command buffers and per-path resources. In windowed rendering, `pathId`
is the acquired swapchain image's index. Separate paths let frames run concurrently
without overwriting one another's data; immutable model buffers may be shared.

For memory placement, CPU transfers and per-path buffer storage, see {doc}`buffers`.

## Graph

```{doxygenclass} klartraum::ComputeGraph
```

```{doxygenclass} klartraum::ComputeGraphElement
```

## Groups

Elements that are built from several graph elements.

```{doxygenclass} klartraum::ComputeGraphGroup
```

```{doxygenclass} klartraum::RenderGraphElement
```

## Buffers and tensors

The buffer allocation and interface types are documented in {doc}`buffers`.

```{doxygentypedef} klartraum::DispatchIndirectCommandBufferElement
```

```{doxygentypedef} klartraum::DrawIndirectCommandBufferElement
```

```{doxygenclass} klartraum::HostValues
```

```{doxygenclass} klartraum::TensorElementInterface
```

```{doxygenclass} klartraum::TensorElement
```

```{doxygenclass} klartraum::TensorElementSinglePath
```

```{doxygenclass} klartraum::TensorMemoryPlanner
```

```{doxygenstruct} klartraum::TensorLifetimeRequest
```

```{doxygenstruct} klartraum::TensorMemoryPlan
```

```{doxygenstruct} klartraum::TensorMemoryAssignment
```

```{doxygenclass} klartraum::UniformBufferObject
```

## Computations

```{doxygenclass} klartraum::GeneralComputation
```

```{doxygenclass} klartraum::BufferTransformation
```

```{doxygenclass} klartraum::CopyBuffer
```

```{doxygenclass} klartraum::ImageResample
```

```{doxygenenum} klartraum::ResampleFilter
```

## Rendering

```{doxygenclass} klartraum::RenderPass
```

```{doxygenclass} klartraum::ImageViewSrc
```

```{doxygenclass} klartraum::ImageViewForward
```

```{doxygenclass} klartraum::ClearImage
```

```{doxygenclass} klartraum::ImageComposite
```

```{doxygenfunction} klartraum::recordClearImage
```

```{doxygenclass} klartraum::SinglePathImage
```

## Synchronization and placeholders

```{doxygenclass} klartraum::ImageViewSrcTransition
```

```{doxygenclass} klartraum::BufferToGraphicsBarrier
```

```{doxygenclass} klartraum::NoOp
```
