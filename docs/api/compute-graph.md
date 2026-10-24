# Compute graph

A directed acyclic graph of elements. The graph records its elements into one
Vulkan command buffer per path, in dependency order, with memory barriers
between consecutive elements.

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

## Chains

```{doxygenclass} klartraum::GraphChain
```

## Export

```{doxygenfunction} klartraum::writeGraphJson
```

```{doxygenstruct} klartraum::ExportGroup
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

For clearing, image composition and shared images, see
{doc}`../guides/render-targets` and {doc}`../requirements/render-targets`.

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

```{doxygenenum} klartraum::CompositeMode
```

```{doxygenenum} klartraum::CompositeFit
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
