# Compute graph

A directed acyclic graph of elements. Each element records its own Vulkan
command buffers; the graph connects them with semaphores.

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

```{doxygenclass} klartraum::BufferElementInterface
```

```{doxygenclass} klartraum::TemplatedBufferElementInterface
```

```{doxygenclass} klartraum::BufferElement
```

```{doxygenclass} klartraum::BufferElementSinglePath
```

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

```{doxygenclass} klartraum::RenderPass
```

```{doxygenclass} klartraum::ImageViewSrc
```

## Synchronization and placeholders

```{doxygenclass} klartraum::ImageViewSrcTransition
```

```{doxygenclass} klartraum::BufferToGraphicsBarrier
```

```{doxygenclass} klartraum::NoOp
```
