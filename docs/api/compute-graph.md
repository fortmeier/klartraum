# Compute graph

A directed acyclic graph of elements. Each element records its own Vulkan
command buffers; the graph connects them with semaphores.

## Graph

```{doxygenclass} klartraum::ComputeGraph
```

```{doxygenclass} klartraum::ComputeGraphElement
```

## Buffers and tensors

```{doxygenclass} klartraum::BufferElementInterface
```

```{doxygenclass} klartraum::BufferElement
```

```{doxygenclass} klartraum::TensorElementInterface
```

```{doxygenclass} klartraum::TensorElement
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

## Rendering

```{doxygenclass} klartraum::RenderPass
```

```{doxygenclass} klartraum::ImageViewSrc
```
