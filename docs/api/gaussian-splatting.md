# Gaussian splatting

## Data

```{doxygenclass} klartraum::GaussianDataStandard
```

```{doxygenfunction} klartraum::loadGaussiansSpz
```

```{doxygenstruct} klartraum::Gaussian3D
```

```{doxygenstruct} klartraum::GaussianSoABuffers
```

```{doxygenstruct} klartraum::BufferRef
```

## Backends

{cpp:func}`klartraum::createGaussianSplatting` builds either backend from the
same inputs.

```{doxygenfile} gaussian_splatting_factory.hpp
```

```{doxygenclass} klartraum::VulkanGaussianSplatting
```

```{doxygenclass} klartraum::VulkanGaussianSplattingRaster
```

```{doxygenstruct} klartraum::GsplatConfig
```

## Draw components of the raster backend

```{doxygenclass} klartraum::GaussianSplatRasterizer
```

```{doxygenclass} klartraum::GaussianSplatMeshRasterizer
```

## Operations on Gaussians

```{doxygenclass} klartraum::GaussianTransform
```

```{doxygenfunction} klartraum::createGaussianTransform
```

```{doxygenstruct} klartraum::GaussianTransformResult
```

```{doxygenclass} klartraum::GaussianMerge
```

```{doxygenfunction} klartraum::createGaussianMerge
```

```{doxygenstruct} klartraum::GaussianMergeResult
```

```{doxygenclass} klartraum::TransformBuffer
```

```{doxygenfunction} klartraum::createTransformBuffer
```

```{doxygenstruct} klartraum::TransformBufferResult
```
