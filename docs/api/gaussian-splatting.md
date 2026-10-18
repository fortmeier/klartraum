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

```{doxygenclass} klartraum::GaussianSplattingInterface
```

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

```{doxygenenum} klartraum::GaussianDepthMode
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

```{doxygenclass} klartraum::GaussianMorph
```

```{doxygenfunction} klartraum::createGaussianMorph
```

```{doxygenstruct} klartraum::GaussianMorphResult
```

```{doxygenstruct} klartraum::MorphParams
```

```{doxygenenum} klartraum::MorphVariant
```

```{doxygenfunction} klartraum::prepareTextMorph
```

```{doxygenstruct} klartraum::TextMorphStart
```

```{doxygenclass} klartraum::GaussianLighting
```

```{doxygenfunction} klartraum::createGaussianLighting
```

```{doxygenfunction} klartraum::bakeSplatMaterials
```

```{doxygenstruct} klartraum::SplatMaterial
```

```{doxygenstruct} klartraum::SplatLight
```

```{doxygenstruct} klartraum::LightingFrame
```

## Text as Gaussians

```{doxygenclass} klartraum::Font
```

```{doxygenstruct} klartraum::TitleLayout
```

```{doxygenstruct} klartraum::TextLine
```

```{doxygenfunction} klartraum::rasterizeTitle
```

```{doxygenstruct} klartraum::CoverageMask
```

```{doxygenstruct} klartraum::TextSplatParams
```

```{doxygenfunction} klartraum::sampleTextGaussians(const CoverageMask&, const TextSplatParams&)
```

```{doxygenfunction} klartraum::sampleTextGaussians(const CoverageMask&, TextSplatParams, uint32_t)
```
