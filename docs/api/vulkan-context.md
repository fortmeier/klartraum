# Vulkan context

Owns the Vulkan instance, the physical and logical device, the swapchain,
queue families and synchronization primitives.

```{doxygenclass} klartraum::VulkanContext
```

```{doxygenstruct} klartraum::QueueFamilyIndices
```

## Buffers

```{doxygenclass} klartraum::VulkanBuffer
```

## Shader and asset files

Shaders are loaded by relative paths such as `shaders/gsplat/gsplat_projection.comp.spv`;
see {doc}`../getting-started/building` for how they are found.

```{doxygenfunction} klartraum::readFile
```

```{doxygenfunction} klartraum::setAssetRoot
```

```{doxygenfunction} klartraum::getAssetRoot
```
