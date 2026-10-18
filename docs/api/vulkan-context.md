# Vulkan context

Owns the Vulkan instance, the physical and logical device, the swapchain,
queue families and synchronization primitives.

```{doxygenclass} klartraum::VulkanContext
```

```{doxygenstruct} klartraum::QueueFamilyIndices
```

## Buffers

{cpp:class}`klartraum::VulkanBuffer` and the buffer elements are documented in {doc}`buffers`.

## Shader and asset files

Shaders are loaded by relative paths such as `shaders/gsplat/gsplat_projection.comp.spv`;
see {doc}`../getting-started/building` for how they are found.

```{doxygenfunction} klartraum::readFile
```

```{doxygenfunction} klartraum::setAssetRoot
```

```{doxygenfunction} klartraum::getAssetRoot
```
