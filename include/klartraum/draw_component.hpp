#ifndef DRAW_COMPONENT_HPP
#define DRAW_COMPONENT_HPP

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <memory>

#include "klartraum/vulkan_kernel.hpp"

namespace klartraum {
class DrawComponent {
public:
    virtual void initialize(VulkanKernel& vulkanKernel, VkRenderPass& renderPass) {}

    virtual void recordCommandBuffer(VkCommandBuffer commandBuffer, VkFramebuffer framebuffer, uint32_t pathId) = 0;

};

} // namespace klartraum

#endif // DRAW_COMPONENT_HPP