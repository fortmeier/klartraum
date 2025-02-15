#ifndef DRAW_COMPONENT_HPP
#define DRAW_COMPONENT_HPP

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <memory>

#include "klartraum/vulkan_kernel.hpp"

namespace klartraum {
class DrawComponent {
public:
    virtual void draw(uint32_t currentFrame, VkCommandBuffer& commandBuffer, VkFramebuffer& framebuffer, VkSemaphore& imageAvailableSemaphore, uint32_t imageIndex) = 0;

    virtual void initialize(VulkanKernel& vulkanKernel) {}
};

} // namespace klartraum

#endif // DRAW_COMPONENT_HPP