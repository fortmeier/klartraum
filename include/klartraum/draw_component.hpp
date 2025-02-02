#ifndef DRAW_COMPONENT_HPP
#define DRAW_COMPONENT_HPP

#include <vulkan/vulkan.h>
#include <memory>

namespace klartraum {
class DrawComponent {
public:
    virtual void draw(uint32_t currentFrame, VkCommandBuffer& commandBuffer, VkFramebuffer& framebuffer, VkSemaphore& imageAvailableSemaphore) = 0;
};

} // namespace klartraum

#endif // DRAW_COMPONENT_HPP