#ifndef DRAW_COMPONENT_HPP
#define DRAW_COMPONENT_HPP

#include <vulkan/vulkan.h>
#include <memory>

namespace klartraum {
class DrawComponent {
public:
    virtual VkSemaphore draw(uint32_t currentFrame, VkFramebuffer framebuffer, VkSemaphore imageAvailableSemaphore) = 0;
};

} // namespace klartraum

#endif // DRAW_COMPONENT_HPP