#ifndef KLARTRAUM_DRAW_BASICS_HPP
#define KLARTRAUM_DRAW_BASICS_HPP

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vector>
#include <string>

#include "klartraum/draw_component.hpp"

namespace klartraum {

enum class DrawBasicsType {
    Triangle,
    Cube,
    Axes
};

class DrawBasics : public DrawComponent {
public:

    DrawBasics(DrawBasicsType type);
    ~DrawBasics();

    virtual void draw(uint32_t currentFrame, VkCommandBuffer& commandBuffer, VkFramebuffer& framebuffer, VkSemaphore& imageAvailableSemaphore) override;

    virtual void initialize(VulkanKernel& vulkanKernel) override;

private:
    void createGraphicsPipeline();
    void createSyncObjects();
    void createVertexBuffer();

    void recordCommandBuffer(uint32_t currentFrame, VkCommandBuffer commandBuffer, VkFramebuffer framebuffer);

    VulkanKernel* vulkanKernel;

    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;

    DrawBasicsType type;

};

} // namespace klartraum

#endif // KLARTRAUM_DRAW_BASICS_HPP