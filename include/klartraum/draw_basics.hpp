#ifndef KLARTRAUM_DRAW_BASICS_HPP
#define KLARTRAUM_DRAW_BASICS_HPP

#include <vulkan/vulkan.h>
#include <vector>
#include <string>

#include "klartraum/backend_vulkan.hpp"

namespace klartraum {

enum class DrawBasicsType {
    Triangle,
    Cube,
    Axes
};

class DrawBasics : public DrawComponent {
public:

    DrawBasics(BackendVulkan &backendVulkan, DrawBasicsType type);
    ~DrawBasics();

    virtual VkSemaphore draw(uint32_t currentFrame, VkFramebuffer framebuffer, VkSemaphore imageAvailableSemaphore) override;

private:
    void createGraphicsPipeline();
    void createSyncObjects();
    void createCommandPool();
    void createCommandBuffers();
    void createVertexBuffer();

    void recordCommandBuffer(uint32_t currentFrame, VkCommandBuffer commandBuffer, VkFramebuffer framebuffer);

    BackendVulkan &backendVulkan;

    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;

    std::vector<VkSemaphore> renderFinishedSemaphores;
    
    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;

    DrawBasicsType type;

};

} // namespace klartraum

#endif // KLARTRAUM_DRAW_BASICS_HPP