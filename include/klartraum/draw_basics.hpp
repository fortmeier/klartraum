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

    virtual void draw(uint32_t currentFrame, VkCommandBuffer& commandBuffer, VkFramebuffer& framebuffer, VkSemaphore& imageAvailableSemaphore) override;

private:
    void createGraphicsPipeline();
    void createSyncObjects();
    void createVertexBuffer();

    void recordCommandBuffer(uint32_t currentFrame, VkCommandBuffer commandBuffer, VkFramebuffer framebuffer);

    BackendVulkan &backendVulkan;

    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;

    DrawBasicsType type;

};

} // namespace klartraum

#endif // KLARTRAUM_DRAW_BASICS_HPP