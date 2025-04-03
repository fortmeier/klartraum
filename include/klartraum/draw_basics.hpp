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

    virtual void initialize(VulkanKernel& vulkanKernel, VkRenderPass& renderpass) override;

    void recordCommandBuffer(VkCommandBuffer commandBuffer, VkFramebuffer framebuffer, uint32_t pathId) override;

private:
    void createGraphicsPipeline();
    void createSyncObjects();
    void createVertexBuffer();


    VulkanKernel* vulkanKernel;

    VkRenderPass* renderPass;

    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;

    DrawBasicsType type;

};

} // namespace klartraum

#endif // KLARTRAUM_DRAW_BASICS_HPP