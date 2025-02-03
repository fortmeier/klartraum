#ifndef VULKAN_GAUSSIAN_SPLATTING_HPP
#define VULKAN_GAUSSIAN_SPLATTING_HPP

#include <vulkan/vulkan.h>
#include <vector>
#include <string>

#include "klartraum/backend_vulkan.hpp"

namespace klartraum {

class VulkanGaussianSplatting : public DrawComponent {
public:
    VulkanGaussianSplatting(GlfwFrontend &backendVulkan, std::string path);
    ~VulkanGaussianSplatting();

    virtual void draw(uint32_t currentFrame, VkCommandBuffer& commandBuffer, VkFramebuffer& framebuffer, VkSemaphore& imageAvailableSemaphore) override;

private:
    void createGraphicsPipeline();
    void createSyncObjects();

    void createVertexBuffer();

    void recordCommandBuffer(uint32_t currentFrame, VkCommandBuffer commandBuffer, VkFramebuffer framebuffer);

    void loadSPZModel(std::string path);

    GlfwFrontend &backendVulkan;

    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
 
    
};

} // namespace klartraum

#endif // VULKAN_GAUSSIAN_SPLATTING_HPP