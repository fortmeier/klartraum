#ifndef VULKAN_GAUSSIAN_SPLATTING_HPP
#define VULKAN_GAUSSIAN_SPLATTING_HPP

#include <vulkan/vulkan.h>
#include <vector>
#include <string>

#include "klartraum/backend_vulkan.hpp"

namespace klartraum {

class VulkanGaussianSplatting : public DrawComponent {
public:
    VulkanGaussianSplatting(BackendVulkan &backendVulkan, std::string path);
    ~VulkanGaussianSplatting();

    virtual VkSemaphore draw(uint32_t currentFrame, VkFramebuffer framebuffer, VkSemaphore imageAvailableSemaphore) override;

private:
    void createGraphicsPipeline();
    void createSyncObjects();
    void createCommandPool();
    void createCommandBuffers();
    void createVertexBuffer();

    void recordCommandBuffer(uint32_t currentFrame, VkCommandBuffer commandBuffer, VkFramebuffer framebuffer);

    void loadSPZModel(std::string path);

    BackendVulkan &backendVulkan;

    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;

    std::vector<VkSemaphore> renderFinishedSemaphores;
    
    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;
};

} // namespace klartraum

#endif // VULKAN_GAUSSIAN_SPLATTING_HPP