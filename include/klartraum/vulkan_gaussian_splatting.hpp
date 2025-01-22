#ifndef VULKAN_GAUSSIAN_SPLATTING_HPP
#define VULKAN_GAUSSIAN_SPLATTING_HPP

#include <vulkan/vulkan.h>
#include <vector>

#include "klartraum/backend_vulkan.hpp"

namespace klartraum {

class VulkanGaussianSplatting : public DrawComponent {
public:
    VulkanGaussianSplatting(BackendVulkan &backendVulkan);
    ~VulkanGaussianSplatting();

    virtual VkSemaphore draw(uint32_t currentFrame, VkFramebuffer framebuffer, VkSemaphore imageAvailableSemaphore) override;

private:
    void createGraphicsPipeline();
    void createSyncObjects();
    void createCommandPool();
    void createCommandBuffers();

    void recordCommandBuffer(VkCommandBuffer commandBuffer, VkFramebuffer framebuffer);

    BackendVulkan &backendVulkan;

    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;


    std::vector<VkSemaphore> renderFinishedSemaphores;
    
    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;
};

} // namespace klartraum

#endif // VULKAN_GAUSSIAN_SPLATTING_HPP