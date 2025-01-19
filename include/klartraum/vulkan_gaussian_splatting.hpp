#ifndef VULKAN_GAUSSIAN_SPLATTING_HPP
#define VULKAN_GAUSSIAN_SPLATTING_HPP

#include <vulkan/vulkan.h>
#include <vector>

#include "klartraum/backend_vulkan.hpp"

namespace klartraum {

class VulkanGaussianSplatting {
public:
    VulkanGaussianSplatting(BackendVulkan &backendVulkan);
    ~VulkanGaussianSplatting();

    void drawFrame();

private:
    void createSyncObjects();
    void createCommandPool();
    void createCommandBuffers();

    BackendVulkan &backendVulkan;

    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;
};

} // namespace klartraum

#endif // VULKAN_GAUSSIAN_SPLATTING_HPP