#include "klartraum/vulkan_gaussian_splatting.hpp"
#include <stdexcept>

namespace klartraum {

VulkanGaussianSplatting::VulkanGaussianSplatting(BackendVulkan &backendVulkan) : backendVulkan(backendVulkan) {
    createCommandPool();
    createCommandBuffers();
}

VulkanGaussianSplatting::~VulkanGaussianSplatting() {

}

void VulkanGaussianSplatting::draw() {
}

void VulkanGaussianSplatting::createCommandPool() {
    auto device = backendVulkan.getDevice();

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = backendVulkan.getQueueFamilyIndices().graphicsFamily.value();

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }    
}

void VulkanGaussianSplatting::createCommandBuffers() {
    auto device = backendVulkan.getDevice();

    commandBuffers.resize(BackendVulkan::MAX_FRAMES_IN_FLIGHT);
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = commandBuffers.size();

    if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate command buffers!");
    }
}

} // namespace klartraum
