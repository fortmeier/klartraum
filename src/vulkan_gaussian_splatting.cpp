#include "klartraum/vulkan_gaussian_splatting.hpp"
#include <stdexcept>

namespace klartraum {

VulkanGaussianSplatting::VulkanGaussianSplatting(BackendVulkan &backendVulkan) : device(backendVulkan.getDevice()) {
    createCommandPool();
    createCommandBuffers();
}

VulkanGaussianSplatting::~VulkanGaussianSplatting() {

}

void VulkanGaussianSplatting::createCommandPool() {
}

void VulkanGaussianSplatting::createCommandBuffers() {
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
