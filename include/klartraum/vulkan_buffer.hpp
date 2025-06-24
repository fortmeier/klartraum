#ifndef KLARTRAUM_VULKAN_BUFFER_HPP
#define KLARTRAUM_VULKAN_BUFFER_HPP

#include <vulkan/vulkan.h>

#include <algorithm>

#include <klartraum/vulkan_context.hpp>

namespace klartraum {

template <typename T>
class VulkanBuffer {
public:
    VulkanBuffer(VulkanContext& kernel, uint32_t size) : vulkanContext(kernel), size(size) { //, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties);
        auto& device = kernel.getDevice();

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = sizeof(T) * size;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
        if (vkCreateBuffer(device, &bufferInfo, nullptr, &vertexBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute buffer!");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, vertexBuffer, &memRequirements);
    
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = vulkanContext.findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
        if (vkAllocateMemory(device, &allocInfo, nullptr, &vertexBufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate vertex buffer memory!");
        }
        vkBindBufferMemory(device, vertexBuffer, vertexBufferMemory, 0);
    }

    VulkanBuffer(VulkanBuffer&& other) noexcept
        : size(other.size),
          vertexBuffer(other.vertexBuffer),
          vertexBufferMemory(other.vertexBufferMemory),
          vulkanContext(other.vulkanContext) {
        other.vertexBuffer = VK_NULL_HANDLE;
        other.vertexBufferMemory = VK_NULL_HANDLE;
    }

    ~VulkanBuffer() {
        auto& device = vulkanContext.getDevice();
        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);
    }

    void memcopyFrom(const std::vector<T>& src) {
        auto& device = vulkanContext.getDevice();
        void* mappedData;
        vkMapMemory(device, vertexBufferMemory, 0, sizeof(T) * size, 0, &mappedData);
        size_t dataSize = sizeof(T) * std::min((uint32_t)src.size(), (uint32_t)size);
        memcpy(mappedData, src.data(), dataSize);
        vkUnmapMemory(device, vertexBufferMemory);
    }

    void memcopyTo(std::vector<T>& dst) {
        auto& device = vulkanContext.getDevice();
        void* mappedData;
        vkMapMemory(device, vertexBufferMemory, 0, sizeof(T) * size, 0, &mappedData);
        size_t dataSize = sizeof(T) * std::min((uint32_t)dst.size(), (uint32_t)size);
        memcpy(dst.data(), mappedData, dataSize);
        vkUnmapMemory(device, vertexBufferMemory);
    }

    void zero()
    {
        auto& device = vulkanContext.getDevice();
        void* mappedData;
        vkMapMemory(device, vertexBufferMemory, 0, sizeof(T) * size, 0, &mappedData);
        memset(mappedData, 0, sizeof(T) * size);
        vkUnmapMemory(device, vertexBufferMemory);
    }

    void _recordZero(VkCommandBuffer commandBuffer) {
        auto& device = vulkanContext.getDevice();
        vkCmdFillBuffer(commandBuffer, vertexBuffer, 0, sizeof(T) * size, 0);
    }

    VkBuffer& getBuffer() {
        return vertexBuffer;
    }

    uint32_t getSize() const {
        return size;
    }

    size_t getBufferMemSize() const {
        return sizeof(T) * size;
    }

private:
    const uint32_t size; // Number of elements in the buffer

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VulkanContext& vulkanContext;


};

} // namespace klartraum

#endif // KLARTRAUM_VULKAN_BUFFER_HPP