#ifndef KLARTRAUM_VULKAN_BUFFER_HPP
#define KLARTRAUM_VULKAN_BUFFER_HPP

#include <vulkan/vulkan.h>

#include <klartraum/vulkan_kernel.hpp>

namespace klartraum {

template <typename T>
class VulkanBuffer {
public:
    VulkanBuffer(VulkanKernel& kernel, uint32_t size) : vulkanKernel(kernel), size(size) { //, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties);
        auto& device = kernel.getDevice();

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = sizeof(T) * size;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT ;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
        if (vkCreateBuffer(device, &bufferInfo, nullptr, &vertexBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute buffer!");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, vertexBuffer, &memRequirements);
    
        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = vulkanKernel.findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    
        if (vkAllocateMemory(device, &allocInfo, nullptr, &vertexBufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate vertex buffer memory!");
        }        
    }

    ~VulkanBuffer() {
        auto& device = vulkanKernel.getDevice();
        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);
    }

    void memcopy_from(const std::vector<T>& src) {
        auto& device = vulkanKernel.getDevice();
        void* mappedData;
        vkMapMemory(device, vertexBufferMemory, 0, sizeof(T) * size, 0, &mappedData);
        memcpy(mappedData, src.data(), sizeof(T) * size);
        vkUnmapMemory(device, vertexBufferMemory);
    }

    void memcopy_to(std::vector<T>& dst) {
        auto& device = vulkanKernel.getDevice();
        void* mappedData;
        vkMapMemory(device, vertexBufferMemory, 0, sizeof(T) * size, 0, &mappedData);
        memcpy(dst.data(), mappedData, sizeof(T) * size);
        vkUnmapMemory(device, vertexBufferMemory);
    }

    VkBuffer& getBuffer() {
        return vertexBuffer;
    }

    size_t getBufferMemSize() {
        return sizeof(T) * size;
    }

private:
    const uint32_t size;

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VulkanKernel& vulkanKernel;


};

} // namespace klartraum

#endif // KLARTRAUM_VULKAN_BUFFER_HPP