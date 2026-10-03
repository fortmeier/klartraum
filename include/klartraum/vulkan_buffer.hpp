#ifndef KLARTRAUM_VULKAN_BUFFER_HPP
#define KLARTRAUM_VULKAN_BUFFER_HPP

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <klartraum/vulkan_context.hpp>

namespace klartraum {

template <typename T>
class VulkanBuffer {
public:
    VulkanBuffer(
        VulkanContext& kernel,
        uint32_t size,
        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VkMemoryPropertyFlags memoryProperties =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        : size(size),
          usageFlags(usage),
          memoryProperties(memoryProperties),
          vulkanContext(kernel) {
        if (size == 0) {
            throw std::invalid_argument("Buffer size must be greater than 0");
        }
        vulkanContext.createBuffer(
            sizeof(T) * static_cast<VkDeviceSize>(size), usage, memoryProperties,
            vertexBuffer, vertexBufferMemory);
    }

    VulkanBuffer(VulkanBuffer&& other) noexcept
        : size(other.size),
          vertexBuffer(other.vertexBuffer),
          vertexBufferMemory(other.vertexBufferMemory),
          usageFlags(other.usageFlags),
          memoryProperties(other.memoryProperties),
          vulkanContext(other.vulkanContext) {
        other.vertexBuffer = VK_NULL_HANDLE;
        other.vertexBufferMemory = VK_NULL_HANDLE;
    }

    ~VulkanBuffer() {
        auto& device = vulkanContext.getDevice();
        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);
    }

    // Uploads and downloads of a device-local buffer go through a staging
    // buffer, copied on `queue` (default: the graphics queue; see
    // VulkanContext::submitImmediate()). Host-visible buffers are mapped.
    void memcopyFrom(const std::vector<T>& src, VkQueue queue = VK_NULL_HANDLE) {
        size_t dataSize = sizeof(T) * std::min((uint32_t)src.size(), (uint32_t)size);
        upload(src.data(), dataSize, queue);
    }

    void memcopyFrom(const T* src, size_t count, VkQueue queue = VK_NULL_HANDLE) {
        size_t dataSize = sizeof(T) * std::min((uint32_t)count, size);
        upload(src, dataSize, queue);
    }

    void memcopyFrom(const char* src, size_t count, VkQueue queue = VK_NULL_HANDLE) {
        size_t dataSize = std::min(count, sizeof(T) * size_t(size));
        upload(src, dataSize, queue);
    }

    void memcopyTo(std::vector<T>& dst, VkQueue queue = VK_NULL_HANDLE) {
        size_t dataSize = sizeof(T) * std::min((uint32_t)dst.size(), (uint32_t)size);
        download(dst.data(), dataSize, queue);
    }

    void zero(VkQueue queue = VK_NULL_HANDLE)
    {
        std::vector<T> zeros(size);
        upload(zeros.data(), sizeof(T) * size, queue);
    }

    void _recordZero(VkCommandBuffer commandBuffer) {
        auto& device = vulkanContext.getDevice();
        vkCmdFillBuffer(commandBuffer, vertexBuffer, 0, sizeof(T) * size, 0);
    }

    // Fills the whole buffer with a repeating 32-bit pattern each frame — e.g.
    // resetting a uint sort-key buffer to 0xFFFFFFFF (a sentinel guaranteed to
    // sort after any encoded depth key) so stale entries from a previous
    // frame's larger visible-splat count never contaminate this frame's sort.
    void _recordFill(VkCommandBuffer commandBuffer, uint32_t value) {
        vkCmdFillBuffer(commandBuffer, vertexBuffer, 0, sizeof(T) * size, value);
    }

    // Zeroes only [byteOffset, byteOffset + byteSize) — e.g. to reset a single
    // field of a struct buffer (such as VkDrawIndirectCommand::instanceCount)
    // each frame while leaving the rest of the buffer untouched.
    void _recordZero(VkCommandBuffer commandBuffer, VkDeviceSize byteOffset, VkDeviceSize byteSize) {
        vkCmdFillBuffer(commandBuffer, vertexBuffer, byteOffset, byteSize, 0);
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

    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;
    VkBufferUsageFlags usageFlags;
    VkMemoryPropertyFlags memoryProperties;
    VulkanContext& vulkanContext;

    bool isHostVisible() const {
        return (memoryProperties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    }

    void upload(const void* source, size_t byteCount, VkQueue queue) {
        if (byteCount == 0) return;
        auto& device = vulkanContext.getDevice();

        if (isHostVisible()) {
            void* mappedData = nullptr;
            if (vkMapMemory(device, vertexBufferMemory, 0, byteCount, 0, &mappedData) != VK_SUCCESS) {
                throw std::runtime_error("failed to map Vulkan buffer for upload");
            }
            std::memcpy(mappedData, source, byteCount);
            vkUnmapMemory(device, vertexBufferMemory);
            return;
        }
        if ((usageFlags & VK_BUFFER_USAGE_TRANSFER_DST_BIT) == 0) {
            throw std::runtime_error("device-local buffer upload requires TRANSFER_DST usage");
        }

        VkBuffer stagingBuffer = VK_NULL_HANDLE;
        VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
        vulkanContext.createBuffer(
            byteCount, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuffer, stagingMemory);
        try {
            void* mappedData = nullptr;
            if (vkMapMemory(device, stagingMemory, 0, byteCount, 0, &mappedData) != VK_SUCCESS) {
                throw std::runtime_error("failed to map Vulkan staging buffer for upload");
            }
            std::memcpy(mappedData, source, byteCount);
            vkUnmapMemory(device, stagingMemory);
            vulkanContext.copyBufferImmediate(stagingBuffer, vertexBuffer, byteCount, queue);
        } catch (...) {
            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingMemory, nullptr);
            throw;
        }
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingMemory, nullptr);
    }

    void download(void* destination, size_t byteCount, VkQueue queue) {
        if (byteCount == 0) return;
        auto& device = vulkanContext.getDevice();

        if (isHostVisible()) {
            void* mappedData = nullptr;
            if (vkMapMemory(device, vertexBufferMemory, 0, byteCount, 0, &mappedData) != VK_SUCCESS) {
                throw std::runtime_error("failed to map Vulkan buffer for download");
            }
            std::memcpy(destination, mappedData, byteCount);
            vkUnmapMemory(device, vertexBufferMemory);
            return;
        }
        if ((usageFlags & VK_BUFFER_USAGE_TRANSFER_SRC_BIT) == 0) {
            throw std::runtime_error("device-local buffer download requires TRANSFER_SRC usage");
        }

        VkBuffer stagingBuffer = VK_NULL_HANDLE;
        VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
        vulkanContext.createBuffer(
            byteCount, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuffer, stagingMemory);
        try {
            vulkanContext.copyBufferImmediate(vertexBuffer, stagingBuffer, byteCount, queue);
            void* mappedData = nullptr;
            if (vkMapMemory(device, stagingMemory, 0, byteCount, 0, &mappedData) != VK_SUCCESS) {
                throw std::runtime_error("failed to map Vulkan staging buffer for download");
            }
            std::memcpy(destination, mappedData, byteCount);
            vkUnmapMemory(device, stagingMemory);
        } catch (...) {
            vkDestroyBuffer(device, stagingBuffer, nullptr);
            vkFreeMemory(device, stagingMemory, nullptr);
            throw;
        }
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingMemory, nullptr);
    }


};

} // namespace klartraum

#endif // KLARTRAUM_VULKAN_BUFFER_HPP
