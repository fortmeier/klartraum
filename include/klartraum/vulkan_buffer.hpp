// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_VULKAN_BUFFER_HPP
#define KLARTRAUM_VULKAN_BUFFER_HPP

#include <vulkan/vulkan.h>

#include <algorithm>
#include <functional>
#include <cstring>

#include <klartraum/vulkan_context.hpp>

namespace klartraum {

template <typename T>
class VulkanBuffer {
public:
    // Allocate size T elements with the requested GPU usage and memory properties.
    // Host-visible storage accepts mapped copies; unmapped storage uses synchronous staging.
    // Host-visible allocations must be coherent: mapped copies do not flush or invalidate.
    VulkanBuffer(VulkanContext& kernel, uint32_t size,
                 VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VkMemoryPropertyFlags memory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        : vulkanContext(kernel),
          size(size),
          hostVisible((memory & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) {
        auto& device = kernel.getDevice();

        if (size == 0) {
            throw std::invalid_argument("Buffer size must be greater than 0");
        }

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = sizeof(T) * size;
        // Every allocation supports initialization copies, GPU-to-GPU copies and readback,
        // independently of the storage/indirect usage requested by its consumer.
        bufferInfo.usage = usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(device, &bufferInfo, nullptr, &vertexBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute buffer!");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, vertexBuffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = vulkanContext.findMemoryType(memRequirements.memoryTypeBits, memory);
        // Expose the selected memory type's properties for allocation inspection.
        VkPhysicalDeviceMemoryProperties properties{};
        vkGetPhysicalDeviceMemoryProperties(vulkanContext.getPhysicalDevice(), &properties);
        memoryProperties = properties.memoryTypes[allocInfo.memoryTypeIndex].propertyFlags;

        if (vkAllocateMemory(device, &allocInfo, nullptr, &vertexBufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate vertex buffer memory!");
        }
        vkBindBufferMemory(device, vertexBuffer, vertexBufferMemory, 0);
    }

    // Preserve the allocation's memory policy so copy helpers access it correctly after a move.
    VulkanBuffer(VulkanBuffer&& other) noexcept
        : size(other.size),
          vertexBuffer(other.vertexBuffer),
          vertexBufferMemory(other.vertexBufferMemory),
          vulkanContext(other.vulkanContext),
          hostVisible(other.hostVisible),
          memoryProperties(other.memoryProperties) {
        other.vertexBuffer = VK_NULL_HANDLE;
        other.vertexBufferMemory = VK_NULL_HANDLE;
    }

    ~VulkanBuffer() {
        auto& device = vulkanContext.getDevice();
        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);
    }

    // Upload the vector's elements, limited to this buffer's capacity.
    void memcopyFrom(const std::vector<T>& src) { memcopyFrom(src.data(), src.size()); }

    // count is in T elements. The caller must ensure earlier GPU use of this buffer has finished.
    void memcopyFrom(const T* src, size_t count) {
        if (!hostVisible) {
            // CPU data is mapped into a temporary coherent buffer, then copied to GPU storage.
            // commands() waits for completion before the temporary allocation is destroyed.
            size_t n = std::min(count, size_t(size));
            if (!n)
                return;
            VulkanBuffer<T> staging(vulkanContext, uint32_t(n));
            staging.memcopyFrom(src, n);
            commands([&](VkCommandBuffer cmd) {
                VkBufferCopy copy{0, 0, n * sizeof(T)};
                vkCmdCopyBuffer(cmd, staging.vertexBuffer, vertexBuffer, 1, &copy);
            });
            return;
        }
        auto& device = vulkanContext.getDevice();
        void* mappedData;
        vkMapMemory(device, vertexBufferMemory, 0, sizeof(T) * size, 0, &mappedData);
        size_t dataSize = sizeof(T) * std::min((uint32_t)count, size);
        memcpy(mappedData, src, dataSize);
        vkUnmapMemory(device, vertexBufferMemory);
    }

    // Upload count raw bytes, limited to the allocation's byte capacity regardless of T.
    void memcopyFrom(const char* src, size_t count) {
        if (!hostVisible) {
            // Byte-sized staging lets the transfer use the raw byte count rather than T elements.
            size_t n = std::min(count, sizeof(T) * size_t(size));
            if (!n)
                return;
            VulkanBuffer<uint8_t> staging(vulkanContext, uint32_t(n));
            staging.memcopyFrom(reinterpret_cast<const uint8_t*>(src), n);
            commands([&](VkCommandBuffer cmd) {
                VkBufferCopy copy{0, 0, n};
                vkCmdCopyBuffer(cmd, staging.getBuffer(), vertexBuffer, 1, &copy);
            });
            return;
        }
        auto& device = vulkanContext.getDevice();
        void* mappedData;
        vkMapMemory(device, vertexBufferMemory, 0, sizeof(T) * size, 0, &mappedData);
        size_t dataSize = std::min(count, sizeof(T) * size_t(size));
        memcpy(mappedData, src, dataSize);
        vkUnmapMemory(device, vertexBufferMemory);
    }

    // Explicit readback for tests or inspection; call after the producing GPU work completes.
    void memcopyTo(std::vector<T>& dst) {
        if (!hostVisible) {
            // Copy into coherent mapped storage and wait before exposing the result to the CPU.
            size_t n = std::min(dst.size(), size_t(size));
            if (!n)
                return;
            VulkanBuffer<T> staging(vulkanContext, uint32_t(n));
            commands([&](VkCommandBuffer cmd) {
                VkBufferCopy copy{0, 0, n * sizeof(T)};
                vkCmdCopyBuffer(cmd, vertexBuffer, staging.vertexBuffer, 1, &copy);
            });
            staging.memcopyTo(dst);
            return;
        }
        auto& device = vulkanContext.getDevice();
        void* mappedData;
        vkMapMemory(device, vertexBufferMemory, 0, sizeof(T) * size, 0, &mappedData);
        size_t dataSize = sizeof(T) * std::min((uint32_t)dst.size(), (uint32_t)size);
        memcpy(dst.data(), mappedData, dataSize);
        vkUnmapMemory(device, vertexBufferMemory);
    }

    void zero() {
        if (!hostVisible) {
            // Fill unmapped storage directly on the GPU to initialize it to zero.
            commands([&](VkCommandBuffer cmd) { vkCmdFillBuffer(cmd, vertexBuffer, 0, VK_WHOLE_SIZE, 0); });
            return;
        }
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

    VkBuffer& getBuffer() { return vertexBuffer; }

    uint32_t getSize() const { return size; }

    size_t getBufferMemSize() const { return sizeof(T) * size; }

    // Properties of the actual Vulkan memory type backing this allocation.
    VkMemoryPropertyFlags getMemoryProperties() const { return memoryProperties; }

private:
    // Synchronous staging is for initialization and explicit readback only.
    // Per-frame GPU-only buffers stay in device-local memory.
    void commands(const std::function<void(VkCommandBuffer)>& record) {
        auto device = vulkanContext.getDevice();
        VkCommandBuffer cmd{};
        // A one-time command buffer carries the transfer on the graph's graphics queue.
        VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocation.commandPool = vulkanContext.getCommandPool();
        allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocation.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(device, &allocation, &cmd) != VK_SUCCESS)
            throw std::runtime_error("Cannot allocate buffer transfer command");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &begin);
        record(cmd);
        // Make transfer writes visible to later GPU consumers and mapped staging readback.
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0,
                             nullptr, 0, nullptr);
        vkEndCommandBuffer(cmd);
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &cmd;
        VkResult result = vkQueueSubmit(vulkanContext.getGraphicsQueue(), 1, &submit, VK_NULL_HANDLE);
        // Wait for the transfer before releasing its command buffer and temporary staging storage.
        // This stalls the graphics queue, so use coherent mapped buffers for per-frame uploads.
        if (result == VK_SUCCESS)
            result = vkQueueWaitIdle(vulkanContext.getGraphicsQueue());
        vkFreeCommandBuffers(device, vulkanContext.getCommandPool(), 1, &cmd);
        if (result != VK_SUCCESS)
            throw std::runtime_error("Device-local buffer transfer failed");
    }

public:
    // Copy the source's first count T elements, e.g. static data into each render path's output.
    // Destination capacity bounds the copy; the source must provide that many readable elements.
    // Both buffers must be available for the transfer; the copy completes before this call returns.
    void copyFrom(VkBuffer source, uint32_t count) {
        VkDeviceSize bytes = sizeof(T) * std::min(count, size);
        if (!bytes)
            return;
        commands([&](VkCommandBuffer cmd) {
            VkBufferCopy copy{0, 0, bytes};
            vkCmdCopyBuffer(cmd, source, vertexBuffer, 1, &copy);
        });
    }

private:
    const uint32_t size; // Number of elements in the buffer

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    VulkanContext& vulkanContext;
    // Whether copy helpers can map this allocation directly or need a staging buffer.
    bool hostVisible = true;
    VkMemoryPropertyFlags memoryProperties = 0;
};

} // namespace klartraum

#endif // KLARTRAUM_VULKAN_BUFFER_HPP
