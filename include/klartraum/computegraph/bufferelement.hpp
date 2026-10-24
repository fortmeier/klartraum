// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_COMPUTEGRAPH_BUFFERELEMENT_HPP
#define KLARTRAUM_COMPUTEGRAPH_BUFFERELEMENT_HPP

#include "klartraum/computegraph/computegraphelement.hpp"
#include "klartraum/vulkan_buffer.hpp"

namespace klartraum {

/// Untyped graph interface for a path's Vulkan buffer handle and byte capacity.
class BufferElementInterface : public ComputeGraphElement {
public:
    virtual void _setup(VulkanContext& vulkanContext, uint32_t numberPaths,
                        VkQueue setupQueue = VK_NULL_HANDLE) override = 0;

    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) override = 0;

    virtual const char* getType() const override { return "BufferElement"; }

    /// Returns the byte capacity of one buffer after setup.
    virtual size_t getBufferMemSize() const = 0;

    /// Returns the Vulkan handle for an initialized path.
    virtual VkBuffer& getVkBuffer(uint32_t pathId) = 0;

private:
};

/**
 * @brief Buffer element whose buffers have the type @p BufferType; getBuffer() returns the buffer of one path.
 */
template <typename BufferType>
class TemplatedBufferElementInterface : public BufferElementInterface {
public:
    /// Returns the typed allocation for an initialized path.
    virtual BufferType& getBuffer(uint32_t pathId) = 0;
};

/**
 * @brief Owns one independent buffer per compute-graph path.
 * @tparam BufferType Buffer implementation, usually VulkanBuffer<T>.
 *
 * Graph compilation creates the allocations. Access getBuffer(pathId) after
 * compilation and synchronize that path's GPU use before CPU transfers.
 * VulkanBuffer allocations default to coherent host-visible memory.
 */
template <typename BufferType>
class BufferElement : public TemplatedBufferElementInterface<BufferType> {
public:
    /// Configures per-path capacity using the buffer type's default usage and memory properties.
    BufferElement(VulkanContext& vulkanContext, uint32_t numberElements)
        : vulkanContext(vulkanContext),
          numberElements(numberElements) {}

    /// Configures GPU usage and per-path capacity, with coherent host-visible memory.
    BufferElement(VulkanContext& vulkanContext, uint32_t numberElements, VkBufferUsageFlags flags)
        : vulkanContext(vulkanContext),
          numberElements(numberElements),
          bufferUsageFlags(flags) {}

    /**
     * @brief Configures per-path buffers with explicit usage and memory properties.
     * @param vulkanContext Context owning the device; it must outlive the allocations.
     * @param numberElements Capacity of each buffer in elements.
     * @param flags GPU buffer usage flags.
     * @param memory Required memory properties; pass DEVICE_LOCAL without HOST_VISIBLE for staged CPU access.
     */
    BufferElement(VulkanContext& vulkanContext, uint32_t numberElements, VkBufferUsageFlags flags,
                  VkMemoryPropertyFlags memory)
        : vulkanContext(vulkanContext),
          numberElements(numberElements),
          bufferUsageFlags(flags),
          memoryProperties(memory) {}

    // A buffer element may be part of several compiled graphs, e.g. the links of a GraphChain that pass data on
    // the GPU. Its buffers are allocated by the first graph and reused by the others, which must have as many
    // paths (as TensorElement does). Review: whether reusing them is right for every graph that shares an element.
    virtual void _setup(VulkanContext& vulkanContext, uint32_t numberPaths, VkQueue setupQueue = VK_NULL_HANDLE) {
        if (!buffers.empty()) {
            if (buffers.size() != numberPaths) {
                throw std::runtime_error("BufferElement: shared by graphs with differing path counts");
            }
            return;
        }
        buffers.reserve(numberPaths);
        for (uint32_t i = 0; i < numberPaths; i++) {
            if (bufferUsageFlags == VK_BUFFER_USAGE_FLAG_BITS_MAX_ENUM) {
                buffers.emplace_back(vulkanContext, numberElements);
            } else {
                // Separate allocations let each path update its data while other frames are in flight.
                buffers.emplace_back(vulkanContext, numberElements, bufferUsageFlags, memoryProperties);
            }
        }
    };

    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) {
        if (recordToZeroRange) {
            buffers[pathId]._recordZero(commandBuffer, zeroRangeOffset, zeroRangeSize);
        } else if (recordToZero) {
            buffers[pathId]._recordZero(commandBuffer);
        } else if (recordToFill) {
            buffers[pathId]._recordFill(commandBuffer, fillValue);
        }
    };

    virtual const char* getType() const { return "BufferElement"; }

    virtual size_t getBufferMemSize() const { return buffers[0].getBufferMemSize(); }

    /// Returns the allocation for pathId after graph compilation.
    virtual BufferType& getBuffer(uint32_t pathId) { return buffers[pathId]; }

    /// Immediately clears every path through the underlying buffer's zero() helper.
    void zero() {
        for (auto& buffer : buffers) {
            buffer.zero();
        }
    }

    /// Configures whole-buffer GPU zeroing on each submission; set before graph compilation.
    void setRecordToZero(bool _setToZero) { recordToZero = _setToZero; }

    /**
     * @brief Configures a GPU-cleared byte range on each submission, preserving other fields.
     * @param byteOffset Start of the range in bytes.
     * @param byteSize Length of the range in bytes.
     * @note Set before graph compilation. Range zeroing takes precedence over whole-buffer zeroing and filling.
     */
    void setRecordToZeroRange(VkDeviceSize byteOffset, VkDeviceSize byteSize) {
        recordToZeroRange = true;
        zeroRangeOffset = byteOffset;
        zeroRangeSize = byteSize;
    }

    /**
     * @brief Configures a repeating 32-bit GPU fill pattern for each submission.
     * @param value Pattern to repeat, e.g. a sort-key sentinel.
     * @note Set before graph compilation. Zeroing options take precedence over filling.
     */
    void setRecordToFill(uint32_t value) {
        recordToFill = true;
        fillValue = value;
    }

    virtual VkBuffer& getVkBuffer(uint32_t pathId) { return buffers[pathId].getBuffer(); };

private:
    VulkanContext& vulkanContext;
    uint32_t numberPaths = 0;
    uint32_t numberElements = 0;
    std::vector<BufferType> buffers;
    bool recordToZero = false;
    bool recordToZeroRange = false;
    VkDeviceSize zeroRangeOffset = 0;
    VkDeviceSize zeroRangeSize = 0;
    bool recordToFill = false;
    uint32_t fillValue = 0;
    VkBufferUsageFlags bufferUsageFlags = VK_BUFFER_USAGE_FLAG_BITS_MAX_ENUM;
    // Coherent mapped storage is the default for direct CPU updates to a render path's buffer.
    VkMemoryPropertyFlags memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
};

/**
 * @brief Owns one buffer shared by every compute-graph path.
 * @tparam BufferType Buffer implementation, usually VulkanBuffer<T>.
 *
 * Construction forwards its arguments to BufferType and allocates immediately.
 * The path index is ignored. Use shared storage for data such as immutable models;
 * writes must be synchronized with GPU use across all paths.
 */
template <typename BufferType>
class BufferElementSinglePath : public TemplatedBufferElementInterface<BufferType> {
public:
    /// Constructs the shared buffer with the supplied allocation arguments.
    template <typename... Args>
    BufferElementSinglePath(Args&&... args)
        : buffer(std::forward<Args>(args)...) {}

    virtual void _setup(VulkanContext& vulkanContext, uint32_t numberPaths, VkQueue setupQueue = VK_NULL_HANDLE) {};

    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) {};

    virtual const char* getType() const { return "BufferElementSinglePath"; }

    virtual size_t getBufferMemSize() const { return buffer.getBufferMemSize(); }

    /// Returns the shared typed allocation; pathId is ignored.
    BufferType& getBuffer(uint32_t pathId = 0) { return buffer; }

    /// Returns the shared Vulkan handle; pathId is ignored.
    virtual VkBuffer& getVkBuffer(uint32_t pathId = 0) { return buffer.getBuffer(); };

private:
    BufferType buffer;
};

/** @brief Buffer element holding the arguments of an indirect compute dispatch (vkCmdDispatchIndirect). */
using DispatchIndirectCommandBufferElement = BufferElement<VulkanBuffer<VkDispatchIndirectCommand>>;

/** @brief Buffer element holding the arguments of an indirect draw (vkCmdDrawIndirect). */
using DrawIndirectCommandBufferElement = BufferElement<VulkanBuffer<VkDrawIndirectCommand>>;

} // namespace klartraum

#endif // KLARTRAUM_COMPUTEGRAPH_BUFFERELEMENT_HPP
