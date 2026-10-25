// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_IPC_SHARED_BUFFER_ELEMENT_HPP
#define KLARTRAUM_IPC_SHARED_BUFFER_ELEMENT_HPP

#include <memory>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/ipc/shared_memory.hpp"

namespace klartraum::ipc {

/**
 * @brief A buffer element whose data lives in shared memory, for graphs that run in another process.
 *
 * Describing a graph for a RemoteSession names the element's SharedMemoryRegion;
 * the server maps the same pages and uses them as the buffer, so the inputs the
 * application writes with data() reach the remote shaders, and their results
 * appear in data(), without copies.
 *
 * The element can also be part of a graph compiled locally: its setup then
 * imports the region as the memory of one VkBuffer shared by all paths, which
 * needs VK_EXT_external_memory_host (VulkanContext::isExternalMemoryHostSupported()).
 */
class SharedBufferElement : public BufferElementInterface {
public:
    /// A zero-filled buffer of `byteSize` bytes in a new shared-memory region.
    explicit SharedBufferElement(size_t byteSize);

    /// A zero-filled buffer of `count` elements of type T.
    template <typename T>
    static std::shared_ptr<SharedBufferElement> create(size_t count) {
        return std::make_shared<SharedBufferElement>(count * sizeof(T));
    }

    ~SharedBufferElement();

    const char* getType() const override { return "SharedBufferElement"; }

    void checkInput(ComputeGraphElementPtr input, int index = 0) override;

    void _setup(VulkanContext& vulkanContext, uint32_t numberPaths, VkQueue setupQueue = VK_NULL_HANDLE) override;

    void _record(VkCommandBuffer commandBuffer, uint32_t pathId) override {}

    size_t getBufferMemSize() const override { return byteSize; }

    /// The buffer importing the region (after a local setup); the same for every path.
    VkBuffer& getVkBuffer(uint32_t pathId) override;

    /// The data, `getBufferMemSize()` bytes. Write inputs before and read results after
    /// an execution; the GPU (local or remote) uses the same memory while it runs.
    template <typename T>
    T* data() {
        return region.as<T>();
    }

    const SharedMemoryRegion& getRegion() const { return region; }

private:
    /// The shared memory holding the data, rounded up to the region alignment.
    SharedMemoryRegion region;
    /// The buffer's size in bytes, at most the region's.
    size_t byteSize;

    /// For a local setup: the device, the buffer and the imported memory.
    VulkanContext* vulkanContext = nullptr;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
};

} // namespace klartraum::ipc

#endif // KLARTRAUM_IPC_SHARED_BUFFER_ELEMENT_HPP
