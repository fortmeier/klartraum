// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_BATCHED_UPLOAD_HPP
#define KLARTRAUM_BATCHED_UPLOAD_HPP

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "klartraum/vulkan_context.hpp"

namespace klartraum {

/**
 * @brief Uploads host data into device-local buffers with few submissions.
 *
 * Each add() copies the data into a staging buffer of `chunkBytes` (or more,
 * for data larger than that); when the staging buffer is full, its copies are
 * submitted together with one submitImmediate(), which waits for them.
 * submit() copies what is left. Data that was added but not submitted when
 * the BatchedUpload is destroyed is not copied.
 *
 * Every immediate submission waits for the queue to work through what was
 * submitted before it, e.g. frames waiting for the swapchain; uploading many
 * buffers, such as an ONNX network's weights, one submission each is
 * therefore slow.
 */
class BatchedUpload {
public:
    static constexpr VkDeviceSize kDefaultChunkBytes = VkDeviceSize{128} << 20;

    // `queue` takes the copies (default: the graphics queue; see
    // VulkanContext::submitImmediate()).
    explicit BatchedUpload(VulkanContext& vulkanContext, VkQueue queue = VK_NULL_HANDLE,
                           VkDeviceSize chunkBytes = kDefaultChunkBytes)
        : vulkanContext(vulkanContext),
          queue(queue),
          chunkBytes(std::max<VkDeviceSize>(chunkBytes, 1)) {}

    ~BatchedUpload() { releaseStaging(); }

    BatchedUpload(const BatchedUpload&) = delete;
    BatchedUpload& operator=(const BatchedUpload&) = delete;

    // Copies `bytes` of `data` to `destination` at `destinationOffset`
    // (default: its start), a buffer with TRANSFER_DST usage, with a later
    // submission.
    void add(VkBuffer destination, const void* data, VkDeviceSize bytes, VkDeviceSize destinationOffset = 0) {
        if (bytes == 0) {
            return;
        }
        if (used + bytes > capacity) {
            submit();
            reserve(std::max(chunkBytes, bytes));
        }
        std::memcpy(static_cast<char*>(mapped) + used, data, static_cast<size_t>(bytes));
        copies.push_back(Copy{destination, used, bytes, destinationOffset});
        used += bytes;
    }

    // Copies everything added since the last submission and waits for it.
    void submit() {
        if (copies.empty()) {
            return;
        }
        vulkanContext.submitImmediate(
            [&](VkCommandBuffer commandBuffer) {
                for (const Copy& copy : copies) {
                    VkBufferCopy region{};
                    region.srcOffset = copy.offset;
                    region.dstOffset = copy.destinationOffset;
                    region.size = copy.bytes;
                    vkCmdCopyBuffer(commandBuffer, staging, copy.destination, 1, &region);
                }
            },
            queue);
        ++submissionCount;
        copies.clear();
        used = 0;
        // A staging buffer enlarged for one large upload is not kept.
        if (capacity > chunkBytes) {
            releaseStaging();
        }
    }

    // How many submissions copied data so far.
    uint32_t submissions() const { return submissionCount; }

private:
    struct Copy {
        VkBuffer destination;
        VkDeviceSize offset; // in the staging buffer
        VkDeviceSize bytes;
        VkDeviceSize destinationOffset;
    };

    void reserve(VkDeviceSize bytes) {
        if (capacity >= bytes) {
            return;
        }
        releaseStaging();
        vulkanContext.createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging,
                                   stagingMemory);
        if (vkMapMemory(vulkanContext.getDevice(), stagingMemory, 0, bytes, 0, &mapped) != VK_SUCCESS) {
            releaseStaging();
            throw std::runtime_error("failed to map an upload staging buffer");
        }
        capacity = bytes;
    }

    void releaseStaging() {
        auto& device = vulkanContext.getDevice();
        if (mapped != nullptr) {
            vkUnmapMemory(device, stagingMemory);
            mapped = nullptr;
        }
        if (staging != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, staging, nullptr);
            staging = VK_NULL_HANDLE;
        }
        if (stagingMemory != VK_NULL_HANDLE) {
            vkFreeMemory(device, stagingMemory, nullptr);
            stagingMemory = VK_NULL_HANDLE;
        }
        capacity = 0;
        used = 0;
    }

    VulkanContext& vulkanContext;
    VkQueue queue;
    VkDeviceSize chunkBytes;
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize capacity = 0;
    VkDeviceSize used = 0;
    std::vector<Copy> copies;
    uint32_t submissionCount = 0;
};

} // namespace klartraum

#endif // KLARTRAUM_BATCHED_UPLOAD_HPP
