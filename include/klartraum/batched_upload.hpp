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
        : vulkanContext_(vulkanContext), queue_(queue), chunkBytes_(std::max<VkDeviceSize>(chunkBytes, 1)) {}

    ~BatchedUpload() { releaseStaging(); }

    BatchedUpload(const BatchedUpload&) = delete;
    BatchedUpload& operator=(const BatchedUpload&) = delete;

    // Copies `bytes` of `data` to the start of `destination`, a buffer with
    // TRANSFER_DST usage, with a later submission.
    void add(VkBuffer destination, const void* data, VkDeviceSize bytes) {
        if (bytes == 0) {
            return;
        }
        if (used_ + bytes > capacity_) {
            submit();
            reserve(std::max(chunkBytes_, bytes));
        }
        std::memcpy(static_cast<char*>(mapped_) + used_, data, static_cast<size_t>(bytes));
        copies_.push_back(Copy{destination, used_, bytes});
        used_ += bytes;
    }

    // Copies everything added since the last submission and waits for it.
    void submit() {
        if (copies_.empty()) {
            return;
        }
        vulkanContext_.submitImmediate([&](VkCommandBuffer commandBuffer) {
            for (const Copy& copy : copies_) {
                VkBufferCopy region{};
                region.srcOffset = copy.offset;
                region.size = copy.bytes;
                vkCmdCopyBuffer(commandBuffer, staging_, copy.destination, 1, &region);
            }
        }, queue_);
        ++submissions_;
        copies_.clear();
        used_ = 0;
        // A staging buffer enlarged for one large upload is not kept.
        if (capacity_ > chunkBytes_) {
            releaseStaging();
        }
    }

    // How many submissions copied data so far.
    uint32_t submissions() const { return submissions_; }

private:
    struct Copy {
        VkBuffer destination;
        VkDeviceSize offset;  // in the staging buffer
        VkDeviceSize bytes;
    };

    void reserve(VkDeviceSize bytes) {
        if (capacity_ >= bytes) {
            return;
        }
        releaseStaging();
        vulkanContext_.createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                    staging_, stagingMemory_);
        if (vkMapMemory(vulkanContext_.getDevice(), stagingMemory_, 0, bytes, 0, &mapped_) != VK_SUCCESS) {
            releaseStaging();
            throw std::runtime_error("failed to map an upload staging buffer");
        }
        capacity_ = bytes;
    }

    void releaseStaging() {
        auto& device = vulkanContext_.getDevice();
        if (mapped_ != nullptr) {
            vkUnmapMemory(device, stagingMemory_);
            mapped_ = nullptr;
        }
        if (staging_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device, staging_, nullptr);
            staging_ = VK_NULL_HANDLE;
        }
        if (stagingMemory_ != VK_NULL_HANDLE) {
            vkFreeMemory(device, stagingMemory_, nullptr);
            stagingMemory_ = VK_NULL_HANDLE;
        }
        capacity_ = 0;
        used_ = 0;
    }

    VulkanContext& vulkanContext_;
    VkQueue queue_;
    VkDeviceSize chunkBytes_;
    VkBuffer staging_ = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory_ = VK_NULL_HANDLE;
    void* mapped_ = nullptr;
    VkDeviceSize capacity_ = 0;
    VkDeviceSize used_ = 0;
    std::vector<Copy> copies_;
    uint32_t submissions_ = 0;
};

} // namespace klartraum

#endif // KLARTRAUM_BATCHED_UPLOAD_HPP
