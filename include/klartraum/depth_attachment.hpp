// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_DEPTH_ATTACHMENT_HPP
#define KLARTRAUM_DEPTH_ATTACHMENT_HPP

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "klartraum/computegraph/bufferelement.hpp"

namespace klartraum {

/** @brief Independent depth-test and depth-write controls; without an attachment neither is used. */
struct DepthState {
    bool test = true;
    bool write = true;
};

/** @brief Per-path D32_SFLOAT images shared by raster passes and Gaussian splatting. */
class DepthAttachment : public ComputeGraphElement {
public:
    DepthAttachment(VulkanContext& context, VkExtent2D extent, uint32_t paths, VkQueue queue = VK_NULL_HANDLE)
        : context(context),
          imageExtent(extent),
          images(paths),
          memories(paths),
          views(paths) {
        if (!extent.width || !extent.height || !paths) {
            throw std::invalid_argument("DepthAttachment: extent and path count must be positive");
        }
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(context.physicalDevice, format(), &properties);
        const VkFormatFeatureFlags required = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                              VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        if ((properties.optimalTilingFeatures & required) != required) {
            throw std::runtime_error("DepthAttachment: D32_SFLOAT attachments and transfers are not supported");
        }
        try {
            for (uint32_t path = 0; path < paths; ++path) {
                VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
                info.imageType = VK_IMAGE_TYPE_2D;
                info.format = format();
                info.extent = {extent.width, extent.height, 1};
                info.mipLevels = info.arrayLayers = 1;
                info.samples = VK_SAMPLE_COUNT_1_BIT;
                info.tiling = VK_IMAGE_TILING_OPTIMAL;
                info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                             VK_IMAGE_USAGE_TRANSFER_DST_BIT;
                context.setSharing(info);
                check(vkCreateImage(context.getDevice(), &info, nullptr, &images[path]), "create image");
                VkMemoryRequirements requirements{};
                vkGetImageMemoryRequirements(context.getDevice(), images[path], &requirements);
                VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                allocation.allocationSize = requirements.size;
                allocation.memoryTypeIndex =
                    context.findMemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                check(vkAllocateMemory(context.getDevice(), &allocation, nullptr, &memories[path]), "allocate memory");
                check(vkBindImageMemory(context.getDevice(), images[path], memories[path], 0), "bind memory");
                VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
                view.image = images[path];
                view.viewType = VK_IMAGE_VIEW_TYPE_2D;
                view.format = format();
                view.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
                check(vkCreateImageView(context.getDevice(), &view, nullptr, &views[path]), "create view");
            }
            context.submitImmediate(
                [&](VkCommandBuffer cmd) {
                    for (uint32_t path = 0; path < paths; ++path)
                        clear(cmd, path, VK_IMAGE_LAYOUT_UNDEFINED);
                },
                queue);
        } catch (...) {
            release();
            throw;
        }
    }

    ~DepthAttachment() { release(); }
    DepthAttachment(const DepthAttachment&) = delete;
    DepthAttachment& operator=(const DepthAttachment&) = delete;
    const char* getType() const override { return "DepthAttachment"; }
    static constexpr VkFormat format() { return VK_FORMAT_D32_SFLOAT; }
    VkExtent2D extent() const { return imageExtent; }
    VkImage getImage(uint32_t path) const { return images.at(path); }
    VkImageView getImageView(uint32_t path) const { return views.at(path); }
    void setClear(bool enabled) { clearEnabled = enabled; }
    void setClearDepth(float value) {
        if (!std::isfinite(value) || value < 0.0f || value > 1.0f)
            throw std::invalid_argument("DepthAttachment: clear depth must be in [0, 1]");
        clearValue = value;
    }

    void _setup(VulkanContext& context, uint32_t paths, VkQueue queue = VK_NULL_HANDLE) override {
        if (paths != images.size())
            throw std::invalid_argument("DepthAttachment: path count differs from graph");
        ComputeGraphElement::_setup(context, paths, queue);
    }
    void _record(VkCommandBuffer cmd, uint32_t path) override {
        if (clearEnabled)
            clear(cmd, path, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    }

    void transition(VkCommandBuffer cmd, uint32_t path, VkImageLayout from, VkImageLayout to) const {
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.oldLayout = from;
        barrier.newLayout = to;
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = getImage(path);
        barrier.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        barrier.srcAccessMask =
            from == VK_IMAGE_LAYOUT_UNDEFINED ? 0 : VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr,
                             0, nullptr, 1, &barrier);
    }

private:
    static void check(VkResult result, const char* operation) {
        if (result != VK_SUCCESS)
            throw std::runtime_error(std::string("DepthAttachment: failed to ") + operation);
    }
    void clear(VkCommandBuffer cmd, uint32_t path, VkImageLayout oldLayout) const {
        transition(cmd, path, oldLayout, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkClearDepthStencilValue value{clearValue, 0};
        VkImageSubresourceRange range{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        vkCmdClearDepthStencilImage(cmd, getImage(path), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &value, 1, &range);
        transition(cmd, path, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    }
    void release() {
        for (auto view : views)
            if (view)
                vkDestroyImageView(context.getDevice(), view, nullptr);
        for (auto image : images)
            if (image)
                vkDestroyImage(context.getDevice(), image, nullptr);
        for (auto memory : memories)
            if (memory)
                vkFreeMemory(context.getDevice(), memory, nullptr);
    }
    VulkanContext& context;
    VkExtent2D imageExtent;
    std::vector<VkImage> images;
    std::vector<VkDeviceMemory> memories;
    std::vector<VkImageView> views;
    bool clearEnabled = true;
    float clearValue = 1.0f;
};

/** @brief GPU-only bridge between a depth attachment and a per-path float storage buffer. */
class DepthTransfer : public ComputeGraphElement {
public:
    using Buffer = BufferElement<VulkanBuffer<float>>;
    DepthTransfer(std::shared_ptr<DepthAttachment> attachment, std::shared_ptr<Buffer> buffer, bool upload)
        : attachment(std::move(attachment)),
          buffer(std::move(buffer)),
          upload(upload) {
        setInput(this->buffer, 0);
        setInput(this->attachment, 1);
    }
    const char* getType() const override { return "DepthTransfer"; }
    void checkInput(ComputeGraphElementPtr, int = 0) override {}
    void _record(VkCommandBuffer cmd, uint32_t path) override {
        const auto layout = upload ? VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        attachment->transition(cmd, path, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, layout);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
        region.imageExtent = {attachment->extent().width, attachment->extent().height, 1};
        if (upload) {
            vkCmdCopyBufferToImage(cmd, buffer->getVkBuffer(path), attachment->getImage(path), layout, 1, &region);
        } else {
            vkCmdCopyImageToBuffer(cmd, attachment->getImage(path), layout, buffer->getVkBuffer(path), 1, &region);
        }
        attachment->transition(cmd, path, layout, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    }

private:
    std::shared_ptr<DepthAttachment> attachment;
    std::shared_ptr<Buffer> buffer;
    bool upload;
};

} // namespace klartraum

#endif // KLARTRAUM_DEPTH_ATTACHMENT_HPP
