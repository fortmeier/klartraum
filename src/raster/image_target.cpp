// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/raster/image_target.hpp"

#include <stdexcept>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"

namespace klartraum::raster {

namespace {

struct ImageTargetPushConstants {
    uint32_t pixels, view, bgra;
};

// Copies its image (input 0, in `layout` before and after) into its buffer (input 1, one uint per pixel).
class ImageCopy : public ComputeGraphElement {
public:
    explicit ImageCopy(VkImageLayout layout)
        : layout(layout) {}
    const char* getType() const override { return "ImageCopy"; }
    void checkInput(ComputeGraphElementPtr input, int index) override {}
    void _record(VkCommandBuffer commandBuffer, uint32_t pathId) override {
        auto image = std::dynamic_pointer_cast<ImageViewSrc>(getInputElement(0));
        auto buffer = std::dynamic_pointer_cast<BufferElementInterface>(getInputElement(1));
        const VkExtent2D extent = image->getImageExtent(pathId);
        transition(commandBuffer, image->getImage(pathId), layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {extent.width, extent.height, 1};
        vkCmdCopyImageToBuffer(commandBuffer, image->getImage(pathId), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               buffer->getVkBuffer(pathId), 1, &region);
        transition(commandBuffer, image->getImage(pathId), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, layout);
        VkMemoryBarrier copied{};
        copied.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        copied.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        copied.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                             &copied, 0, nullptr, 0, nullptr);
    }

private:
    static void transition(VkCommandBuffer commandBuffer, VkImage image, VkImageLayout from, VkImageLayout to) {
        if (from == to)
            return;
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = from;
        barrier.newLayout = to;
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                             0, nullptr, 0, nullptr, 1, &barrier);
    }

    // The image's layout before and after the copy.
    VkImageLayout layout;
};

} // namespace

ComputeGraphElementPtr imageTarget(VulkanContext& vulkanContext, const std::shared_ptr<ImageViewSrc>& image,
                                   VkImageLayout layout, const ComputeGraphElementPtr& renderer,
                                   const std::shared_ptr<TensorElement<float>>& targets, uint32_t view) {
    const VkExtent2D extent = image->getImageExtent(0);
    const uint32_t pixels = extent.width * extent.height;
    if (targets->getDataElementCount() < 4 * size_t(pixels) * (view + 1))
        throw std::invalid_argument("imageTarget: the targets hold no such view of the image's size");
    const VkFormat format = vulkanContext.getSwapChainImageFormat();
    if (format != VK_FORMAT_B8G8R8A8_UNORM && format != VK_FORMAT_R8G8B8A8_UNORM && format != VK_FORMAT_B8G8R8A8_SRGB &&
        format != VK_FORMAT_R8G8B8A8_SRGB)
        throw std::invalid_argument("imageTarget: the images need 8 bits per channel");
    auto picture = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, pixels);
    auto copy = std::make_shared<ImageCopy>(layout);
    copy->setName("ImageTargetCopy");
    copy->setInput(image, 0);
    copy->setInput(picture, 1);
    copy->addDependency(renderer);
    auto convert =
        vulkanContext.create<GeneralComputation<ImageTargetPushConstants>>("shaders/raster/image_target.comp.spv");
    convert->setName("ImageTarget");
    const uint32_t bgra = format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB;
    convert->setPushConstants({{pixels, view, bgra}});
    convert->setGroupCountX((pixels + 255) / 256);
    convert->setInput(copy, 0, 1);
    convert->setInput(targets, 1);
    return convert;
}

} // namespace klartraum::raster
