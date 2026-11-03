// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/image_fill.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace klartraum {

namespace {

void imageBarrier(VkCommandBuffer commandBuffer, VkImage image, VkImageLayout from, VkImageLayout to,
                  VkAccessFlags srcAccess, VkAccessFlags dstAccess) {
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = from;
    barrier.newLayout = to;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &barrier);
}

} // namespace

ImageFill::~ImageFill() {
    if (!vulkanContext)
        return;
    VkDevice device = vulkanContext->getDevice();
    for (size_t i = 0; i < buffers.size(); ++i) {
        vkUnmapMemory(device, bufferMemories[i]);
        vkDestroyBuffer(device, buffers[i], nullptr);
        vkFreeMemory(device, bufferMemories[i], nullptr);
        vkDestroyImage(device, texels[i], nullptr);
        vkFreeMemory(device, texelMemories[i], nullptr);
    }
}

void ImageFill::setColour(glm::vec3 value) {
    colour = value;
    writeColour();
}

void ImageFill::writeColour() {
    const auto byte = [](float v) { return uint8_t(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f)); };
    const uint8_t r = byte(colour.r), g = byte(colour.g), b = byte(colour.b);
    const uint8_t bytes[4] = {bgra ? b : r, g, bgra ? r : b, 255};
    for (void* memory : mapped)
        std::memcpy(memory, bytes, 4);
}

void ImageFill::_setup(VulkanContext& context, uint32_t numberPaths, VkQueue setupQueue) {
    ImageViewForward::_setup(context, numberPaths, setupQueue);
    if (vulkanContext)
        return;
    vulkanContext = &context;
    const VkFormat format = context.getSwapChainImageFormat();
    if (format != VK_FORMAT_B8G8R8A8_UNORM && format != VK_FORMAT_R8G8B8A8_UNORM && format != VK_FORMAT_B8G8R8A8_SRGB &&
        format != VK_FORMAT_R8G8B8A8_SRGB)
        throw std::runtime_error("ImageFill: the images need 8 bits per channel");
    bgra = format == VK_FORMAT_B8G8R8A8_UNORM || format == VK_FORMAT_B8G8R8A8_SRGB;
    VkDevice device = context.getDevice();
    buffers.resize(numberPaths);
    bufferMemories.resize(numberPaths);
    mapped.resize(numberPaths);
    texels.resize(numberPaths);
    texelMemories.resize(numberPaths);
    for (uint32_t i = 0; i < numberPaths; ++i) {
        context.createBuffer(4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buffers[i],
                             bufferMemories[i]);
        vkMapMemory(device, bufferMemories[i], 0, 4, 0, &mapped[i]);
        VkImageCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = format;
        info.extent = {1, 1, 1};
        info.mipLevels = info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(device, &info, nullptr, &texels[i]) != VK_SUCCESS)
            throw std::runtime_error("ImageFill: cannot create the texel image");
        VkMemoryRequirements requirements;
        vkGetImageMemoryRequirements(device, texels[i], &requirements);
        VkMemoryAllocateInfo allocation{};
        allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex =
            context.findMemoryType(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (vkAllocateMemory(device, &allocation, nullptr, &texelMemories[i]) != VK_SUCCESS)
            throw std::runtime_error("ImageFill: cannot allocate the texel image");
        vkBindImageMemory(device, texels[i], texelMemories[i], 0);
    }
    writeColour();
}

void ImageFill::_record(VkCommandBuffer commandBuffer, uint32_t pathId) {
    ComputeGraphElement::_record(commandBuffer, pathId);
    VkImage texel = texels.at(pathId), image = getImage(pathId);
    const VkExtent2D extent = getImageExtent(pathId);
    imageBarrier(commandBuffer, texel, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0,
                 VK_ACCESS_TRANSFER_WRITE_BIT);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {1, 1, 1};
    vkCmdCopyBufferToImage(commandBuffer, buffers.at(pathId), texel, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    imageBarrier(commandBuffer, texel, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    // Every pixel is overwritten: the previous contents need not be kept.
    imageBarrier(commandBuffer, image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    VkImageBlit blit{};
    blit.srcSubresource = blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    blit.srcOffsets[1] = {1, 1, 1};
    blit.dstOffsets[1] = {int32_t(extent.width), int32_t(extent.height), 1};
    vkCmdBlitImage(commandBuffer, texel, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
    imageBarrier(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                 VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
}

} // namespace klartraum
