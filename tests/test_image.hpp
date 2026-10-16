// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_TESTS_TEST_IMAGE_HPP
#define KLARTRAUM_TESTS_TEST_IMAGE_HPP

#include <cstdint>
#include <vector>

#include "klartraum/vulkan_context.hpp"

namespace klartraum::test_image {

// Reads an image back as tightly-packed BGRA bytes. The splatting backends
// leave an OffscreenTarget in TRANSFER_SRC_OPTIMAL.
inline std::vector<uint8_t> readImage(VulkanContext& vc, VkImage image, VkExtent2D extent,
                                      VkImageLayout layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL) {
    const VkDeviceSize bytes = VkDeviceSize(extent.width) * extent.height * 4;
    VkBuffer buf;
    VkDeviceMemory mem;
    vc.createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buf, mem);
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = vc.getCommandPool();
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(vc.getDevice(), &ai, &cmd);
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {extent.width, extent.height, 1};
    vkCmdCopyImageToBuffer(cmd, image, layout, buf, 1, &region);
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    vkQueueSubmit(vc.getGraphicsQueue(), 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(vc.getGraphicsQueue());
    vkFreeCommandBuffers(vc.getDevice(), vc.getCommandPool(), 1, &cmd);
    void* data;
    vkMapMemory(vc.getDevice(), mem, 0, bytes, 0, &data);
    std::vector<uint8_t> result(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + bytes);
    vkUnmapMemory(vc.getDevice(), mem);
    vkFreeMemory(vc.getDevice(), mem, nullptr);
    vkDestroyBuffer(vc.getDevice(), buf, nullptr);
    return result;
}

} // namespace klartraum::test_image

#endif // KLARTRAUM_TESTS_TEST_IMAGE_HPP
