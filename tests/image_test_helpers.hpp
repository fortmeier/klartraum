// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// Reading rendered images back and writing them for a look, for the tests of renderers.

#ifndef KLARTRAUM_TESTS_IMAGE_TEST_HELPERS_HPP
#define KLARTRAUM_TESTS_IMAGE_TEST_HELPERS_HPP

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "klartraum/vulkan_context.hpp"

namespace klartraum::image_test {

// The image's pixels as BGRA bytes (the headless swapchain format).
inline std::vector<uint8_t> readImage(VulkanContext& vc, VkImage image, VkImageLayout layout, VkExtent2D extent) {
    const VkDeviceSize bytes = VkDeviceSize{extent.width} * extent.height * 4;
    VkBuffer buffer;
    VkDeviceMemory memory;
    vc.createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buffer, memory);
    vc.submitImmediate([&](VkCommandBuffer commandBuffer) {
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {extent.width, extent.height, 1};
        vkCmdCopyImageToBuffer(commandBuffer, image, layout, buffer, 1, &region);
    });
    void* data;
    vkMapMemory(vc.getDevice(), memory, 0, bytes, 0, &data);
    std::vector<uint8_t> pixels(static_cast<uint8_t*>(data), static_cast<uint8_t*>(data) + bytes);
    vkUnmapMemory(vc.getDevice(), memory);
    vkDestroyBuffer(vc.getDevice(), buffer, nullptr);
    vkFreeMemory(vc.getDevice(), memory, nullptr);
    return pixels;
}

// Writes BGRA pixels to build/TestingOutput/<directory>/<name>.ppm.
inline void writePpm(const std::string& directory, const std::string& name, const std::vector<uint8_t>& bgra,
                     VkExtent2D extent) {
    std::filesystem::create_directories("build/TestingOutput/" + directory);
    std::ofstream file("build/TestingOutput/" + directory + "/" + name + ".ppm", std::ios::binary);
    file << "P6\n" << extent.width << " " << extent.height << "\n255\n";
    for (size_t i = 0; i < bgra.size(); i += 4) {
        const char rgb[3] = {char(bgra[i + 2]), char(bgra[i + 1]), char(bgra[i])};
        file.write(rgb, 3);
    }
}

} // namespace klartraum::image_test

#endif // KLARTRAUM_TESTS_IMAGE_TEST_HELPERS_HPP
