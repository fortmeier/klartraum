// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_TESTS_TREE_TEST_HELPERS_HPP
#define KLARTRAUM_TESTS_TREE_TEST_HELPERS_HPP

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "klartraum/draw_component.hpp"
#include "klartraum/tree/tree_model.hpp"
#include "klartraum/vulkan_context.hpp"

namespace klartraum {
namespace tree_test {

inline TreeInstance makeInstance(uint32_t type, uint32_t seed, glm::vec3 pos = glm::vec3(0.0f), float scale = 1.0f) {
    TreeInstance inst{};
    inst.posScale = glm::vec4(pos, scale);
    inst.rot = treeUpRotation(glm::vec3(0.0f, 1.0f, 0.0f));
    inst.ids = glm::uvec4(type, seed, 0u, 0u);
    return inst;
}

// A camera at `eye` looking at `target` with +y up and a vertical field of view of `fovDegrees`.
inline void setCamera(CameraMVP& ubo, glm::vec3 eye, glm::vec3 target, float fovDegrees, float aspect,
                      float nearPlane = 0.01f, float farPlane = 100.0f) {
    ubo.model = glm::mat4(1.0f);
    ubo.view = glm::lookAt(eye, target, glm::vec3(0.0f, 1.0f, 0.0f));
    ubo.proj = glm::perspective(glm::radians(fovDegrees), aspect, nearPlane, farPlane);
    ubo.proj[1][1] *= -1.0f;
    ubo.cameraWorldPos = glm::vec4(eye, 1.0f);
}

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

// Writes BGRA pixels to build/TestingOutput/tree/<name>.ppm.
inline void writePpm(const std::string& name, const std::vector<uint8_t>& bgra, VkExtent2D extent) {
    std::filesystem::create_directories("build/TestingOutput/tree");
    std::ofstream file("build/TestingOutput/tree/" + name + ".ppm", std::ios::binary);
    file << "P6\n" << extent.width << " " << extent.height << "\n255\n";
    for (size_t i = 0; i < bgra.size(); i += 4) {
        const char rgb[3] = {char(bgra[i + 2]), char(bgra[i + 1]), char(bgra[i])};
        file.write(rgb, 3);
    }
}

// Whether a pixel differs from the clear colour (black).
inline bool covered(const std::vector<uint8_t>& bgra, VkExtent2D extent, uint32_t x, uint32_t y) {
    const size_t i = (size_t(y) * extent.width + x) * 4;
    return bgra[i] != 0 || bgra[i + 1] != 0 || bgra[i + 2] != 0;
}

inline size_t coveredCount(const std::vector<uint8_t>& bgra, VkExtent2D extent) {
    size_t count = 0;
    for (uint32_t y = 0; y < extent.height; ++y) {
        for (uint32_t x = 0; x < extent.width; ++x) {
            count += covered(bgra, extent, x, y) ? 1 : 0;
        }
    }
    return count;
}

} // namespace tree_test
} // namespace klartraum

#endif // KLARTRAUM_TESTS_TREE_TEST_HELPERS_HPP
