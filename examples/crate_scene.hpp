// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_EXAMPLES_CRATE_SCENE_HPP
#define KLARTRAUM_EXAMPLES_CRATE_SCENE_HPP

// The crate toy scene of crate_gbuffer_example and crate_training_example: a crate (a box whose six faces have their
// own surface coordinates and object ids) standing on a floor, its cameras and its G-buffers.

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/gbuffer_mesh_renderer.hpp"
#include "klartraum/neural/photo_enhancer.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/tree/tree_model.hpp"

namespace crate {

constexpr float kFovDegrees = 40.0f;
constexpr float kDepthFar = 60.0f; // the depth window, beyond the floor's far corners

// The crate and the floor. The crate's faces get ids spread over the 16 bits, (face + 0.5) / 6 of the range, so that
// the texture's id axis holds six well separated slices; the floor sits in between at 1/2.
inline std::vector<klartraum::GBufferVertex> vertices() {
    std::array<uint32_t, 6> faceIds{};
    for (uint32_t face = 0; face < 6; ++face)
        faceIds[face] = uint32_t(std::lround((float(face) + 0.5f) / 6.0f * 65535.0f));
    auto mesh = klartraum::gbufferBox(glm::vec3(-0.5f, 0.0f, -0.5f), glm::vec3(0.5f, 1.0f, 0.5f),
                                      klartraum::tree::TREE_CLASS_STEM, glm::vec3(0.62f, 0.45f, 0.28f), faceIds);
    const auto floor = klartraum::gbufferFloor(40.0f, 0.0f, 4.0f, klartraum::tree::TREE_CLASS_SOIL,
                                               glm::vec3(0.5f, 0.5f, 0.48f), 32768u);
    mesh.insert(mesh.end(), floor.begin(), floor.end());
    return mesh;
}

inline void setCamera(klartraum::CameraMVP& ubo, glm::vec3 eye, glm::vec3 look) {
    ubo.model = glm::mat4(1.0f);
    ubo.view = glm::lookAt(eye, look, glm::vec3(0.0f, 1.0f, 0.0f));
    ubo.proj = glm::perspective(glm::radians(kFovDegrees), 1.0f, 0.05f, 100.0f);
    ubo.proj[1][1] *= -1.0f;
    ubo.cameraWorldPos = glm::vec4(eye, 1.0f);
}

// Reads an OffscreenTarget image back as tightly packed BGRA bytes; the render pass leaves it in TRANSFER_SRC_OPTIMAL.
inline std::vector<uint8_t> readImage(klartraum::VulkanContext& vc, VkImage image, VkExtent2D extent) {
    const VkDeviceSize bytes = VkDeviceSize(extent.width) * extent.height * 4;
    VkBuffer buffer;
    VkDeviceMemory memory;
    vc.createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buffer, memory);
    vc.submitImmediate([&](VkCommandBuffer commandBuffer) {
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {extent.width, extent.height, 1};
        vkCmdCopyImageToBuffer(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &region);
    });
    void* data;
    vkMapMemory(vc.getDevice(), memory, 0, bytes, 0, &data);
    std::vector<uint8_t> pixels(static_cast<uint8_t*>(data), static_cast<uint8_t*>(data) + bytes);
    vkUnmapMemory(vc.getDevice(), memory);
    vkDestroyBuffer(vc.getDevice(), buffer, nullptr);
    vkFreeMemory(vc.getDevice(), memory, nullptr);
    return pixels;
}

// The G-buffers the photo enhancer takes, of the view the renderer's camera shows.
inline klartraum::neural::GBufferFrame renderFrame(klartraum::VulkanContext& vc,
                                                   klartraum::GBufferMeshRenderer& renderer,
                                                   klartraum::ComputeGraph& graph, klartraum::OffscreenTarget& target,
                                                   uint32_t size) {
    using klartraum::GBufferOutput;
    klartraum::neural::GBufferFrame frame;
    frame.width = frame.height = size;
    frame.depthFar = kDepthFar;
    const std::pair<std::vector<uint8_t>*, GBufferOutput> outputs[] = {{&frame.colour, GBufferOutput::Colour},
                                                                       {&frame.classes, GBufferOutput::Classes},
                                                                       {&frame.depth16, GBufferOutput::Depth16},
                                                                       {&frame.u, GBufferOutput::CoordinatesU},
                                                                       {&frame.v, GBufferOutput::CoordinatesV}};
    for (const auto& [buffer, output] : outputs) {
        renderer.setOutput(output, 0.0f, kDepthFar);
        graph.submitAndWait(vc.getGraphicsQueue(), 0);
        *buffer = readImage(vc, target.getImage(0), VkExtent2D{size, size});
    }
    return frame;
}

} // namespace crate

#endif // KLARTRAUM_EXAMPLES_CRATE_SCENE_HPP
