// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - drawsFixedAndIndirectDraws: a triangle drawn with fixed counts and one drawn with arguments from a buffer (each
 *   with its own push constants, vertices from a storage buffer) cover their halves of the target in their colours
 *   and leave the rest clear
 **/

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include <glm/glm.hpp>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/renderpass.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/storage_draw_component.hpp"

using namespace klartraum;

namespace {

constexpr VkExtent2D kExtent{32, 32};

// The image's pixels as BGRA bytes (the headless swapchain format).
std::vector<uint8_t> readImage(VulkanContext& vc, VkImage image, VkExtent2D extent) {
    const VkDeviceSize bytes = VkDeviceSize{extent.width} * extent.height * 4;
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

struct PushConstants {
    glm::vec4 color;
    uint32_t first;
    uint32_t padding[3];
};

} // namespace

TEST(StorageDrawComponent, drawsFixedAndIndirectDraws) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    // Two triangles in clip space: the left half's lower-left corner triangle and the right half's.
    const std::vector<glm::vec4> positions{{-1.0f, -1.0f, 0.5f, 1.0f}, {0.0f, -1.0f, 0.5f, 1.0f},
                                           {-1.0f, 1.0f, 0.5f, 1.0f},  {0.0f, -1.0f, 0.5f, 1.0f},
                                           {1.0f, -1.0f, 0.5f, 1.0f},  {0.0f, 1.0f, 0.5f, 1.0f}};
    auto vertices = std::make_shared<BufferElementSinglePath<VulkanBuffer<glm::vec4>>>(vc, uint32_t(positions.size()));
    vertices->getBuffer().memcopyFrom(positions);
    // The second draw's arguments (VkDrawIndirectCommand: 3 vertices, 1 instance).
    auto arguments = std::make_shared<BufferElementSinglePath<VulkanBuffer<uint32_t>>>(vc, 4u);
    arguments->getBuffer().memcopyFrom(std::vector<uint32_t>{3, 1, 0, 0});

    StorageDrawConfig config;
    config.vertexShader = "shaders/test/storage_draw.vert.spv";
    config.fragmentShader = "shaders/test/storage_draw.frag.spv";
    config.extent = kExtent;
    config.depthTest = config.depthWrite = false;
    config.pushConstantSize = sizeof(PushConstants);
    auto draw = std::make_shared<StorageDrawComponent>(config, std::vector<BufferRef>{{vertices, -1}});
    StorageDrawComponent::Draw fixed;
    fixed.vertexCount = 3;
    fixed.pushConstants = StorageDrawComponent::bytes(PushConstants{{1.0f, 0.0f, 0.0f, 1.0f}, 0, {}});
    draw->addDraw(fixed);
    StorageDrawComponent::Draw indirect;
    indirect.indirect = {arguments, -1};
    indirect.pushConstants = StorageDrawComponent::bytes(PushConstants{{0.0f, 1.0f, 0.0f, 1.0f}, 3, {}});
    draw->addDraw(indirect);

    auto target = std::make_shared<OffscreenTarget>(vc, kExtent, 1);
    auto camera = std::make_shared<CameraUboType>();
    auto pass = std::make_shared<RenderPass>(vc.getSwapChainImageFormat(), kExtent);
    pass->setInput(target, 0);
    pass->setInput(camera, 1);
    pass->addDrawComponent(draw);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(pass);
    graph.submitAndWait(vc.getGraphicsQueue(), 0);

    const auto pixels = readImage(vc, target->getImage(0), kExtent);
    const auto bgra = [&](uint32_t x, uint32_t y) {
        const size_t k = (size_t(y) * kExtent.width + x) * 4;
        return glm::ivec3(pixels[k + 2], pixels[k + 1], pixels[k]);
    };
    // Near the lower-left corner of each half (y grows downwards in Vulkan's clip space, so clip y = -1 is the
    // top row), and the half's opposite corner, which neither triangle covers.
    EXPECT_EQ(bgra(2, 2), glm::ivec3(255, 0, 0));
    EXPECT_EQ(bgra(18, 2), glm::ivec3(0, 255, 0));
    EXPECT_EQ(bgra(14, 29), glm::ivec3(0, 0, 0));
    EXPECT_EQ(bgra(30, 29), glm::ivec3(0, 0, 0));
}
