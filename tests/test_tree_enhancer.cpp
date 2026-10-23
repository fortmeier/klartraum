// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - matchesPyTorch: the enhancer fixtures (scripts/neural/make_enhancer_fixture.py: each network, v1 with transposed
 *   convolutions and v2 with nearest-neighbour resizes, with fixed random weights and three random 96 x 64 images)
 *   run by TreeEnhancer give PyTorch's output within 1e-3, and write it clamped to [0, 1] into the output image
 *   within one step of 255
 **/

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <memory>
#include <vector>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/tree/tree_enhancer.hpp"

#include "tree_test_helpers.hpp"

using namespace klartraum;
using namespace klartraum::tree_test;

namespace {

constexpr uint32_t kWidth = 96, kHeight = 64;

template <typename T>
std::vector<T> readFile(const std::string& path, size_t count) {
    std::vector<T> values(count);
    std::ifstream file(path, std::ios::binary);
    file.read(reinterpret_cast<char*>(values.data()), std::streamsize(count * sizeof(T)));
    if (!file) {
        throw std::runtime_error("cannot read " + path);
    }
    return values;
}

// Fills `image` with BGRA bytes and leaves it in TRANSFER_SRC_OPTIMAL, as a splatting leaves its target.
void upload(VulkanContext& vc, VkImage image, const uint8_t* bgra) {
    const VkDeviceSize bytes = VkDeviceSize(kWidth) * kHeight * 4;
    VkBuffer buffer;
    VkDeviceMemory memory;
    vc.createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buffer, memory);
    void* data;
    vkMapMemory(vc.getDevice(), memory, 0, bytes, 0, &data);
    std::memcpy(data, bgra, bytes);
    vkUnmapMemory(vc.getDevice(), memory);
    vc.submitImmediate([&](VkCommandBuffer commandBuffer) {
        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &barrier);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {kWidth, kHeight, 1};
        vkCmdCopyBufferToImage(commandBuffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &barrier);
    });
    vkDestroyBuffer(vc.getDevice(), buffer, nullptr);
    vkFreeMemory(vc.getDevice(), memory, nullptr);
}

} // namespace

TEST(TreeEnhancerTest, matchesPyTorch) {
    HeadlessFrontend frontend;
    VulkanContext& vc = frontend.getKlartraumEngine().getVulkanContext();
    for (const std::string stem : {"data/onnx/tree_enhancer_fixture", "data/onnx/tree_enhancer_v2_fixture"}) {
        const size_t pixels = size_t(kWidth) * kHeight;
        const std::vector<uint8_t> images = readFile<uint8_t>(stem + "_images.bin", 3 * pixels * 4);
        const std::vector<float> expected = readFile<float>(stem + "_output.bin", 3 * pixels);

        std::shared_ptr<OffscreenTarget> inputs[3];
        for (int k = 0; k < 3; ++k) {
            inputs[k] = std::make_shared<OffscreenTarget>(vc, VkExtent2D{kWidth, kHeight}, 1);
            inputs[k]->setClear(false);
            upload(vc, inputs[k]->getImage(0), images.data() + size_t(k) * pixels * 4);
        }
        auto output = std::make_shared<OffscreenTarget>(vc, VkExtent2D{kWidth, kHeight}, 1);
        TreeEnhancer enhancer(vc, stem + ".onnx", inputs[0], inputs[1], inputs[2], output, inputs[0]);
        ComputeGraph graph(vc, 1);
        graph.compileFrom(enhancer.getOutput());
        graph.submitAndWait(vc.getGraphicsQueue(), 0);

        auto tensor =
            std::dynamic_pointer_cast<TensorElement<float>>(enhancer.getNetwork()->getOutputElement("output"));
        ASSERT_NE(tensor, nullptr);
        std::vector<float> values(tensor->getDataElementCount());
        tensor->getDataBuffer(0).memcopyTo(values);
        ASSERT_EQ(values.size(), expected.size());
        float largest = 0.0f;
        for (size_t i = 0; i < values.size(); ++i) {
            largest = std::max(largest, std::abs(values[i] - expected[i]));
        }
        std::printf("%s: largest difference to PyTorch: %g\n", stem.c_str(), largest);
        EXPECT_LT(largest, 1e-3f) << stem;

        const std::vector<uint8_t> image =
            readImage(vc, output->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VkExtent2D{kWidth, kHeight});
        int worst = 0;
        for (size_t p = 0; p < pixels; ++p) {
            for (int c = 0; c < 3; ++c) {
                const float want = std::clamp(expected[size_t(c) * pixels + p], 0.0f, 1.0f) * 255.0f;
                worst = std::max(worst, int(std::lround(std::abs(float(image[4 * p + 2 - c]) - want))));
            }
        }
        std::printf("%s: largest difference of the output image: %d\n", stem.c_str(), worst);
        EXPECT_LE(worst, 1) << stem;
    }
}
