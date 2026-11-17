// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - matchesPyTorch: the watercolor fixtures (scripts/neural/make_watercolor_fixture.py: a U-Net on the rendering, one
 *   on the rendering with its classes and depth, and the G-buffers modulating the decoder; fixed random weights; a
 *   192 x 128 frame of random colour, classes and depth) run by StyleTransfer at half the frame's size, and the
 *   G-buffer U-Net at the frame's full size, give PyTorch's output within 1e-3; at full blend the output image is that
 * output resized to the frame within one step of 255, and at blend 0 it is the frame's colour unchanged
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/style_transfer.hpp"

#include "tree_test_helpers.hpp"

using namespace klartraum;
using namespace klartraum::tree_test;

namespace {

constexpr uint32_t kWidth = 192, kHeight = 128;

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

// Fills `image` with BGRA bytes and leaves it in TRANSFER_SRC_OPTIMAL, as a renderer leaves its target.
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

TEST(StyleTransferTest, matchesPyTorch) {
    HeadlessFrontend frontend;
    VulkanContext& vc = frontend.getKlartraumEngine().getVulkanContext();
    const VkExtent2D extent{kWidth, kHeight};
    const size_t pixels = size_t(kWidth) * kHeight;
    for (const std::string kind : {"rgb", "gbuffers", "spade", "full"}) {
        SCOPED_TRACE(kind);
        const std::string stem = "data/onnx/watercolor_" + kind + "_fixture";
        const std::vector<uint8_t> images = readFile<uint8_t>(stem + "_images.bin", 3 * pixels * 4);
        const size_t networkPixels = kind == "full" ? pixels : pixels / 4;
        const std::vector<float> expected = readFile<float>(stem + "_output.bin", 3 * networkPixels);
        const std::vector<float> upsampled = readFile<float>(stem + "_upsampled.bin", 3 * pixels);

        std::shared_ptr<OffscreenTarget> frame[3];
        for (int k = 0; k < 3; ++k) {
            frame[k] = std::make_shared<OffscreenTarget>(vc, extent, 1);
            frame[k]->setClear(false);
            upload(vc, frame[k]->getImage(0), images.data() + size_t(k) * pixels * 4);
        }
        auto output = std::make_shared<OffscreenTarget>(vc, extent, 1);
        const bool gbuffers = kind != "rgb";
        StyleTransfer transfer(vc, stem + ".onnx", frame[0], output, frame[0], gbuffers ? frame[1] : nullptr,
                               gbuffers ? frame[2] : nullptr);
        ComputeGraph graph(vc, 1);
        graph.compileFrom(transfer.getOutput());

        transfer.setBlend(1.0f);
        graph.submitAndWait(vc.getGraphicsQueue(), 0);
        auto tensor =
            std::dynamic_pointer_cast<TensorElement<float>>(transfer.getNetwork()->getOutputElement("output"));
        ASSERT_NE(tensor, nullptr);
        std::vector<float> values(tensor->getDataElementCount());
        tensor->getDataBuffer(0).memcopyTo(values);
        ASSERT_EQ(values.size(), expected.size());
        float largest = 0.0f;
        for (size_t i = 0; i < values.size(); ++i)
            largest = std::max(largest, std::abs(values[i] - expected[i]));
        std::printf("%s: largest difference to PyTorch: %g\n", kind.c_str(), largest);
        EXPECT_LT(largest, 1e-3f);

        const std::vector<uint8_t> styled =
            readImage(vc, output->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
        int worst = 0;
        for (size_t p = 0; p < pixels; ++p)
            for (int c = 0; c < 3; ++c) {
                const int want = int(std::lround(upsampled[size_t(c) * pixels + p] * 255.0f));
                worst = std::max(worst, std::abs(int(styled[4 * p + 2 - c]) - want));
            }
        EXPECT_LE(worst, 1) << "the output image at full blend";

        transfer.setBlend(0.0f);
        graph.submitAndWait(vc.getGraphicsQueue(), 0);
        const std::vector<uint8_t> plain =
            readImage(vc, output->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
        int changed = 0;
        for (size_t p = 0; p < pixels; ++p)
            for (int c = 0; c < 3; ++c)
                changed = std::max(changed, std::abs(int(plain[4 * p + c]) - int(images[4 * p + c])));
        EXPECT_EQ(changed, 0) << "the output image at blend 0";
    }
}
