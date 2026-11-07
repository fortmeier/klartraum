// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - matchesPyTorch: the field fixture (scripts/neural/make_field_fixture.py: a small world-anchored style with fixed
 *   random weights and a 96 x 64 frame of random colour, classes, partial coverages and depth) run by FieldStyle
 *   packs PyTorch's input planes (colour, classes, depth, grid and paper look-ups, conditions) and gives its output,
 *   both within 1e-3; at full blend the output image is that output clamped within one step of 255, and at blend 0 it
 * is the frame's colour unchanged
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/field_style.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/offscreen_target.hpp"

#include "tree_test_helpers.hpp"

using namespace klartraum;
using namespace klartraum::tree_test;

namespace {

constexpr uint32_t kWidth = 96, kHeight = 64;
const std::string kFixture = "data/onnx/field_fixture";

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

TEST(FieldStyleTest, matchesPyTorch) {
    HeadlessFrontend frontend;
    VulkanContext& vc = frontend.getKlartraumEngine().getVulkanContext();
    const VkExtent2D extent{kWidth, kHeight};
    const size_t pixels = size_t(kWidth) * kHeight;
    const std::vector<uint8_t> images = readFile<uint8_t>(kFixture + "_images.bin", 4 * pixels * 4);
    const std::vector<float> frame = readFile<float>(kFixture + "_frame.bin", 28);
    const std::vector<float> expected = readFile<float>(kFixture + "_output.bin", 3 * pixels);

    std::shared_ptr<OffscreenTarget> targets[4]; // colour, classes, coverage, depth
    for (int k = 0; k < 4; ++k) {
        targets[k] = std::make_shared<OffscreenTarget>(vc, extent, 1);
        targets[k]->setClear(false);
        upload(vc, targets[k]->getImage(0), images.data() + size_t(k) * pixels * 4);
    }
    auto output = std::make_shared<OffscreenTarget>(vc, extent, 1);
    FieldStyle style(vc, kFixture + ".onnx", kFixture + ".bin", targets[0], targets[1], targets[2], targets[3], output,
                     targets[0]);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(style.getOutput());

    const glm::mat4 camera = glm::transpose(glm::make_mat4(frame.data())); // the file is row-major
    std::array<float, 7> conditions;
    std::copy(frame.begin() + 21, frame.begin() + 28, conditions.begin());
    style.setFrame(camera, glm::vec3(frame[16], frame[17], frame[18]), frame[19], frame[20], conditions);
    style.setBlend(1.0f);
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    const std::vector<float> packed = readFile<float>(kFixture + "_input.bin", 41 * pixels);
    std::vector<float> planes(packed.size());
    style.getInput()->getDataBuffer(0).memcopyTo(planes);
    float largestInput = 0.0f;
    size_t worstInput = 0;
    for (size_t i = 0; i < planes.size(); ++i)
        if (std::abs(planes[i] - packed[i]) > largestInput) {
            largestInput = std::abs(planes[i] - packed[i]);
            worstInput = i;
        }
    std::printf("largest difference of the input planes: %g in plane %zu at pixel %zu, %zu\n", largestInput,
                worstInput / pixels, worstInput % pixels % kWidth, worstInput % pixels / kWidth);
    EXPECT_LT(largestInput, 1e-3f) << "the packed input";

    auto tensor = std::dynamic_pointer_cast<TensorElement<float>>(style.getNetwork()->getOutputElement("output"));
    ASSERT_NE(tensor, nullptr);
    std::vector<float> values(tensor->getDataElementCount());
    tensor->getDataBuffer(0).memcopyTo(values);
    ASSERT_EQ(values.size(), expected.size());
    float largest = 0.0f;
    size_t worstPixel = 0;
    for (size_t i = 0; i < values.size(); ++i)
        if (std::abs(values[i] - expected[i]) > largest) {
            largest = std::abs(values[i] - expected[i]);
            worstPixel = i % pixels;
        }
    std::printf("largest difference to PyTorch: %g at pixel %zu, %zu\n", largest, worstPixel % kWidth,
                worstPixel / kWidth);
    EXPECT_LT(largest, 1e-3f);

    const std::vector<uint8_t> styled =
        readImage(vc, output->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
    int worst = 0;
    for (size_t p = 0; p < pixels; ++p)
        for (int c = 0; c < 3; ++c) {
            const int want = int(std::lround(std::clamp(expected[size_t(c) * pixels + p], 0.0f, 1.0f) * 255.0f));
            worst = std::max(worst, std::abs(int(styled[4 * p + 2 - c]) - want));
        }
    EXPECT_LE(worst, 1) << "the output image at full blend";

    style.setBlend(0.0f);
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    const std::vector<uint8_t> plain = readImage(vc, output->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
    int changed = 0;
    for (size_t p = 0; p < pixels; ++p)
        for (int c = 0; c < 3; ++c)
            changed = std::max(changed, std::abs(int(plain[4 * p + c]) - int(images[4 * p + c])));
    EXPECT_EQ(changed, 0) << "the output image at blend 0";
}
