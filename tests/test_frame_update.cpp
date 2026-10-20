// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - receivesAcquiredPathOncePerFrame: each headless frame invokes a callback once
 *   with the acquired path and its associated submission fence
 * - uploadsReachGpuInSameFrameAndStayPathLocal: callback uploads are consumed by
 *   this frame's compute dispatch without overwriting another path's results
 * - clearingCallbackStopsUploads: nullptr disables the callback while subsequent
 *   frames continue to consume the last uploaded values
 * - headlessReleasesCapturedBuffersBeforeDeviceShutdown: destroying a headless frontend
 *   releases callback-owned GPU allocations before destroying the Vulkan device
 * - windowedReleasesCapturedBuffersBeforeDeviceShutdown: destroying a GLFW frontend
 *   releases callback-owned GPU allocations before destroying the Vulkan device
 * - standaloneEngineReleasesCapturedBuffersBeforeContext: callback-owned allocations
 *   used by an active frame are released before the standalone engine's Vulkan context
 * - uninitializedEngineCanReleaseCallbackCaptures: callback captures can be released
 *   without initializing a Vulkan device or frame synchronization objects
 **/
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <vector>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/glfw_frontend.hpp"

using namespace klartraum;

class FrameUpdateTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        auto& engine = frontend->getKlartraumEngine();
        auto& vc = engine.getVulkanContext();
        paths = vc.getNumberOfSwapChainImages();
        ASSERT_GE(paths, 2u);

        input = std::make_shared<BufferElement<VulkanBuffer<float>>>(vc, 4);
        output = std::make_shared<BufferElement<VulkanBuffer<float>>>(vc, 4);
        factor = std::make_shared<BufferElementSinglePath<VulkanBuffer<float>>>(vc, 4);
        factor->getBuffer().memcopyFrom(std::vector<float>(4, 2.0f));

        // A real GPU dispatch copies scaled input into per-path output. The render pass
        // makes this graph participate in the engine's normal acquisition/submission loop.
        auto multiply =
            std::make_shared<GeneralComputation<>>(vc, "shaders/operator_multiply_scalar_element_wise.comp.spv");
        multiply->setInput(input, 0);
        multiply->setInput(factor, 1);
        multiply->setInput(output, 2);
        multiply->setGroupCountX(4);
        auto renderPass = engine.createRenderPass();
        renderPass->addComputeDependency(multiply);
        engine.add(renderPass);

        for (uint32_t path = 0; path < paths; ++path) {
            input->getBuffer(path).memcopyFrom(std::vector<float>(4, -1.0f));
            output->getBuffer(path).memcopyFrom(std::vector<float>(4, -1.0f));
        }
    }

    void TearDown() override {
        if (!frontend)
            return;
        auto& engine = frontend->getKlartraumEngine();
        // Release callback captures and GPU allocations while the frontend's device is valid.
        EXPECT_EQ(vkDeviceWaitIdle(engine.getVulkanContext().getDevice()), VK_SUCCESS);
        engine.setFrameUpdate(nullptr);
        engine.clearComputeGraphs();
        input.reset();
        output.reset();
        factor.reset();
        frontend.reset();
    }

    std::vector<float> readOutput(uint32_t path) {
        std::vector<float> values(4);
        output->getBuffer(path).memcopyTo(values);
        return values;
    }

    std::unique_ptr<HeadlessFrontend> frontend;
    std::shared_ptr<BufferElement<VulkanBuffer<float>>> input;
    std::shared_ptr<BufferElement<VulkanBuffer<float>>> output;
    std::shared_ptr<BufferElementSinglePath<VulkanBuffer<float>>> factor;
    uint32_t paths = 0;
};

TEST_F(FrameUpdateTest, receivesAcquiredPathOncePerFrame) {
    auto& engine = frontend->getKlartraumEngine();
    auto& vc = engine.getVulkanContext();
    std::vector<uint32_t> acquiredPaths;

    // The callback receives the path acquired for this frame's submission
    engine.setFrameUpdate([&](uint32_t path) {
        ASSERT_LT(path, paths);
        EXPECT_EQ(vc.imageFences[path], vc.inFlightFences[vc.currentFrame]);
        acquiredPaths.push_back(path);
    });

    // Run enough frames to acquire each path multiple times
    for (uint32_t frame = 0; frame < 3 * paths; ++frame) {
        const uint32_t expectedPath = vc.currentFrame % paths;
        engine.step();
        ASSERT_EQ(acquiredPaths.size(), frame + 1u);
        EXPECT_EQ(acquiredPaths.back(), expectedPath);
    }

    // Verify that each path was acquired at least once during the test
    for (uint32_t path = 0; path < paths; ++path) {
        EXPECT_NE(std::find(acquiredPaths.begin(), acquiredPaths.end(), path), acquiredPaths.end());
    }
}

TEST_F(FrameUpdateTest, uploadsReachGpuInSameFrameAndStayPathLocal) {
    auto& engine = frontend->getKlartraumEngine();
    auto& vc = engine.getVulkanContext();
    std::vector<std::vector<float>> expected(paths, std::vector<float>(4, -1.0f));
    uint32_t updates = 0;

    // The callback receives the path acquired for this frame's submission and uploads
    // a unique value to that path's input buffer. The compute dispatch multiplies it by
    // 2 and writes it to the output buffer, which is read back after submission.
    engine.setFrameUpdate([&](uint32_t path) {
        ASSERT_LT(path, paths);

        ++updates;
        const float value = static_cast<float>(updates);
        const std::vector<float> values{value, value + 10.0f, -value, value * 4.0f};
        input->getBuffer(path).memcopyFrom(values);
        for (size_t i = 0; i < values.size(); ++i)
            expected[path][i] = values[i] * 2.0f;
    });

    for (uint32_t frame = 0; frame < 3 * paths; ++frame) {
        const VkFence submitted = vc.inFlightFences[vc.currentFrame];
        engine.step();
        ASSERT_EQ(updates, frame + 1u);
        // Wait only for this frame's submission before reading coherent GPU-written output.
        ASSERT_EQ(vkWaitForFences(vc.getDevice(), 1, &submitted, VK_TRUE, UINT64_MAX), VK_SUCCESS);
        for (uint32_t path = 0; path < paths; ++path) {
            EXPECT_EQ(readOutput(path), expected[path]) << "frame " << frame << ", path " << path;
        }
    }
}

TEST_F(FrameUpdateTest, clearingCallbackStopsUploads) {
    auto& engine = frontend->getKlartraumEngine();
    auto& vc = engine.getVulkanContext();
    std::vector<std::vector<float>> expected(paths);
    uint32_t updates = 0;
    engine.setFrameUpdate([&](uint32_t path) {
        ASSERT_LT(path, paths);

        const float value = static_cast<float>(++updates);
        input->getBuffer(path).memcopyFrom(std::vector<float>(4, value));
        expected[path] = std::vector<float>(4, value * 2.0f);
    });
    for (uint32_t frame = 0; frame < paths; ++frame)
        engine.step();
    ASSERT_EQ(updates, paths);

    engine.setFrameUpdate(nullptr);
    for (uint32_t frame = 0; frame < 2 * paths; ++frame)
        engine.step();
    ASSERT_EQ(vkDeviceWaitIdle(vc.getDevice()), VK_SUCCESS);
    EXPECT_EQ(updates, paths);
    for (uint32_t path = 0; path < paths; ++path)
        EXPECT_EQ(readOutput(path), expected[path]);
}

TEST(FrameUpdateLifetimeDeathTest, headlessReleasesCapturedBuffersBeforeDeviceShutdown) {
    EXPECT_EXIT(
        {
            std::weak_ptr<VulkanBuffer<float>> captured;
            {
                HeadlessFrontend frontend;
                auto& engine = frontend.getKlartraumEngine();
                auto buffer = std::make_shared<VulkanBuffer<float>>(engine.getVulkanContext(), 4);
                captured = buffer;
                engine.add(engine.createRenderPass());
                engine.setFrameUpdate([buffer](uint32_t) { buffer->memcopyFrom(std::vector<float>(4, 1.0f)); });
                buffer.reset();
                engine.step();
            }
            std::exit(captured.expired() ? 0 : 1);
        },
        ::testing::ExitedWithCode(0), "");
}

TEST(FrameUpdateLifetimeDeathTest, windowedReleasesCapturedBuffersBeforeDeviceShutdown) {
    EXPECT_EXIT(
        {
            if (!glfwInit()) {
                std::exit(1);
            }
            glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
            std::weak_ptr<VulkanBuffer<float>> captured;
            {
                GlfwFrontend frontend;
                auto& engine = frontend.getKlartraumEngine();
                auto buffer = std::make_shared<VulkanBuffer<float>>(engine.getVulkanContext(), 4);
                captured = buffer;
                engine.add(engine.createRenderPass());
                engine.setFrameUpdate([buffer](uint32_t) { buffer->memcopyFrom(std::vector<float>(4, 1.0f)); });
                buffer.reset();
                engine.step();
            }
            std::exit(captured.expired() ? 0 : 1);
        },
        ::testing::ExitedWithCode(0), "");
}

TEST(FrameUpdateLifetimeDeathTest, standaloneEngineReleasesCapturedBuffersBeforeContext) {
    EXPECT_EXIT(
        {
            using Source = BufferElementSinglePath<VulkanBuffer<float>>;
            std::weak_ptr<Source> captured;
            {
                KlartraumEngine engine;
                auto& vc = engine.getVulkanContext();
                vc.initialize();
                auto buffer = std::make_shared<Source>(vc, 4);
                captured = buffer;
                engine.setGraphBuilder([buffer](KlartraumEngine&) {});
                auto factor = std::make_shared<Source>(vc, 4);
                factor->getBuffer().memcopyFrom(std::vector<float>(4, 2.0f));
                auto output = std::make_shared<BufferElement<VulkanBuffer<float>>>(vc, 4);
                auto multiply = std::make_shared<GeneralComputation<>>(
                    vc, "shaders/operator_multiply_scalar_element_wise.comp.spv");
                multiply->setInput(buffer, 0);
                multiply->setInput(factor, 1);
                multiply->setInput(output, 2);
                multiply->setGroupCountX(4);
                auto renderPass = engine.createRenderPass();
                renderPass->addComputeDependency(multiply);
                engine.add(renderPass);
                engine.setFrameUpdate(
                    [buffer](uint32_t) { buffer->getBuffer().memcopyFrom(std::vector<float>(4, 2.0f)); });
                engine.step();
            }
            std::exit(captured.expired() ? 0 : 1);
        },
        ::testing::ExitedWithCode(0), "");
}

TEST(FrameUpdateLifetimeDeathTest, uninitializedEngineCanReleaseCallbackCaptures) {
    EXPECT_EXIT(
        {
            std::weak_ptr<int> captured;
            {
                KlartraumEngine engine;
                auto value = std::make_shared<int>(7);
                captured = value;
                engine.setFrameUpdate([value](uint32_t) {});
            }
            std::exit(captured.expired() ? 0 : 1);
        },
        ::testing::ExitedWithCode(0), "");
}
