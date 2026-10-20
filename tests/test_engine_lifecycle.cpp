// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - uninitializedEngineCanReleaseCallbackCaptures: callback captures can be released
 *   without initializing a Vulkan device or frame synchronization objects
 * - standaloneEngineReleasesCapturedBuffersBeforeContext: callback-owned allocations
 *   used by an active frame are released before the standalone engine's Vulkan context
 * - headlessReleasesCapturedBuffersBeforeDeviceShutdown: destroying a headless frontend
 *   releases callback-owned GPU allocations before destroying the Vulkan device
 * - windowedReleasesCapturedBuffersBeforeDeviceShutdown: destroying a GLFW frontend
 *   releases callback-owned GPU allocations before destroying the Vulkan device
 **/
#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <vector>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/glfw_frontend.hpp"
#include "klartraum/headless_frontend.hpp"

using namespace klartraum;

TEST(EngineLifecycleDeathTest, uninitializedEngineCanReleaseCallbackCaptures) {
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

TEST(EngineLifecycleDeathTest, standaloneEngineReleasesCapturedBuffersBeforeContext) {
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

TEST(EngineLifecycleDeathTest, headlessReleasesCapturedBuffersBeforeDeviceShutdown) {
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

TEST(EngineLifecycleDeathTest, windowedReleasesCapturedBuffersBeforeDeviceShutdown) {
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
