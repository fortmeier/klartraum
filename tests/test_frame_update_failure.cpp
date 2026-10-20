// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - headlessCanShutDownAfterCallbackThrows: a failed update preserves the exception,
 *   completes the frame fence and allows immediate headless teardown
 * - windowedCanShutDownAfterCallbackThrows: a failed update returns its acquired
 *   swapchain image and allows immediate windowed teardown
 * - headlessCanRenderAfterRepeatedCallbackFailures: failed updates skip GPU dispatches,
 *   and later updates still reach the GPU after several cancelled frames
 * - windowedCanRenderAfterRepeatedCallbackFailures: non-standard callback exceptions
 *   propagate while repeated cancellation preserves swapchain and semaphore reuse
 **/
#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/glfw_frontend.hpp"
#include "klartraum/headless_frontend.hpp"

using namespace klartraum;

namespace {

int exerciseCallbackFailure(KlartraumEngine& engine, bool resume, bool nonStandardException) {
    auto& vc = engine.getVulkanContext();
    const uint32_t paths = vc.getNumberOfSwapChainImages();
    using Buffer = BufferElement<VulkanBuffer<float>>;
    auto input = std::make_shared<Buffer>(vc, 1);
    auto factor = std::make_shared<Buffer>(vc, 1);
    auto output = std::make_shared<Buffer>(vc, 1);
    auto multiply =
        std::make_shared<GeneralComputation<>>(vc, "shaders/operator_multiply_scalar_element_wise.comp.spv");
    multiply->setInput(input, 0);
    multiply->setInput(factor, 1);
    multiply->setInput(output, 2);
    auto renderPass = engine.createRenderPass();
    renderPass->addComputeDependency(multiply);
    engine.add(renderPass);
    for (uint32_t path = 0; path < paths; ++path) {
        input->getBuffer(path).memcopyFrom(std::vector<float>{1.0f});
        factor->getBuffer(path).memcopyFrom(std::vector<float>{2.0f});
        output->getBuffer(path).memcopyFrom(std::vector<float>{-1.0f});
    }

    engine.setFrameUpdate([&](uint32_t path) {
        input->getBuffer(path).memcopyFrom(std::vector<float>{9.0f});
        if (nonStandardException) {
            throw 42;
        }
        throw std::runtime_error("frame upload failed");
    });
    const uint32_t failures = resume ? 2 * paths + 1 : 1;
    for (uint32_t attempt = 0; attempt < failures; ++attempt) {
        const auto fence = vc.inFlightFences[vc.currentFrame];
        bool caught = false;
        try {
            engine.step();
        } catch (const std::runtime_error& error) {
            caught = !nonStandardException && std::string(error.what()) == "frame upload failed";
        } catch (int value) {
            caught = nonStandardException && value == 42;
        }
        // A failed fence check exits the subprocess before teardown can wait on it.
        if (!caught || vkWaitForFences(vc.getDevice(), 1, &fence, VK_TRUE, 1'000'000'000) != VK_SUCCESS) {
            std::_Exit(1);
        }
        for (uint32_t path = 0; path < paths; ++path) {
            std::vector<float> value(1);
            output->getBuffer(path).memcopyTo(value);
            if (value[0] != -1.0f) {
                std::_Exit(2);
            }
        }
    }

    engine.setFrameUpdate(nullptr);
    if (resume) {
        for (uint32_t frame = 0; frame < 2 * paths; ++frame) {
            const auto fence = vc.inFlightFences[vc.currentFrame];
            engine.step();
            if (vkWaitForFences(vc.getDevice(), 1, &fence, VK_TRUE, 1'000'000'000) != VK_SUCCESS) {
                std::_Exit(3);
            }
        }
        bool dispatched = false;
        for (uint32_t path = 0; path < paths; ++path) {
            std::vector<float> value(1);
            output->getBuffer(path).memcopyTo(value);
            dispatched = dispatched || value[0] == 18.0f;
        }
        if (!dispatched) {
            std::_Exit(4);
        }
    }
    return 0;
}

int headlessCallbackFailure(bool resume) {
    HeadlessFrontend frontend;
    return exerciseCallbackFailure(frontend.getKlartraumEngine(), resume, false);
}

int windowedCallbackFailure(bool resume) {
    if (!glfwInit()) {
        return 1;
    }
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GlfwFrontend frontend;
    return exerciseCallbackFailure(frontend.getKlartraumEngine(), resume, true);
}

} // namespace

TEST(FrameUpdateFailureDeathTest, headlessCanShutDownAfterCallbackThrows) {
    EXPECT_EXIT(std::exit(headlessCallbackFailure(false)), ::testing::ExitedWithCode(0), "");
}

TEST(FrameUpdateFailureDeathTest, windowedCanShutDownAfterCallbackThrows) {
    EXPECT_EXIT(std::exit(windowedCallbackFailure(false)), ::testing::ExitedWithCode(0), "");
}

TEST(FrameUpdateFailureDeathTest, headlessCanRenderAfterRepeatedCallbackFailures) {
    EXPECT_EXIT(std::exit(headlessCallbackFailure(true)), ::testing::ExitedWithCode(0), "");
}

TEST(FrameUpdateFailureDeathTest, windowedCanRenderAfterRepeatedCallbackFailures) {
    EXPECT_EXIT(std::exit(windowedCallbackFailure(true)), ::testing::ExitedWithCode(0), "");
}
