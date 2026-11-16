// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - fillTakesTheColourOfEachSubmission: ImageFill fills an offscreen target with its colour, and a colour changed
 *   between submissions of the recorded graph fills the next one, in each channel (red, green, blue apart)
 **/

#include <gtest/gtest.h>

#include <memory>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/image_fill.hpp"
#include "klartraum/offscreen_target.hpp"

#include "tree_test_helpers.hpp"

using namespace klartraum;

TEST(ImageFill, fillTakesTheColourOfEachSubmission) {
    HeadlessFrontend frontend;
    VulkanContext& vc = frontend.getKlartraumEngine().getVulkanContext();
    const VkExtent2D extent{37, 21};
    auto target = std::make_shared<OffscreenTarget>(vc, extent, 1);
    auto fill = std::make_shared<ImageFill>();
    fill->setInput(target, 0);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(fill);
    for (const glm::vec3 colour : {glm::vec3(1.0f), glm::vec3(0.0f), glm::vec3(0.2f, 0.5f, 0.8f)}) {
        fill->setColour(colour);
        graph.submitAndWait(vc.getGraphicsQueue(), 0);
        const auto image = tree_test::readImage(vc, target->getImage(0), VK_IMAGE_LAYOUT_GENERAL, extent);
        ASSERT_EQ(image.size(), 4u * extent.width * extent.height);
        for (size_t p = 0; p < image.size() / 4; ++p)
            for (int c = 0; c < 3; ++c) // BGRA bytes
                ASSERT_NEAR(float(image[4 * p + c]), colour[2 - c] * 255.0f, 1.0f) << "pixel " << p << " byte " << c;
    }
}
