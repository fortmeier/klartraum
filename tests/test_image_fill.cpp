// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - fillTakesTheColourOfEachSubmission: imageFill fills an offscreen target with the buffer's colour, and a colour
 *   changed between submissions of the recorded graph fills the next one
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
    auto colour = std::make_shared<HostValues<glm::vec4>>(vc, std::vector<glm::vec4>{glm::vec4(1.0f)});
    auto fill = imageFill(vc, target, colour);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(fill);
    for (const float grey : {1.0f, 0.0f, 0.5f}) {
        colour->set(0, glm::vec4(grey, grey, grey, 1.0f));
        graph.submitAndWait(vc.getGraphicsQueue(), 0);
        const auto image = tree_test::readImage(vc, target->getImage(0), VK_IMAGE_LAYOUT_GENERAL, extent);
        ASSERT_EQ(image.size(), 4u * extent.width * extent.height);
        for (size_t i = 0; i < image.size(); ++i) {
            if (i % 4 == 3)
                continue;
            ASSERT_NEAR(float(image[i]), grey * 255.0f, 1.0f) << "byte " << i << " grey " << grey;
        }
    }
}
