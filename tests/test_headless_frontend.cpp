#include <gtest/gtest.h>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/draw_basics.hpp"

using namespace klartraum;

TEST(HeadlessFrontend, Smoketest) {
    HeadlessFrontend frontend;
    auto& core = frontend.getKlartraumEngine();
    auto& vulkanContext = core.getVulkanContext();

    core.add(core.createRenderPass());
    core.step();
}

// Renders 10 frames through the full headless render loop with a DrawBasics
// (axes) component.  Verifies that the pipeline compiles, submits, and
// completes without a crash or validation error for repeated frames.
TEST(HeadlessFrontend, RenderDrawBasics10Frames) {
    HeadlessFrontend frontend;
    auto& core = frontend.getKlartraumEngine();

    auto renderpass = core.createRenderPass();
    auto axes = std::make_shared<DrawBasics>(DrawBasicsType::Axes);
    renderpass->addDrawComponent(axes);
    core.add(renderpass);

    for (int i = 0; i < 10; ++i) {
        core.step();
    }
}
