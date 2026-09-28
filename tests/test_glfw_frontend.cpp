#include <gtest/gtest.h>

#include "klartraum/glfw_frontend.hpp"
#include "klartraum/draw_basics.hpp"

using namespace klartraum;

// Renders 10 frames through the GlfwFrontend render loop with a DrawBasics
// component.  GlfwFrontend creates a real OS window and currently uses headless
// offscreen images for rendering (full windowed swapchain is a future task).
// The test verifies that setup, 10 render steps, and teardown all complete
// without a crash or validation abort.
TEST(GlfwFrontend, RenderDrawBasics10Frames) {
    GlfwFrontend frontend;
    auto& engine = frontend.getKlartraumEngine();

    auto renderpass = engine.createRenderPass();
    auto axes = std::make_shared<DrawBasics>(DrawBasicsType::Axes);
    renderpass->addDrawComponent(axes);
    engine.add(renderpass);

    for (int i = 0; i < 10; ++i) {
        engine.step();
    }
}
