/**
 * TESTS:
 * - runsGuiCallbackOncePerFrame: an ImGuiFrontend over a DrawBasics scene
 *   renders 10 frames through loop(); the GUI callback (building a window
 *   with text and a button) runs exactly once per frame, and setup, frames
 *   and teardown finish without a validation error
 * - guiFollowsWindowResize: with a raster Gaussian-splatting scene built by a
 *   graph builder, resizing the window rebuilds the scene for the new
 *   swapchain, ImGui's display size follows the new window size, and frames
 *   with the GUI keep rendering at the new size
 * - rebuildsSceneFromGuiCallback: the GUI callback waits for the device to be
 *   idle and re-sets the graph builder mid-loop (as a GUI "apply settings"
 *   button does); the scene is rebuilt once and the following frames render
 *   the new graphs with the GUI without a validation error
 **/
#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <vector>

#include <imgui.h>

#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/draw_basics.hpp"
#include "klartraum/gaussian_data_standard.hpp"
#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/imgui_frontend.hpp"
#include "klartraum/interface_camera_orbit.hpp"

using namespace klartraum;

namespace {

const std::string kSpzPath = "3rdparty/spz/samples/racoonfamily.spz";

void buildTestWindow() {
    ImGui::Begin("Test");
    ImGui::Text("frame %d", ImGui::GetFrameCount());
    ImGui::Button("Button");
    ImGui::End();
}

std::shared_ptr<ImageViewSrc> makeSwapChainImageViewSrc(VulkanContext& vc) {
    uint32_t numImages = vc.getNumberOfSwapChainImages();
    std::vector<VkImageView> imageViews(numImages);
    std::vector<VkImage> images(numImages);
    std::vector<VkExtent2D> extents(numImages, vc.getSwapChainExtent());
    for (uint32_t i = 0; i < numImages; ++i) {
        imageViews[i] = vc.getImageView(i);
        images[i] = vc.getSwapChainImage(i);
    }
    auto imageViewSrc = std::make_shared<ImageViewSrc>(imageViews, images, extents);
    for (uint32_t i = 0; i < numImages; ++i) {
        imageViewSrc->setWaitFor(i, vc.imageAvailableSemaphoresPerImage[i]);
    }
    return imageViewSrc;
}

}  // namespace

TEST(ImGuiFrontend, runsGuiCallbackOncePerFrame) {
    ImGuiFrontend frontend;
    auto& engine = frontend.getKlartraumEngine();

    auto renderpass = engine.createRenderPass();
    renderpass->addDrawComponent(std::make_shared<DrawBasics>(DrawBasicsType::Axes));
    engine.add(renderpass);

    int guiCalls = 0;
    frontend.setGui([&guiCalls] {
        ++guiCalls;
        buildTestWindow();
    });

    frontend.loop(10);

    EXPECT_EQ(guiCalls, 10);
}

TEST(ImGuiFrontend, guiFollowsWindowResize) {
    if (!std::filesystem::exists(kSpzPath)) {
        GTEST_SKIP() << "SPZ sample not found: " << kSpzPath;
    }
    ImGuiFrontend frontend;
    auto& engine = frontend.getKlartraumEngine();
    auto& vc = engine.getVulkanContext();
    GLFWwindow* window = frontend.getGlfwWindow();

    auto model = std::make_shared<GaussianDataStandard>(vc, kSpzPath);
    auto camera = std::make_shared<InterfaceCameraOrbit>(InterfaceCameraOrbit::UpDirection::Y);
    camera->setDistance(1.0f);
    engine.setInterfaceCamera(camera);

    std::vector<VkExtent2D> builtExtents;
    engine.setGraphBuilder([&builtExtents, model](KlartraumEngine& e) {
        auto& ctx = e.getVulkanContext();
        builtExtents.push_back(ctx.getSwapChainExtent());
        auto cameraUBO = std::make_shared<CameraUboType>();
        e.add(createGaussianSplatting(ctx, GsplatBackend::Raster, makeSwapChainImageViewSrc(ctx), cameraUBO, model));
        e.setCameraUBO(cameraUBO);
    });

    int guiCalls = 0;
    frontend.setGui([&guiCalls] {
        ++guiCalls;
        buildTestWindow();
    });

    frontend.loop(3);
    ASSERT_EQ(builtExtents.size(), 1u);

    int w = 0, h = 0;
    glfwGetWindowSize(window, &w, &h);
    const int newWidth = w + 160;
    const int newHeight = h + 96;
    glfwSetWindowSize(window, newWidth, newHeight);
    int fbWidth = 0, fbHeight = 0;
    for (int i = 0; i < 100; ++i) {
        frontend.pollEvents();
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        if (static_cast<uint32_t>(fbWidth) != builtExtents[0].width) break;
    }
    ASSERT_NE(static_cast<uint32_t>(fbWidth), builtExtents[0].width) << "window did not resize";

    // loop(3) requested the window to close; keep it open for more frames.
    glfwSetWindowShouldClose(window, GLFW_FALSE);
    const int guiCallsBefore = guiCalls;
    frontend.loop(5);

    ASSERT_GE(builtExtents.size(), 2u) << "scene was not rebuilt after the resize";
    EXPECT_EQ(builtExtents.back().width, static_cast<uint32_t>(fbWidth));
    EXPECT_EQ(builtExtents.back().height, static_cast<uint32_t>(fbHeight));
    EXPECT_EQ(vc.getSwapChainExtent().width, static_cast<uint32_t>(fbWidth));
    EXPECT_EQ(vc.getSwapChainExtent().height, static_cast<uint32_t>(fbHeight));
    EXPECT_GE(guiCalls - guiCallsBefore, 5);

    EXPECT_EQ(ImGui::GetIO().DisplaySize.x, static_cast<float>(newWidth));
    EXPECT_EQ(ImGui::GetIO().DisplaySize.y, static_cast<float>(newHeight));
}

TEST(ImGuiFrontend, rebuildsSceneFromGuiCallback) {
    if (!std::filesystem::exists(kSpzPath)) {
        GTEST_SKIP() << "SPZ sample not found: " << kSpzPath;
    }
    ImGuiFrontend frontend;
    auto& engine = frontend.getKlartraumEngine();
    auto& vc = engine.getVulkanContext();

    auto model = std::make_shared<GaussianDataStandard>(vc, kSpzPath);
    auto camera = std::make_shared<InterfaceCameraOrbit>(InterfaceCameraOrbit::UpDirection::Y);
    camera->setDistance(1.0f);
    engine.setInterfaceCamera(camera);

    GsplatConfig config;
    int builds = 0;
    auto builder = [&builds, &config, model](KlartraumEngine& e) {
        ++builds;
        auto& ctx = e.getVulkanContext();
        auto cameraUBO = std::make_shared<CameraUboType>();
        e.add(createGaussianSplatting(ctx, GsplatBackend::Raster, makeSwapChainImageViewSrc(ctx), cameraUBO, model,
                                      config));
        e.setCameraUBO(cameraUBO);
    };
    engine.setGraphBuilder(builder);
    ASSERT_EQ(builds, 1);

    int guiCalls = 0;
    frontend.setGui([&] {
        ++guiCalls;
        buildTestWindow();
        if (guiCalls == 3) {
            vkDeviceWaitIdle(vc.getDevice());
            config.shDegree = 0;
            engine.setGraphBuilder(builder);
        }
    });

    frontend.loop(8);

    EXPECT_EQ(builds, 2);
    EXPECT_EQ(guiCalls, 8);
}
