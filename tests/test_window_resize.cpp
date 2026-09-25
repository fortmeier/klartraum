/**
 * TESTS:
 * - recreateSwapChainKeepsSize: recreating the swapchain of an unresized
 *   window succeeds, keeps extent and image count, clears the outdated flag
 *   and yields valid image views and per-image semaphores
 * - framebufferExtentChangeMarksSwapChainOutOfDate: reporting a framebuffer
 *   size that differs from the last reported one marks the swapchain as
 *   outdated; reporting the same size again does not
 * - graphBuilderRunsOnceWhenSet: setGraphBuilder() runs the builder exactly
 *   once right away, makes the engine resizable, and rendering frames without
 *   a resize does not run it again
 * - computeBackendFollowsWindowResize: a Gaussian-splatting graph on the
 *   compute backend is rebuilt by the graph builder after the window is
 *   resized; the rebuilt graph sees the new swapchain extent, which matches
 *   the window's framebuffer size, and further frames render without error
 * - rasterBackendFollowsWindowResize: same as above for the raster backend
 * - consecutiveResizesRebuildOnce: two resizes between frames lead to a
 *   single rebuild at the final size
 **/
#include <gtest/gtest.h>

#include <filesystem>
#include <memory>
#include <vector>

#include "klartraum/glfw_frontend.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/gaussian_data_standard.hpp"
#include "klartraum/interface_camera_orbit.hpp"

using namespace klartraum;

namespace {

const std::string kSpzPath = "3rdparty/spz/samples/racoonfamily.spz";

VkExtent2D framebufferSize(GLFWwindow* window) {
    int w = 0, h = 0;
    glfwGetFramebufferSize(window, &w, &h);
    return { static_cast<uint32_t>(w), static_cast<uint32_t>(h) };
}

// Renders `frames` frames the way GlfwFrontend::loop() does, but without
// closing the window afterwards so the test can continue.
void renderFrames(KlartraumEngine& engine, int frames) {
    for (int i = 0; i < frames; ++i) {
        glfwPollEvents();
        engine.step();
    }
}

// Resizes the window (in screen coordinates) and waits until GLFW reports
// the new framebuffer size, which is what triggers the swapchain rebuild.
void resizeWindow(GLFWwindow* window, int width, int height) {
    VkExtent2D before = framebufferSize(window);
    glfwSetWindowSize(window, width, height);
    for (int i = 0; i < 100; ++i) {
        glfwPollEvents();
        VkExtent2D now = framebufferSize(window);
        if (now.width != before.width || now.height != before.height) {
            return;
        }
    }
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

// Sets up a Gaussian-splatting scene through a graph builder and records the
// swapchain extent each (re)build saw.
struct GsplatResizeScene {
    std::shared_ptr<GaussianDataStandard> model;
    std::vector<VkExtent2D> builtExtents;

    GsplatResizeScene(KlartraumEngine& engine, GsplatBackend backend) {
        auto& vc = engine.getVulkanContext();
        model = std::make_shared<GaussianDataStandard>(vc, kSpzPath);

        auto camera = std::make_shared<InterfaceCameraOrbit>(InterfaceCameraOrbit::UpDirection::Y);
        camera->setAzimuth(0.9f);
        camera->setElevation(-0.5f);
        camera->setPosition({-0.5f, 0.0f, 0.5f});
        camera->setDistance(1.0f);
        engine.setInterfaceCamera(camera);

        auto modelRef = model;
        engine.setGraphBuilder([this, backend, modelRef](KlartraumEngine& e) {
            auto& ctx = e.getVulkanContext();
            builtExtents.push_back(ctx.getSwapChainExtent());
            auto cameraUBO = std::make_shared<CameraUboType>();
            auto splatting = createGaussianSplatting(
                ctx, backend, makeSwapChainImageViewSrc(ctx), cameraUBO, modelRef);
            e.add(splatting);
            e.setCameraUBO(cameraUBO);
        });
    }
};

void expectBackendFollowsResize(GsplatBackend backend) {
    if (!std::filesystem::exists(kSpzPath)) {
        GTEST_SKIP() << "SPZ sample not found: " << kSpzPath;
    }
    GlfwFrontend frontend;
    auto& engine = frontend.getKlartraumEngine();
    auto& vc = engine.getVulkanContext();
    GLFWwindow* window = frontend.getGlfwWindow();

    GsplatResizeScene scene(engine, backend);
    renderFrames(engine, 3);
    ASSERT_EQ(scene.builtExtents.size(), 1u);

    int w = 0, h = 0;
    glfwGetWindowSize(window, &w, &h);
    resizeWindow(window, w + 160, h + 96);
    VkExtent2D fb = framebufferSize(window);
    ASSERT_NE(fb.width, scene.builtExtents[0].width) << "window did not resize";

    renderFrames(engine, 5);

    ASSERT_EQ(scene.builtExtents.size(), 2u) << "graph was not rebuilt after the resize";
    EXPECT_EQ(scene.builtExtents[1].width, fb.width);
    EXPECT_EQ(scene.builtExtents[1].height, fb.height);
    EXPECT_EQ(vc.getSwapChainExtent().width, fb.width);
    EXPECT_EQ(vc.getSwapChainExtent().height, fb.height);
    EXPECT_FALSE(vc.isSwapChainOutOfDate());
}

} // namespace

TEST(WindowResize, recreateSwapChainKeepsSize) {
    GlfwFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();

    VkExtent2D extentBefore = vc.getSwapChainExtent();
    uint32_t imagesBefore = vc.getNumberOfSwapChainImages();

    ASSERT_TRUE(vc.recreateSwapChain());

    EXPECT_EQ(vc.getSwapChainExtent().width, extentBefore.width);
    EXPECT_EQ(vc.getSwapChainExtent().height, extentBefore.height);
    EXPECT_EQ(vc.getNumberOfSwapChainImages(), imagesBefore);
    EXPECT_FALSE(vc.isSwapChainOutOfDate());
    ASSERT_EQ(vc.imageAvailableSemaphoresPerImage.size(), vc.getNumberOfSwapChainImages());
    for (uint32_t i = 0; i < vc.getNumberOfSwapChainImages(); ++i) {
        EXPECT_NE(vc.getImageView(i), VK_NULL_HANDLE);
        EXPECT_NE(vc.imageAvailableSemaphoresPerImage[i], VK_NULL_HANDLE);
    }
}

TEST(WindowResize, framebufferExtentChangeMarksSwapChainOutOfDate) {
    GlfwFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    VkExtent2D current = framebufferSize(frontend.getGlfwWindow());

    vc.setFramebufferExtent(current);
    EXPECT_FALSE(vc.isSwapChainOutOfDate());

    vc.setFramebufferExtent({ current.width + 10, current.height });
    EXPECT_TRUE(vc.isSwapChainOutOfDate());
}

TEST(WindowResize, graphBuilderRunsOnceWhenSet) {
    GlfwFrontend frontend;
    auto& engine = frontend.getKlartraumEngine();
    EXPECT_FALSE(engine.isResizable());

    int calls = 0;
    engine.setGraphBuilder([&calls](KlartraumEngine& e) {
        ++calls;
        e.add(e.createRenderPass());
    });
    EXPECT_EQ(calls, 1);
    EXPECT_TRUE(engine.isResizable());

    renderFrames(engine, 3);
    EXPECT_EQ(calls, 1);
}

TEST(WindowResize, computeBackendFollowsWindowResize) {
    expectBackendFollowsResize(GsplatBackend::Compute);
}

TEST(WindowResize, rasterBackendFollowsWindowResize) {
    expectBackendFollowsResize(GsplatBackend::Raster);
}

TEST(WindowResize, consecutiveResizesRebuildOnce) {
    if (!std::filesystem::exists(kSpzPath)) {
        GTEST_SKIP() << "SPZ sample not found: " << kSpzPath;
    }
    GlfwFrontend frontend;
    auto& engine = frontend.getKlartraumEngine();
    GLFWwindow* window = frontend.getGlfwWindow();

    GsplatResizeScene scene(engine, GsplatBackend::Raster);
    renderFrames(engine, 2);

    int w = 0, h = 0;
    glfwGetWindowSize(window, &w, &h);
    resizeWindow(window, w + 64, h + 64);
    resizeWindow(window, w + 128, h + 32);
    VkExtent2D fb = framebufferSize(window);

    // One step performs the rebuild; the rest render normally.
    engine.step();
    renderFrames(engine, 3);

    ASSERT_EQ(scene.builtExtents.size(), 2u);
    EXPECT_EQ(scene.builtExtents[1].width, fb.width);
    EXPECT_EQ(scene.builtExtents[1].height, fb.height);
}
