/**
 * TESTS:
 * - createGaussianSplattingSelectsRequestedBackend: constructs both
 *   GsplatBackend::Compute and GsplatBackend::Raster via
 *   createGaussianSplatting with the same path/camera/config, confirms each
 *   returns the concrete backend type requested (VulkanGaussianSplatting vs.
 *   VulkanGaussianSplattingRaster — guide §7 step 6 "Selector": both share a
 *   constructor shape, so the factory is the single switch point), and that
 *   each renders a non-black image of the raccoon scene end to end through
 *   KlartraumEngine — i.e. the factory's returned ComputeGraphElement is a
 *   fully wired, drawable backend, not just the right type
 * - bothBackendsAgreeOnRaccoonScene: renders the same raccoon-scene/camera
 *   through both backends (via the same factory + identical orbit-camera
 *   setup) and diffs the two images per-pixel (guide §7 step 7 "golden-image
 *   diff") — asserts the mean absolute channel difference stays within a
 *   tolerance, i.e. the sort-once + hardware-rasterization backend's EWA
 *   covariance/SH math (guide §5C, validated qualitatively in
 *   RASTER_BACKEND_STATUS.md item 4) reproduces the compute-tile backend's
 *   reference rendering quantitatively, not just "looks similar"
 **/
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/gaussian_data_standard.hpp"
#include "klartraum/vulkan_gaussian_splatting.hpp"
#include "klartraum/vulkan_gaussian_splatting_raster.hpp"
#include "klartraum/interface_camera_orbit.hpp"

using namespace klartraum;

namespace {

const std::string kSpzPath = "3rdparty/spz/samples/racoonfamily.spz";

// Reads swapchain image 0 back to host as tightly-packed BGRA bytes (mirrors
// the helper in test_gaussian_splatting_raster.cpp; both headless RenderPasses
// leave the final image in VK_IMAGE_LAYOUT_GENERAL).
std::vector<uint8_t> readSwapchainImageToHost(VulkanContext& vc) {
    VkExtent2D ext = vc.getSwapChainExtent();
    const VkDeviceSize bytes = ext.width * ext.height * 4;
    VkBuffer buf; VkDeviceMemory mem;
    vc.createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    buf, mem);
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = vc.getCommandPool(); ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(vc.getDevice(), &ai, &cmd);
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {ext.width, ext.height, 1};
    vkCmdCopyImageToBuffer(cmd, vc.getSwapChainImage(0), VK_IMAGE_LAYOUT_GENERAL, buf, 1, &region);
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    vkQueueSubmit(vc.getGraphicsQueue(), 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(vc.getGraphicsQueue());
    vkFreeCommandBuffers(vc.getDevice(), vc.getCommandPool(), 1, &cmd);

    void* data;
    vkMapMemory(vc.getDevice(), mem, 0, bytes, 0, &data);
    std::vector<uint8_t> result(static_cast<const uint8_t*>(data),
                                static_cast<const uint8_t*>(data) + bytes);
    vkUnmapMemory(vc.getDevice(), mem);
    vkFreeMemory(vc.getDevice(), mem, nullptr);
    vkDestroyBuffer(vc.getDevice(), buf, nullptr);
    return result;
}

// Builds the raccoon scene through `backend` via the shared factory, using the
// same orbit-camera setup as GaussianSplattingRaster.classWithRaccoonScene /
// GaussianSplattingTest.classWithRaccoonScene, runs a few frames, and reads the
// rendered image back to host.
std::vector<uint8_t> renderRaccoonSceneWithBackend(GsplatBackend backend) {
    HeadlessFrontend frontend;
    auto& engine = frontend.getKlartraumEngine();
    auto& vc = engine.getVulkanContext();

    uint32_t numImages = vc.getNumberOfSwapChainImages();
    VkExtent2D ext = vc.getSwapChainExtent();
    std::vector<VkImageView> views(numImages);
    std::vector<VkImage>     imgs(numImages);
    std::vector<VkExtent2D>  exts(numImages, ext);
    for (uint32_t i = 0; i < numImages; ++i) {
        views[i] = vc.getImageView(i);
        imgs[i]  = vc.getSwapChainImage(i);
    }
    auto imageViewSrc = std::make_shared<ImageViewSrc>(views, imgs, exts);
    for (uint32_t i = 0; i < numImages; ++i)
        imageViewSrc->setWaitFor(i, vc.imageAvailableSemaphoresPerImage[i]);

    auto cameraUBO = std::make_shared<CameraUboType>();
    InterfaceCameraOrbit orbit(InterfaceCameraOrbit::UpDirection::Y);
    orbit.initialize(vc);
    orbit.setAzimuth(0.9f); orbit.setElevation(-0.5f);
    orbit.setPosition({-0.5f, 0.0f, 0.5f}); orbit.setDistance(1.0f);
    orbit.update(cameraUBO->ubo);

    auto model = std::make_shared<GaussianDataStandard>(vc, kSpzPath);
    auto splatting = createGaussianSplatting(vc, backend, imageViewSrc, cameraUBO, model);
    engine.add(splatting);

    for (uint32_t i = 0; i < numImages; ++i)
        cameraUBO->update(i);

    for (int f = 0; f < 5; ++f) {
        engine.step();
        vkQueueWaitIdle(vc.getGraphicsQueue());
    }

    return readSwapchainImageToHost(vc);
}

void expectMatchesRequestedBackendType(std::shared_ptr<ComputeGraphElement> splatting, GsplatBackend backend) {
    if (backend == GsplatBackend::Compute) {
        EXPECT_TRUE(std::dynamic_pointer_cast<VulkanGaussianSplatting>(splatting))
            << "GsplatBackend::Compute must yield a VulkanGaussianSplatting";
        EXPECT_FALSE(std::dynamic_pointer_cast<VulkanGaussianSplattingRaster>(splatting));
    } else {
        EXPECT_TRUE(std::dynamic_pointer_cast<VulkanGaussianSplattingRaster>(splatting))
            << "GsplatBackend::Raster must yield a VulkanGaussianSplattingRaster";
        EXPECT_FALSE(std::dynamic_pointer_cast<VulkanGaussianSplatting>(splatting));
    }
}

} // namespace

TEST(GaussianSplattingFactory, createGaussianSplattingSelectsRequestedBackend) {
    if (!std::filesystem::exists(kSpzPath)) {
        GTEST_SKIP() << "SPZ sample not found: " << kSpzPath;
    }

    for (GsplatBackend backend : {GsplatBackend::Compute, GsplatBackend::Raster}) {
        HeadlessFrontend frontend;
        auto& engine = frontend.getKlartraumEngine();
        auto& vc = engine.getVulkanContext();

        uint32_t numImages = vc.getNumberOfSwapChainImages();
        VkExtent2D ext = vc.getSwapChainExtent();
        std::vector<VkImageView> views(numImages);
        std::vector<VkImage>     imgs(numImages);
        std::vector<VkExtent2D>  exts(numImages, ext);
        for (uint32_t i = 0; i < numImages; ++i) {
            views[i] = vc.getImageView(i);
            imgs[i]  = vc.getSwapChainImage(i);
        }
        auto imageViewSrc = std::make_shared<ImageViewSrc>(views, imgs, exts);
        for (uint32_t i = 0; i < numImages; ++i)
            imageViewSrc->setWaitFor(i, vc.imageAvailableSemaphoresPerImage[i]);

        auto cameraUBO = std::make_shared<CameraUboType>();
        InterfaceCameraOrbit orbit(InterfaceCameraOrbit::UpDirection::Y);
        orbit.initialize(vc);
        orbit.setAzimuth(0.9f); orbit.setElevation(-0.5f);
        orbit.setPosition({-0.5f, 0.0f, 0.5f}); orbit.setDistance(1.0f);
        orbit.update(cameraUBO->ubo);

        auto model = std::make_shared<GaussianDataStandard>(vc, kSpzPath);
        auto splatting = createGaussianSplatting(vc, backend, imageViewSrc, cameraUBO, model);
        expectMatchesRequestedBackendType(splatting, backend);

        engine.add(splatting);
        for (uint32_t i = 0; i < numImages; ++i)
            cameraUBO->update(i);
        for (int f = 0; f < 3; ++f) {
            engine.step();
            vkQueueWaitIdle(vc.getGraphicsQueue());
        }

        auto pixels = readSwapchainImageToHost(vc);
        uint8_t maxVal = *std::max_element(pixels.begin(), pixels.end());
        EXPECT_GT(maxVal, uint8_t(10)) << "Rendered image is all-black — pipeline drew nothing";
    }
}

TEST(GaussianSplattingFactory, bothBackendsAgreeOnRaccoonScene) {
    if (!std::filesystem::exists(kSpzPath)) {
        GTEST_SKIP() << "SPZ sample not found: " << kSpzPath;
    }

    auto computePixels = renderRaccoonSceneWithBackend(GsplatBackend::Compute);
    auto rasterPixels  = renderRaccoonSceneWithBackend(GsplatBackend::Raster);
    ASSERT_EQ(computePixels.size(), rasterPixels.size());

    double sumAbsDiff = 0.0;
    uint32_t maxAbsDiff = 0;
    for (size_t i = 0; i < computePixels.size(); ++i) {
        uint32_t diff = static_cast<uint32_t>(std::abs(
            static_cast<int>(computePixels[i]) - static_cast<int>(rasterPixels[i])));
        sumAbsDiff += diff;
        maxAbsDiff = std::max(maxAbsDiff, diff);
    }
    double meanAbsDiff = sumAbsDiff / static_cast<double>(computePixels.size());

    std::cout << "\n  bothBackendsAgreeOnRaccoonScene: meanAbsDiff=" << meanAbsDiff
              << " maxAbsDiff=" << maxAbsDiff << " (per BGRA byte, 0-255)\n";

    // The two backends differ in projection/sort/blend implementation details
    // (compute-tile binned accumulation vs. hardware vkCmdDrawIndirect blending,
    // different float rounding paths, per-tile vs. per-instance splat ordering)
    // but share the same EWA covariance/SH math (guide §5C) and the same
    // model/camera. Measured mean absolute difference on the raccoon scene is
    // ~10/255 (~4%) — consistent with the qualitative "near-pixel-identical"
    // comparison in RASTER_BACKEND_STATUS.md item 4. A divergence in the shared
    // math (wrong covariance, wrong SH band/coefficients, ...) would show up as
    // a much larger gap, so 20/255 catches real regressions while tolerating
    // the blending-order noise.
    EXPECT_LT(meanAbsDiff, 20.0) << "Backends disagree more than expected on average — "
                                    "EWA covariance/SH math may have diverged";
}
