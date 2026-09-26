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
 * - backendsAgreeAtBinBordersForUnalignedSize: renders both backends into an
 *   offscreen target whose size is not a multiple of the compute backend's
 *   bin/tile granularity (509x381) and checks that the backends differ no
 *   more in narrow bands around the compute backend's 4x4 bin borders and
 *   along the right/bottom image edges than they do on average, i.e. every
 *   pixel is shaded with the Gaussians of the bin it lies in (no seams, no
 *   unwritten strips)
 * - bothBackendsRenderInSinglePathGraph: builds each backend into a standalone
 *   ComputeGraph with a single path (fewer paths than swapchain images) that
 *   renders into a one-image OffscreenTarget, runs it once with
 *   submitAndWait() and checks the image is not black, i.e. the backends size
 *   their per-path resources by the graph's paths, not the swapchain
 * - uncompiledBackendsCanBeDestroyed: each backend is created and released
 *   without ever being compiled into a graph (as when building the rest of a
 *   graph fails), which must not touch Vulkan objects that were never created
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
#include "klartraum/offscreen_target.hpp"

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

// Reads an OffscreenTarget image back as tightly-packed BGRA bytes. Both
// backends leave an OffscreenTarget in its final-layout override
// (TRANSFER_SRC_OPTIMAL).
std::vector<uint8_t> readOffscreenImageToHost(VulkanContext& vc, VkImage image, VkExtent2D extent) {
    const VkDeviceSize bytes = VkDeviceSize(extent.width) * extent.height * 4;
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
    region.imageExtent = {extent.width, extent.height, 1};
    vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           buf, 1, &region);
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

// Renders the raccoon scene through `backend` into an OffscreenTarget of the
// given extent and reads image 0 back as tightly-packed BGRA bytes.
std::vector<uint8_t> renderRaccoonSceneIntoTarget(GsplatBackend backend, VkExtent2D extent) {
    HeadlessFrontend frontend;
    auto& engine = frontend.getKlartraumEngine();
    auto& vc = engine.getVulkanContext();

    uint32_t numImages = vc.getNumberOfSwapChainImages();
    auto target = std::make_shared<OffscreenTarget>(vc, extent, numImages);
    // beginRender() signals the per-image semaphore every frame; the graph
    // has to consume it just as it does for a swapchain-backed ImageViewSrc.
    for (uint32_t i = 0; i < numImages; ++i)
        target->setWaitFor(i, vc.imageAvailableSemaphoresPerImage[i]);

    auto cameraUBO = std::make_shared<CameraUboType>();
    InterfaceCameraOrbit orbit(InterfaceCameraOrbit::UpDirection::Y);
    orbit.initialize(vc);
    orbit.setAzimuth(0.9f); orbit.setElevation(-0.5f);
    orbit.setPosition({-0.5f, 0.0f, 0.5f}); orbit.setDistance(1.0f);
    orbit.setProjectionAspectRatio(extent.width / static_cast<float>(extent.height));
    orbit.update(cameraUBO->ubo);

    auto model = std::make_shared<GaussianDataStandard>(vc, kSpzPath);
    auto splatting = createGaussianSplatting(vc, backend, target, cameraUBO, model);
    engine.add(splatting);
    for (uint32_t i = 0; i < numImages; ++i)
        cameraUBO->update(i);
    for (int f = 0; f < 4; ++f) {
        engine.step();
        vkQueueWaitIdle(vc.getGraphicsQueue());
    }

    auto result = readOffscreenImageToHost(vc, target->getImage(0), extent);

    engine.clearComputeGraphs();
    splatting.reset();
    target.reset();
    return result;
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

TEST(GaussianSplattingFactory, backendsAgreeAtBinBordersForUnalignedSize) {
    if (!std::filesystem::exists(kSpzPath)) {
        GTEST_SKIP() << "SPZ sample not found: " << kSpzPath;
    }

    // Neither dimension is a multiple of 32 (4x4 bins of 8x8 tiles).
    const VkExtent2D extent{509, 381};
    auto computePixels = renderRaccoonSceneIntoTarget(GsplatBackend::Compute, extent);
    auto rasterPixels  = renderRaccoonSceneIntoTarget(GsplatBackend::Raster, extent);
    ASSERT_EQ(computePixels.size(), rasterPixels.size());

    // A pixel is "at a border" when it lies within 3 px of one of the
    // compute backend's interior bin borders (k * size / 4, k = 1..3) or
    // within 8 px of the right/bottom image edge.
    auto nearBorder = [](uint32_t p, uint32_t size) {
        for (uint32_t k = 1; k < 4; ++k) {
            float border = k * size / 4.0f;
            if (std::abs(float(p) - border) <= 3.0f) return true;
        }
        return p + 8 >= size;
    };

    double borderSum = 0.0, otherSum = 0.0;
    uint64_t borderCount = 0, otherCount = 0;
    for (uint32_t y = 0; y < extent.height; ++y) {
        for (uint32_t x = 0; x < extent.width; ++x) {
            bool border = nearBorder(x, extent.width) || nearBorder(y, extent.height);
            size_t base = (size_t(y) * extent.width + x) * 4;
            for (int c = 0; c < 3; ++c) {  // colour channels only
                double diff = std::abs(int(computePixels[base + c]) - int(rasterPixels[base + c]));
                if (border) { borderSum += diff; ++borderCount; }
                else        { otherSum  += diff; ++otherCount;  }
            }
        }
    }
    double borderMean = borderSum / double(borderCount);
    double otherMean  = otherSum / double(otherCount);

    std::cout << "\n  backendsAgreeAtBinBordersForUnalignedSize: borderMeanAbsDiff=" << borderMean
              << " otherMeanAbsDiff=" << otherMean << " (per colour byte, 0-255)\n";

    uint8_t maxVal = *std::max_element(rasterPixels.begin(), rasterPixels.end());
    ASSERT_GT(maxVal, uint8_t(10)) << "raster reference is all-black";

    // Away from the borders the backends differ only by blending-order noise
    // (see bothBackendsAgreeOnRaccoonScene). Seams or unwritten strips at the
    // bin borders would make the border bands differ far more than that.
    EXPECT_LT(borderMean, otherMean * 1.5 + 2.0)
        << "compute backend deviates at its bin borders — seams or unwritten pixels";
}

TEST(GaussianSplattingFactory, bothBackendsRenderInSinglePathGraph) {
    if (!std::filesystem::exists(kSpzPath)) {
        GTEST_SKIP() << "SPZ sample not found: " << kSpzPath;
    }

    const VkExtent2D extent{96, 96};
    for (GsplatBackend backend : {GsplatBackend::Compute, GsplatBackend::Raster}) {
        SCOPED_TRACE(backend == GsplatBackend::Raster ? "raster" : "compute");
        HeadlessFrontend frontend;
        auto& vc = frontend.getKlartraumEngine().getVulkanContext();
        ASSERT_GT(vc.getNumberOfSwapChainImages(), 1u) << "the test needs more swapchain images than graph paths";

        auto target = std::make_shared<OffscreenTarget>(vc, extent, 1u);
        auto cameraUBO = std::make_shared<CameraUboType>();
        InterfaceCameraOrbit orbit(InterfaceCameraOrbit::UpDirection::Y);
        orbit.initialize(vc);
        orbit.setAzimuth(0.9f); orbit.setElevation(-0.5f);
        orbit.setPosition({-0.5f, 0.0f, 0.5f}); orbit.setDistance(1.0f);
        orbit.setProjectionAspectRatio(1.0f);
        orbit.update(cameraUBO->ubo);

        auto model = std::make_shared<GaussianDataStandard>(vc, kSpzPath);
        auto splatting = createGaussianSplatting(vc, backend, target, cameraUBO, model);

        {
            ComputeGraph graph(vc, 1);
            graph.compileFrom(splatting);
            cameraUBO->update(0);
            graph.submitAndWait(vc.getGraphicsQueue(), 0);

            auto pixels = readOffscreenImageToHost(vc, target->getImage(0), extent);
            const uint8_t maxVal = *std::max_element(pixels.begin(), pixels.end());
            EXPECT_GT(maxVal, uint8_t(10)) << "rendered image is all-black";
        }
        splatting.reset();
        target.reset();
    }
}

TEST(GaussianSplattingFactory, uncompiledBackendsCanBeDestroyed) {
    if (!std::filesystem::exists(kSpzPath)) {
        GTEST_SKIP() << "SPZ sample not found: " << kSpzPath;
    }
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    auto model = std::make_shared<GaussianDataStandard>(vc, kSpzPath);
    for (GsplatBackend backend : {GsplatBackend::Compute, GsplatBackend::Raster}) {
        SCOPED_TRACE(backend == GsplatBackend::Raster ? "raster" : "compute");
        auto target = std::make_shared<OffscreenTarget>(vc, VkExtent2D{32, 32}, 1u);
        auto cameraUBO = std::make_shared<CameraUboType>();
        auto splatting = createGaussianSplatting(vc, backend, target, cameraUBO, model);
        splatting.reset();
    }
}
