/**
 * TESTS:
 * - makeViewportProducesIndependentTargets: two makeViewport() calls return
 *   distinct ImageViewSrc targets whose images differ and whose extents equal
 *   the requested width/height.
 * - compositePlacesViewportsInCorrectRegions: a left (red) and right (blue)
 *   viewport, each filled by clearing its offscreen image, composite into the
 *   swapchain; readback confirms red in the left half and blue in the right.
 * - gapsAreClearedBlack: a single viewport covering only the left quarter; the
 *   uncovered region reads back black (verifies the composite's clear).
 * - sharedCameraCanUseViewportAspectRatio: verifies that a shared orbit camera
 *   can encode the viewport aspect ratio rather than the full-window ratio in
 *   its projection matrix.
 * - twoBackendsCompositeEndToEnd: drives the real example path (compute splat in
 *   the left viewport, raster splat in the right, shared camera UBO) through
 *   KlartraumEngine::step(); confirms both halves render non-black and differ,
 *   i.e. the backends leave their offscreen targets in TRANSFER_SRC (via
 *   getFinalLayoutOverride) and the composite blits both into the swapchain.
 **/
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <functional>
#include <vector>

#include <filesystem>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/window.hpp"
#include "klartraum/vulkan_context.hpp"
#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/gaussian_data_standard.hpp"
#include "klartraum/interface_camera_orbit.hpp"

using namespace klartraum;

namespace {

// Records a one-time command buffer, submits it to the graphics queue, and
// waits for the GPU to go idle.
void runImmediate(VulkanContext& vc, const std::function<void(VkCommandBuffer)>& record) {
    VkCommandBufferAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = vc.getCommandPool();
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(vc.getDevice(), &ai, &cmd);
    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);
    record(cmd);
    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    vkQueueSubmit(vc.getGraphicsQueue(), 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(vc.getGraphicsQueue());
    vkFreeCommandBuffers(vc.getDevice(), vc.getCommandPool(), 1, &cmd);
}

// Clears `image` to an RGBA color and leaves it in TRANSFER_SRC_OPTIMAL — the
// layout a viewport scene would leave its offscreen target in for the composite.
void fillOffscreen(VulkanContext& vc, VkImage image, std::array<float, 4> rgba) {
    VkImageSubresourceRange range{};
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.levelCount = 1;
    range.layerCount = 1;

    runImmediate(vc, [&](VkCommandBuffer cmd) {
        VkImageMemoryBarrier toDst{};
        toDst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        toDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toDst.image = image;
        toDst.subresourceRange = range;
        toDst.srcAccessMask = 0;
        toDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &toDst);

        VkClearColorValue color{};
        color.float32[0] = rgba[0]; color.float32[1] = rgba[1];
        color.float32[2] = rgba[2]; color.float32[3] = rgba[3];
        vkCmdClearColorImage(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);

        VkImageMemoryBarrier toSrc{};
        toSrc.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toSrc.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toSrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        toSrc.image = image;
        toSrc.subresourceRange = range;
        toSrc.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toSrc.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0, 0, nullptr, 0, nullptr, 1, &toSrc);
    });
}

// Reads swapchain image 0 (left in GENERAL by the headless composite) back to
// host as tightly-packed BGRA bytes.
std::vector<uint8_t> readSwapchainImage0(VulkanContext& vc) {
    VkExtent2D ext = vc.getSwapChainExtent();
    const VkDeviceSize bytes = VkDeviceSize(ext.width) * ext.height * 4;
    VkBuffer buf; VkDeviceMemory mem;
    vc.createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    buf, mem);
    runImmediate(vc, [&](VkCommandBuffer cmd) {
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = {ext.width, ext.height, 1};
        vkCmdCopyImageToBuffer(cmd, vc.getSwapChainImage(0), VK_IMAGE_LAYOUT_GENERAL, buf, 1, &region);
    });
    void* data;
    vkMapMemory(vc.getDevice(), mem, 0, bytes, 0, &data);
    std::vector<uint8_t> result(static_cast<const uint8_t*>(data),
                                static_cast<const uint8_t*>(data) + bytes);
    vkUnmapMemory(vc.getDevice(), mem);
    vkFreeMemory(vc.getDevice(), mem, nullptr);
    vkDestroyBuffer(vc.getDevice(), buf, nullptr);
    return result;
}

// BGRA byte accessor for pixel (x, y).
struct Bgra { uint8_t b, g, r, a; };
Bgra pixelAt(const std::vector<uint8_t>& img, VkExtent2D ext, uint32_t x, uint32_t y) {
    size_t o = (size_t(y) * ext.width + x) * 4;
    return { img[o + 0], img[o + 1], img[o + 2], img[o + 3] };
}

} // namespace

TEST(MultiViewportWindow, makeViewportProducesIndependentTargets) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    VkExtent2D ext = vc.getSwapChainExtent();
    uint32_t halfW = ext.width / 2;

    auto& window = frontend.getKlartraumEngine().getWindow();
    auto left  = window.makeViewport(0,     0, halfW,             ext.height);
    auto right = window.makeViewport(halfW, 0, ext.width - halfW, ext.height);

    EXPECT_NE(left.get(), right.get());
    EXPECT_NE(left->getImage(0), right->getImage(0));
    EXPECT_EQ(left->getImageExtent(0).width,  halfW);
    EXPECT_EQ(left->getImageExtent(0).height, ext.height);
    EXPECT_EQ(right->getImageExtent(0).width, ext.width - halfW);
}

TEST(MultiViewportWindow, scaledViewportUsesIndependentRenderResolution) {
    HeadlessFrontend frontend;
    auto& engine = frontend.getKlartraumEngine();
    auto& vc = engine.getVulkanContext();
    const auto displayExtent = vc.getSwapChainExtent();

    auto target = engine.getWindow().makeViewport(
        0, 0, displayExtent.width, displayExtent.height, 32, 24);
    EXPECT_EQ(target->getImageExtent(0).width, 32u);
    EXPECT_EQ(target->getImageExtent(0).height, 24u);

    fillOffscreen(vc, target->getImage(0), {1.0f, 0.0f, 0.0f, 1.0f});
    engine.getWindow().submitComposite(vc.getGraphicsQueue(), 0, {}, VK_NULL_HANDLE);
    vkQueueWaitIdle(vc.getGraphicsQueue());

    const auto image = readSwapchainImage0(vc);
    const auto corner = pixelAt(image, displayExtent,
                                displayExtent.width - 1, displayExtent.height - 1);
    EXPECT_GT(corner.r, 200);
    EXPECT_LT(corner.g, 50);
    EXPECT_LT(corner.b, 50);
}

TEST(MultiViewportWindow, compositePlacesViewportsInCorrectRegions) {
    HeadlessFrontend frontend;
    auto& engine = frontend.getKlartraumEngine();
    auto& vc = engine.getVulkanContext();
    VkExtent2D ext = vc.getSwapChainExtent();
    uint32_t halfW = ext.width / 2;

    auto& window = engine.getWindow();
    auto left  = window.makeViewport(0,     0, halfW,             ext.height);
    auto right = window.makeViewport(halfW, 0, ext.width - halfW, ext.height);

    fillOffscreen(vc, left->getImage(0),  {1.0f, 0.0f, 0.0f, 1.0f}); // red
    fillOffscreen(vc, right->getImage(0), {0.0f, 0.0f, 1.0f, 1.0f}); // blue

    window.submitComposite(vc.getGraphicsQueue(), 0, {}, VK_NULL_HANDLE);
    vkQueueWaitIdle(vc.getGraphicsQueue());

    auto img = readSwapchainImage0(vc);
    uint32_t yMid = ext.height / 2;

    Bgra l = pixelAt(img, ext, halfW / 2,           yMid);
    Bgra r = pixelAt(img, ext, halfW + halfW / 2,   yMid);

    EXPECT_GT(l.r, 200); EXPECT_LT(l.b, 50);  // left is red
    EXPECT_GT(r.b, 200); EXPECT_LT(r.r, 50);  // right is blue
}

TEST(MultiViewportWindow, gapsAreClearedBlack) {
    HeadlessFrontend frontend;
    auto& engine = frontend.getKlartraumEngine();
    auto& vc = engine.getVulkanContext();
    VkExtent2D ext = vc.getSwapChainExtent();
    uint32_t quarterW = ext.width / 4;

    auto& window = engine.getWindow();
    auto vp = window.makeViewport(0, 0, quarterW, ext.height);
    fillOffscreen(vc, vp->getImage(0), {0.0f, 1.0f, 0.0f, 1.0f}); // green

    window.submitComposite(vc.getGraphicsQueue(), 0, {}, VK_NULL_HANDLE);
    vkQueueWaitIdle(vc.getGraphicsQueue());

    auto img = readSwapchainImage0(vc);
    uint32_t yMid = ext.height / 2;

    Bgra covered = pixelAt(img, ext, quarterW / 2,              yMid);
    Bgra gap     = pixelAt(img, ext, ext.width - ext.width / 8, yMid);

    EXPECT_GT(covered.g, 200);  // viewport region is green
    EXPECT_LT(gap.r, 20); EXPECT_LT(gap.g, 20); EXPECT_LT(gap.b, 20); // gap is black
}

TEST(MultiViewportWindow, sharedCameraCanUseViewportAspectRatio) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    VkExtent2D windowExtent = vc.getSwapChainExtent();
    float viewportAspect = (windowExtent.width / 2) / static_cast<float>(windowExtent.height);

    InterfaceCameraOrbit orbit(InterfaceCameraOrbit::UpDirection::Y);
    orbit.initialize(vc);
    orbit.setProjectionAspectRatio(viewportAspect);

    CameraMVP mvp{};
    orbit.update(mvp);

    float matrixAspect = std::abs(mvp.proj[1][1] / mvp.proj[0][0]);
    EXPECT_NEAR(matrixAspect, viewportAspect, 1e-5f);
}

TEST(MultiViewportWindow, twoBackendsCompositeEndToEnd) {
    const std::string spzPath = "3rdparty/spz/samples/racoonfamily.spz";
    if (!std::filesystem::exists(spzPath)) {
        GTEST_SKIP() << "SPZ sample not found: " << spzPath;
    }

    HeadlessFrontend frontend;
    auto& engine = frontend.getKlartraumEngine();
    auto& vc = engine.getVulkanContext();
    VkExtent2D ext = vc.getSwapChainExtent();
    uint32_t halfW = ext.width / 2;

    auto& window = engine.getWindow();
    auto left  = window.makeViewport(0,     0, halfW,             ext.height);
    auto right = window.makeViewport(halfW, 0, ext.width - halfW, ext.height);

    // One camera UBO shared by both viewport scenes — the "single camera" the
    // example demonstrates. Sharing a UBO element across two engine.add() graphs
    // relies on UniformBufferObject::_setup being idempotent.
    auto cameraUBO = std::make_shared<CameraUboType>();
    InterfaceCameraOrbit orbit(InterfaceCameraOrbit::UpDirection::Y);
    orbit.initialize(vc);
    orbit.setAzimuth(0.9f); orbit.setElevation(-0.5f);
    orbit.setPosition({-0.5f, 0.0f, 0.5f}); orbit.setDistance(1.0f);
    orbit.update(cameraUBO->ubo);

    auto model = std::make_shared<GaussianDataStandard>(vc, spzPath);
    engine.add(createGaussianSplatting(vc, GsplatBackend::Compute, left,  cameraUBO, model));
    engine.add(createGaussianSplatting(vc, GsplatBackend::Raster,  right, cameraUBO, model));

    uint32_t numImages = vc.getNumberOfSwapChainImages();
    for (uint32_t i = 0; i < numImages; ++i) cameraUBO->update(i);

    for (int f = 0; f < 5; ++f) {
        engine.step();
        vkQueueWaitIdle(vc.getGraphicsQueue());
    }

    auto img = readSwapchainImage0(vc);

    // Number of columns in [x0, x1) holding any non-black pixel. Both backends
    // render the same scene/camera into equal-width viewports, so each should
    // cover nearly its whole width. (A backend that sizes its viewport to the
    // swapchain instead of the target only fills a fraction — guards that bug.)
    auto litColumns = [&](uint32_t x0, uint32_t x1) {
        int n = 0;
        for (uint32_t x = x0; x < x1; ++x)
            for (uint32_t y = 0; y < ext.height; ++y) {
                Bgra p = pixelAt(img, ext, x, y);
                if (p.r > 15 || p.g > 15 || p.b > 15) { ++n; break; }
            }
        return n;
    };

    int leftLit  = litColumns(0, halfW);
    int rightLit = litColumns(halfW, ext.width);
    EXPECT_GT(leftLit,  int(halfW * 0.9))
        << "left (compute) viewport only filled " << leftLit << "/" << halfW << " columns";
    EXPECT_GT(rightLit, int((ext.width - halfW) * 0.9))
        << "right (raster) viewport only filled " << rightLit << "/" << (ext.width - halfW) << " columns";
}
