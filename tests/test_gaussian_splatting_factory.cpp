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
 **/
#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/vulkan_gaussian_splatting.hpp"
#include "klartraum/vulkan_gaussian_splatting_raster.hpp"
#include "klartraum/interface_camera_orbit.hpp"

using namespace klartraum;

namespace {

void expectRendersNonBlackImage(VulkanContext& vc, KlartraumEngine& engine,
                                 std::shared_ptr<ComputeGraphElement> splatting,
                                 std::shared_ptr<CameraUboType> cameraUBO) {
    engine.add(splatting);

    for (uint32_t i = 0; i < vc.getNumberOfSwapChainImages(); ++i)
        cameraUBO->update(i);

    for (int f = 0; f < 3; ++f) {
        engine.step();
        vkQueueWaitIdle(vc.getGraphicsQueue());
    }

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
    const uint8_t* pixels = static_cast<const uint8_t*>(data);
    uint8_t maxVal = *std::max_element(pixels, pixels + bytes);
    vkUnmapMemory(vc.getDevice(), mem);
    vkFreeMemory(vc.getDevice(), mem, nullptr);
    vkDestroyBuffer(vc.getDevice(), buf, nullptr);

    EXPECT_GT(maxVal, uint8_t(10)) << "Rendered image is all-black — pipeline drew nothing";
}

} // namespace

TEST(GaussianSplattingFactory, createGaussianSplattingSelectsRequestedBackend) {
    const std::string spzPath = "3rdparty/spz/samples/racoonfamily.spz";
    if (!std::filesystem::exists(spzPath)) {
        GTEST_SKIP() << "SPZ sample not found: " << spzPath;
    }

    for (auto [backend, expectCompute] : {
             std::pair{GsplatBackend::Compute, true},
             std::pair{GsplatBackend::Raster,  false} }) {

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

        auto splatting = createGaussianSplatting(vc, backend, imageViewSrc, cameraUBO, spzPath);

        if (expectCompute) {
            EXPECT_TRUE(std::dynamic_pointer_cast<VulkanGaussianSplatting>(splatting))
                << "GsplatBackend::Compute must yield a VulkanGaussianSplatting";
            EXPECT_FALSE(std::dynamic_pointer_cast<VulkanGaussianSplattingRaster>(splatting));
        } else {
            EXPECT_TRUE(std::dynamic_pointer_cast<VulkanGaussianSplattingRaster>(splatting))
                << "GsplatBackend::Raster must yield a VulkanGaussianSplattingRaster";
            EXPECT_FALSE(std::dynamic_pointer_cast<VulkanGaussianSplatting>(splatting));
        }

        expectRendersNonBlackImage(vc, engine, splatting, cameraUBO);
    }
}
