/**
 * TESTS:
 * - classWithRaccoonScene: VulkanGaussianSplattingRaster (the sort-once +
 *   hardware-rasterization backend, guide §7 step 5 "Composite") loads the
 *   raccoon SPZ scene, runs several frames through the full dist -> sort ->
 *   barrier -> rasterizer pipeline with the placeholder point-cloud shaders,
 *   and confirms it submits cleanly across all swapchain paths (no
 *   validation-layer errors — the debug callback throws on VK_ERROR severity)
 *   and that the rendered image is not all-black (the cull/sort/indirect-draw
 *   chain actually produced and drew visible splats end to end)
 **/
#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <vector>

#include <glm/glm.hpp>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/vulkan_gaussian_splatting_raster.hpp"
#include "klartraum/interface_camera_orbit.hpp"

using namespace klartraum;

namespace {

// Minimal BGRA swapchain-image readback (mirrors the helper in
// test_gaussian_splatting.cpp; kept local since that one is file-static).
std::vector<uint8_t> readImageToHost(VulkanContext& vc, VkImage image, uint32_t W, uint32_t H) {
    const VkDeviceSize bytes = W * H * 4;
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
    region.imageExtent = {W, H, 1};
    vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_GENERAL, buf, 1, &region);
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

void writePPM(const std::string& path, const uint8_t* bgra, uint32_t W, uint32_t H) {
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << W << " " << H << "\n255\n";
    for (uint32_t y = 0; y < H; ++y)
        for (uint32_t x = 0; x < W; ++x) {
            const uint8_t* p = bgra + (y * W + x) * 4;
            uint8_t rgb[3] = { p[2], p[1], p[0] };
            f.write(reinterpret_cast<const char*>(rgb), 3);
        }
}

} // namespace

TEST(GaussianSplattingRaster, classWithRaccoonScene) {
    const std::string spzPath = "3rdparty/spz/samples/racoonfamily.spz";
    if (!std::filesystem::exists(spzPath)) {
        GTEST_SKIP() << "SPZ sample not found: " << spzPath;
    }

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

    auto splatting = vc.create<VulkanGaussianSplattingRaster>(imageViewSrc, cameraUBO, spzPath);
    engine.add(splatting);

    for (uint32_t i = 0; i < numImages; ++i)
        cameraUBO->update(i);

    const int FRAMES = 5;
    for (int f = 0; f < FRAMES; ++f) {
        engine.step();
        vkQueueWaitIdle(vc.getGraphicsQueue());
    }

    auto pixels = readImageToHost(vc, imgs[0], ext.width, ext.height);
    writePPM("test_gaussian_splatting_raster_render.ppm", pixels.data(), ext.width, ext.height);

    uint8_t maxVal = *std::max_element(pixels.begin(), pixels.end());
    std::cout << "\n  classWithRaccoonScene (raster backend): image max=" << (int)maxVal << "\n";

    EXPECT_GT(maxVal, uint8_t(10)) << "Rendered image is all-black — pipeline drew nothing";
}
