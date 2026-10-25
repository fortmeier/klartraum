// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - dumpLanternGaussians: writes the lantern scene as loaded by loadGaussiansSpz,
 *   with and without the Y flip, as raw Gaussian3D arrays plus the scene centre
 *   of test_scene::loadLantern(), for the Rust port's SPZ loader tests
 * - dumpOrbitCameraMvp: writes the CameraMVP the orbit camera produces for the
 *   frameLantern() parameters at the headless extent, for the Rust port's camera tests
 * - dumpRasterLanternRender: renders the lantern with the raster backend headless,
 *   as GaussianSplattingRaster.classWithLanternScene does, and writes the image
 *   as raw BGRA plus a PPM, the golden image of the Rust port's raster backend
 **/
#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/vulkan_gaussian_splatting_raster.hpp"
#include "klartraum/gaussian_data_standard.hpp"
#include "klartraum/interface_camera_orbit.hpp"

#include "test_scene.hpp"

using namespace klartraum;

namespace {

// The Rust port's tests read the reference data from here (see rust/README.md).
const std::filesystem::path kReferenceDir = "build/TestingOutput/rust_reference";

void writeBytes(const std::string& filename, const void* data, size_t size) {
    std::filesystem::create_directories(kReferenceDir);
    std::ofstream f(kReferenceDir / filename, std::ios::binary);
    f.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
}

std::vector<uint8_t> readImageToHost(VulkanContext& vc, VkImage image, uint32_t width, uint32_t height) {
    const VkDeviceSize bytes = VkDeviceSize(width) * height * 4;
    VkBuffer buffer;
    VkDeviceMemory memory;
    vc.createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buffer, memory);
    vc.submitImmediate([&](VkCommandBuffer cmd) {
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = {width, height, 1};
        vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_GENERAL, buffer, 1, &region);
    });
    void* data;
    vkMapMemory(vc.getDevice(), memory, 0, bytes, 0, &data);
    std::vector<uint8_t> result(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + bytes);
    vkUnmapMemory(vc.getDevice(), memory);
    vkFreeMemory(vc.getDevice(), memory, nullptr);
    vkDestroyBuffer(vc.getDevice(), buffer, nullptr);
    return result;
}

} // namespace

TEST(RustReference, dumpLanternGaussians) {
    if (!std::filesystem::exists(test_scene::kLanternPath)) {
        GTEST_SKIP() << "SPZ scene not found: " << test_scene::kLanternPath;
    }
    static_assert(sizeof(Gaussian3D) == 59 * sizeof(float), "Gaussian3D is written as 59 packed floats");

    const auto scene = test_scene::loadLantern();
    writeBytes("lantern_gaussians_flip_y.bin", scene.gaussians.data(), scene.gaussians.size() * sizeof(Gaussian3D));
    writeBytes("lantern_centre.bin", &scene.centre, sizeof(scene.centre));

    const auto unflipped = loadGaussiansSpz(test_scene::kLanternPath, false);
    writeBytes("lantern_gaussians.bin", unflipped.data(), unflipped.size() * sizeof(Gaussian3D));

    EXPECT_EQ(scene.gaussians.size(), unflipped.size());
}

TEST(RustReference, dumpOrbitCameraMvp) {
    if (!std::filesystem::exists(test_scene::kLanternPath)) {
        GTEST_SKIP() << "SPZ scene not found: " << test_scene::kLanternPath;
    }
    const auto scene = test_scene::loadLantern();

    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    InterfaceCameraOrbit orbit(InterfaceCameraOrbit::UpDirection::Y);
    orbit.initialize(vc);
    test_scene::frameLantern(orbit, scene);
    CameraMVP mvp{};
    orbit.update(mvp);

    static_assert(sizeof(CameraMVP) == 208, "CameraMVP is written in its std140 layout");
    writeBytes("lantern_camera_mvp.bin", &mvp, sizeof(mvp));
    const VkExtent2D extent = vc.getSwapChainExtent();
    writeBytes("headless_extent.bin", &extent, sizeof(extent));
}

TEST(RustReference, dumpRasterLanternRender) {
    if (!std::filesystem::exists(test_scene::kLanternPath)) {
        GTEST_SKIP() << "SPZ scene not found: " << test_scene::kLanternPath;
    }
    const auto scene = test_scene::loadLantern();

    HeadlessFrontend frontend;
    auto& engine = frontend.getKlartraumEngine();
    auto& vc = engine.getVulkanContext();

    const uint32_t numImages = vc.getNumberOfSwapChainImages();
    const VkExtent2D extent = vc.getSwapChainExtent();
    std::vector<VkImageView> views(numImages);
    std::vector<VkImage> images(numImages);
    std::vector<VkExtent2D> extents(numImages, extent);
    for (uint32_t i = 0; i < numImages; ++i) {
        views[i] = vc.getImageView(i);
        images[i] = vc.getSwapChainImage(i);
    }
    auto imageViewSrc = std::make_shared<ImageViewSrc>(views, images, extents);
    for (uint32_t i = 0; i < numImages; ++i) {
        imageViewSrc->setWaitFor(i, vc.imageAvailableSemaphoresPerImage[i]);
    }

    auto cameraUBO = std::make_shared<CameraUboType>();
    InterfaceCameraOrbit orbit(InterfaceCameraOrbit::UpDirection::Y);
    orbit.initialize(vc);
    test_scene::frameLantern(orbit, scene);
    orbit.update(cameraUBO->ubo);

    auto model = std::make_shared<GaussianDataStandard>(vc, scene.gaussians);
    auto splatting = vc.create<VulkanGaussianSplattingRaster>(imageViewSrc, cameraUBO, model->buffers());
    engine.add(splatting);
    for (uint32_t i = 0; i < numImages; ++i) {
        cameraUBO->update(i);
    }

    // Image 0 is rendered by the first and the last of the five frames.
    for (int frame = 0; frame < 5; ++frame) {
        engine.step();
        vkQueueWaitIdle(vc.getGraphicsQueue());
    }

    const auto pixels = readImageToHost(vc, images[0], extent.width, extent.height);
    writeBytes("raster_lantern.bgra", pixels.data(), pixels.size());

    std::filesystem::create_directories(kReferenceDir);
    std::ofstream ppm(kReferenceDir / "raster_lantern.ppm", std::ios::binary);
    ppm << "P6\n" << extent.width << " " << extent.height << "\n255\n";
    for (size_t i = 0; i < pixels.size(); i += 4) {
        const uint8_t rgb[3] = {pixels[i + 2], pixels[i + 1], pixels[i]};
        ppm.write(reinterpret_cast<const char*>(rgb), 3);
    }

    EXPECT_GT(*std::max_element(pixels.begin(), pixels.end()), uint8_t(10));
}
