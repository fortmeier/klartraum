// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - offscreenTargetClearsToBlackByDefault: running a graph with an OffscreenTarget clears its
 *   image to opaque black
 * - clearCanBeDisabled: with setClear(false) the target keeps what it held
 * - clearImageUsesItsColor: a ClearImage over a target that does not clear itself fills it with
 *   its color and stands for the target's images
 * - backendsRenderOverTheirTarget: both Gaussian-splatting backends draw a block of Gaussians over a blue
 *   background: pixels the scene leaves black over black stay blue, covered pixels keep
 *   their red and green and gain blue, and the backends agree as they do over black
 * - compositeDrawsOntoTarget: ImageComposite replaces a blue target with a red source (stretched), draws
 *   a tall source in the middle and keeps the target beside it (fit), covers the whole target with
 *   a wide source (fill), and blends a half-transparent source over the target (over)
 * - compositeFitsPatternWithoutDistortion: a tall patterned source is centred at its original size,
 *   keeping every source pixel and leaving the target beside it
 * - compositeFillsByCroppingCenter: a wide patterned source is cropped symmetrically to cover the target
 * - compositeInterpolatesOpaquePattern: stretching an opaque pattern uses bilinear interpolation
 * - singlePathImageIsReadByEveryPath: a SinglePathImage filled with copyFrom() hands the same image
 *   to every path; a three-path graph copies it into each path's target, run through an
 *   ImageViewForward that stands for the copy's output
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <vector>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/imageresample.hpp"
#include "klartraum/computegraph/imagecomposite.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/gaussian_data_standard.hpp"
#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/interface_camera_orbit.hpp"
#include "klartraum/offscreen_target.hpp"

using namespace klartraum;

namespace {

// The image's pixels as BGRA bytes (the headless swapchain format).
std::vector<uint8_t> readImage(VulkanContext& vc, VkImage image, VkImageLayout layout, VkExtent2D extent) {
    const VkDeviceSize bytes = VkDeviceSize{extent.width} * extent.height * 4;
    VkBuffer buffer;
    VkDeviceMemory memory;
    vc.createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, buffer, memory);
    vc.submitImmediate([&](VkCommandBuffer commandBuffer) {
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {extent.width, extent.height, 1};
        vkCmdCopyImageToBuffer(commandBuffer, image, layout, buffer, 1, &region);
    });
    void* data;
    vkMapMemory(vc.getDevice(), memory, 0, bytes, 0, &data);
    std::vector<uint8_t> pixels(static_cast<uint8_t*>(data), static_cast<uint8_t*>(data) + bytes);
    vkUnmapMemory(vc.getDevice(), memory);
    vkDestroyBuffer(vc.getDevice(), buffer, nullptr);
    vkFreeMemory(vc.getDevice(), memory, nullptr);
    return pixels;
}

void fill(VulkanContext& vc, VkImage image, VkClearColorValue color) {
    vc.submitImmediate([&](VkCommandBuffer commandBuffer) { recordClearImage(commandBuffer, image, color); });
}

// Uploads BGRA pixels to a GENERAL image, ready for the compositor to read.
void uploadImage(VulkanContext& vc, VkImage image, VkExtent2D extent, const std::vector<uint8_t>& pixels) {
    VulkanBuffer<uint8_t> staging(vc, static_cast<uint32_t>(pixels.size()));
    staging.memcopyFrom(pixels);
    vc.submitImmediate([&](VkCommandBuffer commandBuffer) {
        VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        barrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &barrier);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {extent.width, extent.height, 1};
        vkCmdCopyBufferToImage(commandBuffer, staging.getBuffer(), image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                               &copy);
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,
                             nullptr, 0, nullptr, 1, &barrier);
    });
}

std::array<uint8_t, 4> pixelAt(const std::vector<uint8_t>& pixels, uint32_t width, uint32_t x, uint32_t y) {
    const size_t i = (size_t(y) * width + x) * 4;
    return {pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]};
}

// Every pixel has a distinct opaque color, so cropping or distortion is visible.
std::vector<uint8_t> opaquePattern(VkExtent2D extent) {
    std::vector<uint8_t> pixels;
    for (uint32_t y = 0; y < extent.height; ++y) {
        for (uint32_t x = 0; x < extent.width; ++x) {
            pixels.insert(pixels.end(),
                          {50, static_cast<uint8_t>(20 + y * 20), static_cast<uint8_t>(10 + x * 30), 255});
        }
    }
    return pixels;
}

std::vector<uint8_t> compositePattern(VulkanContext& vc, VkExtent2D sourceExtent, VkExtent2D targetExtent,
                                      const std::vector<uint8_t>& pixels, CompositeFit fit) {
    auto source = std::make_shared<OffscreenTarget>(vc, sourceExtent, 1);
    source->setClear(false);
    uploadImage(vc, source->getImage(0), sourceExtent, pixels);
    auto target = std::make_shared<OffscreenTarget>(vc, targetExtent, 1);
    target->setClearColor({{0.0f, 0.0f, 1.0f, 1.0f}});
    auto draw = std::make_shared<ImageComposite>(vc, sourceExtent, targetExtent, CompositeMode::Replace, fit);
    draw->setSource(source);
    draw->setTarget(target);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(draw);
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    return readImage(vc, target->getImage(0), VK_IMAGE_LAYOUT_GENERAL, targetExtent);
}

// Whether every pixel is (b, g, r, a).
bool allPixels(const std::vector<uint8_t>& pixels, uint8_t b, uint8_t g, uint8_t r, uint8_t a) {
    for (size_t i = 0; i < pixels.size(); i += 4) {
        if (pixels[i] != b || pixels[i + 1] != g || pixels[i + 2] != r || pixels[i + 3] != a) {
            return false;
        }
    }
    return true;
}

class RenderOverTargetsTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
    }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
};

constexpr VkExtent2D kSmall{16, 16};
constexpr VkClearColorValue kWhite{{1.0f, 1.0f, 1.0f, 1.0f}};
constexpr VkClearColorValue kRed{{1.0f, 0.0f, 0.0f, 1.0f}};
constexpr VkClearColorValue kBlue{{0.0f, 0.0f, 1.0f, 1.0f}};

} // namespace

TEST_F(RenderOverTargetsTest, offscreenTargetClearsToBlackByDefault) {
    auto target = std::make_shared<OffscreenTarget>(*vc, kSmall, 1);
    EXPECT_TRUE(target->clears());
    fill(*vc, target->getImage(0), kWhite);
    ComputeGraph graph(*vc, 1);
    graph.compileFrom(target);
    graph.submitAndWait(vc->getGraphicsQueue(), 0);
    EXPECT_TRUE(allPixels(readImage(*vc, target->getImage(0), VK_IMAGE_LAYOUT_GENERAL, kSmall), 0, 0, 0, 255));
}

TEST_F(RenderOverTargetsTest, clearCanBeDisabled) {
    auto target = std::make_shared<OffscreenTarget>(*vc, kSmall, 1);
    target->setClear(false);
    fill(*vc, target->getImage(0), kWhite);
    ComputeGraph graph(*vc, 1);
    graph.compileFrom(target);
    graph.submitAndWait(vc->getGraphicsQueue(), 0);
    EXPECT_TRUE(allPixels(readImage(*vc, target->getImage(0), VK_IMAGE_LAYOUT_GENERAL, kSmall), 255, 255, 255, 255));
}

TEST_F(RenderOverTargetsTest, clearImageUsesItsColor) {
    auto target = std::make_shared<OffscreenTarget>(*vc, kSmall, 1);
    target->setClear(false);
    auto clear = std::make_shared<ClearImage>(kRed);
    clear->setInput(target, 0);
    EXPECT_EQ(clear->getImage(0), target->getImage(0));
    EXPECT_EQ(clear->getImageExtent(0).width, kSmall.width);
    EXPECT_EQ(clear->getFinalLayoutOverride(), target->getFinalLayoutOverride());
    ComputeGraph graph(*vc, 1);
    graph.compileFrom(clear);
    graph.submitAndWait(vc->getGraphicsQueue(), 0);
    EXPECT_TRUE(allPixels(readImage(*vc, target->getImage(0), VK_IMAGE_LAYOUT_GENERAL, kSmall), 0, 0, 255, 255));
}

TEST_F(RenderOverTargetsTest, backendsRenderOverTheirTarget) {
    // Reddish, nearly opaque Gaussians where the camera looks (the orbit
    // camera looks at minus its position). As loaded from files, alpha is an
    // opacity and the scale is linear.
    const std::array<float, 3> center{0.5f, 0.0f, -0.5f};
    Gaussian3D gaussian{};
    gaussian.position = center;
    gaussian.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
    gaussian.scale = {0.01f, 0.01f, 0.01f};
    const float c0 = 0.28209479f; // SH band 0: color = 0.5 + c0 * sh
    gaussian.color = {0.5f / c0, -0.5f / c0, -0.5f / c0};
    gaussian.alpha = 0.95f;
    // A 10x10x10 block of them, 0.1 units wide.
    std::vector<Gaussian3D> clump;
    for (int i = 0; i < 1000; ++i) {
        Gaussian3D g = gaussian;
        g.position = {center[0] + 0.01f * static_cast<float>(i % 10) - 0.05f,
                      center[1] + 0.01f * static_cast<float>(i / 10 % 10) - 0.05f,
                      center[2] + 0.01f * static_cast<float>(i / 100) - 0.05f};
        clump.push_back(g);
    }
    auto model = std::make_shared<GaussianDataStandard>(*vc, std::move(clump));
    const VkExtent2D extent{128, 128};
    auto render = [&](GsplatBackend backend, const VkClearColorValue& background) {
        auto target = std::make_shared<OffscreenTarget>(*vc, extent, 1);
        target->setClear(false);
        auto clear = std::make_shared<ClearImage>(background);
        clear->setInput(target, 0);
        auto camera = std::make_shared<CameraUboType>();
        InterfaceCameraOrbit orbit(InterfaceCameraOrbit::UpDirection::Y);
        orbit.initialize(*vc);
        orbit.setAzimuth(0.9f);
        orbit.setElevation(-0.5f);
        orbit.setPosition({-center[0], -center[1], -center[2]});
        orbit.setDistance(1.0f);
        orbit.setProjectionAspectRatio(1.0f);
        orbit.update(camera->ubo);
        auto splatting = createGaussianSplatting(*vc, backend, clear, camera, model);
        ComputeGraph graph(*vc, 1);
        graph.compileFrom(splatting);
        camera->update(0);
        graph.submitAndWait(vc->getGraphicsQueue(), 0);
        return readImage(*vc, target->getImage(0), *target->getFinalLayoutOverride(), extent);
    };

    std::vector<std::vector<uint8_t>> overBlue;
    for (GsplatBackend backend : {GsplatBackend::Compute, GsplatBackend::Raster}) {
        SCOPED_TRACE(backend == GsplatBackend::Compute ? "compute" : "raster");
        const auto black = render(backend, {{0.0f, 0.0f, 0.0f, 1.0f}});
        const auto blue = render(backend, kBlue);
        size_t empty = 0, covered = 0;
        for (size_t i = 0; i < black.size(); i += 4) {
            if (black[i] == 0 && black[i + 1] == 0 && black[i + 2] == 0) {
                // Nothing visible drawn here: the background shows (faint
                // splats may still dim it by a step or two).
                ++empty;
                EXPECT_GE(blue[i], 253);
                EXPECT_EQ(blue[i + 1], 0);
                EXPECT_EQ(blue[i + 2], 0);
            } else {
                // Drawn over: red and green are the splats', blue gains the
                // part of the background that shows through.
                ++covered;
                EXPECT_NEAR(blue[i + 1], black[i + 1], 1);
                EXPECT_NEAR(blue[i + 2], black[i + 2], 1);
                EXPECT_GE(blue[i] + 1, black[i]);
            }
            EXPECT_EQ(blue[i + 3], 255);
        }
        EXPECT_GT(empty, black.size() / 4 / 10) << "the scene should leave some of the image free";
        EXPECT_GT(covered, 100u) << "the Gaussians should cover some of the image";
        overBlue.push_back(blue);
    }
    double difference = 0.0;
    for (size_t i = 0; i < overBlue[0].size(); ++i) {
        difference += std::abs(int(overBlue[0][i]) - int(overBlue[1][i]));
    }
    // As bothBackendsAgreeOnLanternScene, over a colored background.
    EXPECT_LT(difference / overBlue[0].size(), 20.0) << "mean absolute channel difference between the backends";
}

TEST_F(RenderOverTargetsTest, singlePathImageIsReadByEveryPath) {
    auto source = std::make_shared<OffscreenTarget>(*vc, kSmall, 1);
    fill(*vc, source->getImage(0), kRed);
    auto shared = std::make_shared<SinglePathImage>(*vc, kSmall);
    EXPECT_FALSE(shared->clears());
    shared->copyFrom(source->getImage(0), VK_IMAGE_LAYOUT_GENERAL, kSmall);
    EXPECT_EQ(shared->getImage(0), shared->getImage(2));
    EXPECT_THROW(shared->copyFrom(source->getImage(0), VK_IMAGE_LAYOUT_GENERAL, {8, 8}), std::invalid_argument);

    constexpr uint32_t paths = 3;
    auto targets = std::make_shared<OffscreenTarget>(*vc, kSmall, paths);
    auto copy = std::make_shared<ImageResample>(*vc, kSmall, kSmall, ResampleFilter::Nearest);
    copy->setInput(shared, 0);
    copy->setInput(targets, 1);
    // What uses the copied images depends on the copy through a forward.
    auto copied = std::make_shared<ImageViewForward>();
    copied->setInput(copy, 0, 1);
    EXPECT_EQ(copied->getImage(1), targets->getImage(1));
    EXPECT_FALSE(copied->clears());
    ComputeGraph graph(*vc, paths);
    graph.compileFrom(copied);
    for (const auto color : {kRed, kWhite, kBlue}) {
        // Replace the shared contents only after all previous readers finished.
        fill(*vc, source->getImage(0), color);
        shared->copyFrom(source->getImage(0), VK_IMAGE_LAYOUT_GENERAL, kSmall);
        const auto expected = readImage(*vc, source->getImage(0), VK_IMAGE_LAYOUT_GENERAL, kSmall);
        EXPECT_EQ(readImage(*vc, shared->getImage(0), VK_IMAGE_LAYOUT_GENERAL, kSmall), expected);
        for (uint32_t path = 0; path < paths; ++path) {
            graph.submitAndWait(vc->getGraphicsQueue(), path);
            EXPECT_EQ(readImage(*vc, targets->getImage(path), VK_IMAGE_LAYOUT_GENERAL, kSmall), expected)
                << "path " << path;
        }
        EXPECT_EQ(readImage(*vc, shared->getImage(0), VK_IMAGE_LAYOUT_GENERAL, kSmall), expected);
    }
}

TEST_F(RenderOverTargetsTest, compositeFitsPatternWithoutDistortion) {
    constexpr VkExtent2D sourceExtent{4, 8}, targetExtent{8, 8};
    const auto pattern = opaquePattern(sourceExtent);
    const auto result = compositePattern(*vc, sourceExtent, targetExtent, pattern, CompositeFit::Fit);
    const std::array<uint8_t, 4> blue{255, 0, 0, 255};
    for (uint32_t y = 0; y < targetExtent.height; ++y) {
        for (uint32_t x = 0; x < targetExtent.width; ++x) {
            const auto expected = x >= 2 && x < 6 ? pixelAt(pattern, sourceExtent.width, x - 2, y) : blue;
            EXPECT_EQ(pixelAt(result, targetExtent.width, x, y), expected) << x << "," << y;
        }
    }
}

TEST_F(RenderOverTargetsTest, compositeFillsByCroppingCenter) {
    constexpr VkExtent2D sourceExtent{8, 4}, targetExtent{4, 4};
    const auto pattern = opaquePattern(sourceExtent);
    const auto result = compositePattern(*vc, sourceExtent, targetExtent, pattern, CompositeFit::Fill);
    for (uint32_t y = 0; y < targetExtent.height; ++y) {
        for (uint32_t x = 0; x < targetExtent.width; ++x) {
            EXPECT_EQ(pixelAt(result, targetExtent.width, x, y), pixelAt(pattern, sourceExtent.width, x + 2, y))
                << x << "," << y;
        }
    }
}

TEST_F(RenderOverTargetsTest, compositeInterpolatesOpaquePattern) {
    // BGRA corners: red, green, blue, white.
    const std::vector<uint8_t> pattern{0, 0, 255, 255, 0, 255, 0, 255, 255, 0, 0, 255, 255, 255, 255, 255};
    const auto result = compositePattern(*vc, {2, 2}, {4, 4}, pattern, CompositeFit::Stretch);
    EXPECT_EQ(pixelAt(result, 4, 0, 0), pixelAt(pattern, 2, 0, 0));
    EXPECT_EQ(pixelAt(result, 4, 3, 3), pixelAt(pattern, 2, 1, 1));
    const auto upper = pixelAt(result, 4, 1, 1), lower = pixelAt(result, 4, 2, 2);
    const std::array<int, 4> expectedUpper{64, 64, 159, 255}, expectedLower{191, 191, 159, 255};
    for (size_t channel = 0; channel < 4; ++channel) {
        EXPECT_NEAR(upper[channel], expectedUpper[channel], 1);
        EXPECT_NEAR(lower[channel], expectedLower[channel], 1);
    }
}

TEST_F(RenderOverTargetsTest, compositeDrawsOntoTarget) {
    // Draws a `color` source of `sourceExtent` onto an 8x8 blue target.
    auto composite = [&](VkExtent2D sourceExtent, VkClearColorValue color, CompositeMode mode, CompositeFit fit) {
        auto source = std::make_shared<OffscreenTarget>(*vc, sourceExtent, 1);
        fill(*vc, source->getImage(0), color);
        source->setClear(false);
        auto target = std::make_shared<OffscreenTarget>(*vc, VkExtent2D{8, 8}, 1);
        target->setClear(false);
        auto blue = std::make_shared<ClearImage>(kBlue);
        blue->setInput(target, 0);
        auto drawn = std::make_shared<ImageComposite>(*vc, sourceExtent, VkExtent2D{8, 8}, mode, fit);
        drawn->setSource(source);
        drawn->setTarget(blue);
        EXPECT_EQ(drawn->getImage(0), target->getImage(0));
        ComputeGraph graph(*vc, 1);
        graph.compileFrom(drawn);
        graph.submitAndWait(vc->getGraphicsQueue(), 0);
        return readImage(*vc, target->getImage(0), VK_IMAGE_LAYOUT_GENERAL, {8, 8});
    };
    auto pixel = [](const std::vector<uint8_t>& pixels, int x, int y) {
        const size_t i = (static_cast<size_t>(y) * 8 + x) * 4;
        return std::array<int, 4>{pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]}; // BGRA
    };
    const std::array<int, 4> red{0, 0, 255, 255}, blue{255, 0, 0, 255};

    EXPECT_TRUE(allPixels(composite({4, 4}, kRed, CompositeMode::Replace, CompositeFit::Stretch), 0, 0, 255, 255));

    // A 4x8 source fits as 4x8 in the middle: columns 2..5.
    const auto fitted = composite({4, 8}, kRed, CompositeMode::Replace, CompositeFit::Fit);
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            EXPECT_EQ(pixel(fitted, x, y), (x >= 2 && x < 6 ? red : blue)) << x << "," << y;
        }
    }

    // A 8x4 source filling the target is scaled to 16x8 and cropped.
    EXPECT_TRUE(allPixels(composite({8, 4}, kRed, CompositeMode::Replace, CompositeFit::Fill), 0, 0, 255, 255));

    // Half-transparent green over blue.
    const auto over = composite({8, 8}, {{0.0f, 1.0f, 0.0f, 0.5f}}, CompositeMode::Over, CompositeFit::Stretch);
    const auto p = pixel(over, 3, 3);
    EXPECT_NEAR(p[0], 128, 2); // blue
    EXPECT_NEAR(p[1], 128, 2); // green
    EXPECT_EQ(p[2], 0);
    EXPECT_EQ(p[3], 255);
}
