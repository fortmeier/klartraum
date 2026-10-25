// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - noDepthUnlessRequested: without GsplatConfig::outputDepth neither backend has a depth buffer
 * - depthLeavesColorUnchanged: requesting depth does not change the colour image of either backend
 * - emptyPixelsAreInfinite: where no Gaussian covers a pixel, both backends report +inf
 * - opaqueGaussianHasItsProjectedDepth: the pixel at an opaque Gaussian's centre holds the NDC depth of its centre,
 *   projected with the same camera on the CPU, in both backends
 * - nearerOpaqueGaussianWins: of two opaque Gaussians on the same line of sight, the nearer one's depth is reported,
 *   in both backends
 * - translucentStackDepthDiffersByBackend: behind three Gaussians of opacity 0.32 the compute backend reports the
 *   depth where transmittance first falls to 0.5 (the second Gaussian), the raster backend no surface, as no single
 *   Gaussian is opaque enough
 * - backendsAgreeOnLantern: on the lantern scene both backends find a surface at nearly the same pixels and depths
 * - depthMethodsAreSelectableOnBothBackends: both backends provide either selected surface estimate
 * - rasterDepthPrepassHonorsOpacityCulling: accumulated-opacity depth excludes splats culled from raster RGB
 * - rasterDepthPrepassHonorsClipRange: accumulated-opacity depth excludes depth-clipped raster splats
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <vector>

#include <glm/glm.hpp>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/rendering/gaussian_data_standard.hpp"
#include "klartraum/rendering/gaussian_splatting_factory.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/interface_camera_orbit.hpp"
#include "klartraum/offscreen_target.hpp"

using namespace klartraum;

namespace {

constexpr VkExtent2D kExtent{64, 64};
constexpr float kInfinity = std::numeric_limits<float>::infinity();

struct Render {
    std::vector<uint8_t> color;
    std::vector<float> depth; // empty without a depth buffer
    bool hasDepthBuffer = false;
};

Gaussian3D gaussianAt(glm::vec3 position, float alpha, float scale = 0.02f) {
    Gaussian3D gaussian{};
    gaussian.position = {position.x, position.y, position.z};
    gaussian.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
    gaussian.scale = {scale, scale, scale};
    gaussian.color = {1.0f, 0.5f, -0.5f};
    gaussian.alpha = alpha;
    return gaussian;
}

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

class GaussianSplatDepthTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
    }

    // A camera at distance 1 looking at the origin along -z (azimuth chosen so the view axis is z).
    CameraMVP camera(VkExtent2D extent) {
        InterfaceCameraOrbit orbit(InterfaceCameraOrbit::UpDirection::Y);
        orbit.initialize(*vc);
        orbit.setAzimuth(0.5f * 3.14159265f);
        orbit.setElevation(0.0f);
        orbit.setDistance(1.0f);
        orbit.setProjectionAspectRatio(float(extent.width) / float(extent.height));
        CameraMVP result{};
        orbit.update(result);
        return result;
    }

    // NDC depth and pixel of a world position.
    std::pair<float, glm::ivec2> project(const CameraMVP& matrices, glm::vec3 position, VkExtent2D extent) {
        const glm::vec4 clip = matrices.proj * matrices.view * matrices.model * glm::vec4(position, 1.0f);
        const glm::vec3 ndc = glm::vec3(clip) / clip.w;
        return {ndc.z, glm::ivec2(int((ndc.x * 0.5f + 0.5f) * float(extent.width)),
                                  int((ndc.y * 0.5f + 0.5f) * float(extent.height)))};
    }

    Render render(GsplatBackend backend, std::vector<Gaussian3D> gaussians, bool outputDepth,
                  VkExtent2D extent = kExtent, GaussianDepthMode mode = GaussianDepthMode::BackendDefault,
                  float alphaCull = 1.0f / 255.0f) {
        auto target = std::make_shared<OffscreenTarget>(*vc, extent, 1);
        auto cameraUbo = std::make_shared<CameraUboType>();
        cameraUbo->ubo = camera(extent);
        auto model = std::make_shared<GaussianDataStandard>(*vc, std::move(gaussians));
        GsplatConfig config;
        config.outputDepth = outputDepth;
        config.depthMode = mode;
        config.alphaCullThreshold = alphaCull;
        // The compute backend's binned buffer holds maxMod entries per Gaussian on average; a scene of a few small
        // Gaussians that each overlap up to four bins needs more.
        config.maxMod = 16;
        auto splatting = createGaussianSplatting(*vc, backend, target, cameraUbo, model, config);
        ComputeGraph graph(*vc, 1);
        graph.compileFrom(splatting);
        cameraUbo->update(0);
        graph.submitAndWait(vc->getGraphicsQueue(), 0);

        Render result;
        result.color = readImage(*vc, target->getImage(0), *target->getFinalLayoutOverride(), extent);
        auto depth = splatting->getDepth();
        result.hasDepthBuffer = depth != nullptr;
        if (depth) {
            result.depth.resize(size_t(extent.width) * extent.height);
            depth->getBuffer(0).memcopyTo(result.depth);
        }
        return result;
    }

    float depthAt(const Render& render, glm::ivec2 pixel, VkExtent2D extent = kExtent) {
        return render.depth.at(size_t(pixel.y) * extent.width + size_t(pixel.x));
    }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
};

const GsplatBackend kBackends[] = {GsplatBackend::Compute, GsplatBackend::Raster};
const char* name(GsplatBackend backend) { return backend == GsplatBackend::Compute ? "compute" : "raster"; }

} // namespace

TEST_F(GaussianSplatDepthTest, noDepthUnlessRequested) {
    for (GsplatBackend backend : kBackends) {
        EXPECT_FALSE(render(backend, {gaussianAt({0, 0, 0}, 0.99f)}, false).hasDepthBuffer) << name(backend);
    }
}

TEST_F(GaussianSplatDepthTest, depthLeavesColorUnchanged) {
    const std::vector<Gaussian3D> scene = {gaussianAt({0, 0, 0}, 0.9f), gaussianAt({0.02f, 0.01f, -0.1f}, 0.6f),
                                           gaussianAt({-0.03f, 0.0f, 0.05f}, 0.3f)};
    for (GsplatBackend backend : kBackends) {
        EXPECT_EQ(render(backend, scene, false).color, render(backend, scene, true).color) << name(backend);
    }
}

TEST_F(GaussianSplatDepthTest, emptyPixelsAreInfinite) {
    for (GsplatBackend backend : kBackends) {
        const auto result = render(backend, {gaussianAt({0, 0, 0}, 0.99f)}, true);
        ASSERT_TRUE(result.hasDepthBuffer) << name(backend);
        EXPECT_EQ(depthAt(result, {0, 0}), kInfinity) << name(backend);
        EXPECT_EQ(depthAt(result, {63, 63}), kInfinity) << name(backend);
    }
}

TEST_F(GaussianSplatDepthTest, opaqueGaussianHasItsProjectedDepth) {
    const glm::vec3 position(0.05f, -0.03f, 0.1f);
    const auto [expected, pixel] = project(camera(kExtent), position, kExtent);
    for (GsplatBackend backend : kBackends) {
        const auto result = render(backend, {gaussianAt(position, 0.99f)}, true);
        EXPECT_NEAR(depthAt(result, pixel), expected, 1e-5f) << name(backend);
    }
}

TEST_F(GaussianSplatDepthTest, nearerOpaqueGaussianWins) {
    // Two Gaussians on the view axis; the one with the smaller NDC depth is nearer. The farther one is larger, so it
    // surrounds the nearer one on screen.
    glm::vec3 nearer(0.0f, 0.0f, 0.2f), farther(0.0f, 0.0f, -0.2f);
    if (project(camera(kExtent), nearer, kExtent).first > project(camera(kExtent), farther, kExtent).first) {
        std::swap(nearer, farther);
    }
    const auto [expected, pixel] = project(camera(kExtent), nearer, kExtent);
    for (GsplatBackend backend : kBackends) {
        const auto result = render(backend, {gaussianAt(farther, 0.99f, 0.05f), gaussianAt(nearer, 0.99f)}, true);
        EXPECT_NEAR(depthAt(result, pixel), expected, 1e-5f) << name(backend);
    }
}

TEST_F(GaussianSplatDepthTest, translucentStackDepthDiffersByBackend) {
    const glm::vec3 first(0.0f, 0.0f, 0.2f), second(0.0f, 0.0f, 0.0f), third(0.0f, 0.0f, -0.2f);
    const std::vector<Gaussian3D> stack = {gaussianAt(first, 0.32f, 0.04f), gaussianAt(second, 0.32f, 0.04f),
                                           gaussianAt(third, 0.32f, 0.04f)};
    const auto [expected, pixel] = project(camera(kExtent), second, kExtent);
    EXPECT_NEAR(depthAt(render(GsplatBackend::Compute, stack, true), pixel), expected, 1e-5f);
    EXPECT_EQ(depthAt(render(GsplatBackend::Raster, stack, true), pixel), kInfinity);
}

TEST_F(GaussianSplatDepthTest, depthMethodsAreSelectableOnBothBackends) {
    const std::vector<Gaussian3D> stack = {gaussianAt({0, 0, 0.2f}, 0.32f, 0.04f), gaussianAt({0, 0, 0}, 0.32f, 0.04f),
                                           gaussianAt({0, 0, -0.2f}, 0.32f, 0.04f)};
    const auto [expected, pixel] = project(camera(kExtent), {0, 0, 0}, kExtent);
    for (auto backend : kBackends) {
        EXPECT_NEAR(depthAt(render(backend, stack, true, kExtent, GaussianDepthMode::AccumulatedOpacity), pixel),
                    expected, 1e-5f);
        EXPECT_EQ(depthAt(render(backend, stack, true, kExtent, GaussianDepthMode::NearestOpaqueSplat), pixel),
                  kInfinity);
    }
}

TEST_F(GaussianSplatDepthTest, rasterDepthPrepassHonorsOpacityCulling) {
    const std::vector<Gaussian3D> stack = {gaussianAt({0, 0, 0.2f}, 0.32f, 0.04f), gaussianAt({0, 0, 0}, 0.32f, 0.04f),
                                           gaussianAt({0, 0, -0.2f}, 0.32f, 0.04f)};
    const auto pixel = project(camera(kExtent), {0, 0, 0}, kExtent).second;
    const auto result =
        render(GsplatBackend::Raster, stack, true, kExtent, GaussianDepthMode::AccumulatedOpacity, 0.4f);
    EXPECT_EQ(depthAt(result, pixel), kInfinity);
}

TEST_F(GaussianSplatDepthTest, rasterDepthPrepassHonorsClipRange) {
    const glm::vec3 position(0, 0, 0.85f);
    const auto [z, pixel] = project(camera(kExtent), position, kExtent);
    ASSERT_LT(z, 0.0f);
    for (auto mode : {GaussianDepthMode::AccumulatedOpacity, GaussianDepthMode::NearestOpaqueSplat}) {
        const auto result = render(GsplatBackend::Raster, {gaussianAt(position, 0.99f, 0.001f)}, true, kExtent, mode);
        EXPECT_EQ(depthAt(result, pixel), kInfinity);
    }
}

TEST_F(GaussianSplatDepthTest, backendsAgreeOnLantern) {
    if (!std::filesystem::exists("data/lantern.spz")) {
        GTEST_SKIP() << "Missing data/lantern.spz";
    }
    const VkExtent2D extent{128, 128};
    const auto gaussians = loadGaussiansSpz("data/lantern.spz", true);
    const auto compute = render(GsplatBackend::Compute, gaussians, true, extent);
    const auto raster = render(GsplatBackend::Raster, gaussians, true, extent);

    // Compare linear depths: distance = proj[3][2] / (ndc + proj[2][2]).
    const glm::mat4 proj = camera(extent).proj;
    auto linear = [&](float ndc) { return proj[3][2] / (ndc + proj[2][2]); };
    size_t both = 0, onlyOne = 0;
    std::vector<float> relative;
    for (size_t i = 0; i < compute.depth.size(); ++i) {
        const bool a = std::isfinite(compute.depth[i]), b = std::isfinite(raster.depth[i]);
        if (a && b) {
            ++both;
            relative.push_back(std::abs(linear(compute.depth[i]) - linear(raster.depth[i])) / linear(compute.depth[i]));
        } else if (a != b) {
            ++onlyOne;
        }
    }
    ASSERT_GT(both, 1000u);
    std::nth_element(relative.begin(), relative.begin() + relative.size() / 2, relative.end());
    const float median = relative[relative.size() / 2];
    std::cout << "lantern depth: " << both << " pixels in both, " << onlyOne << " in only one; median relative "
              << "difference " << median << std::endl;
    EXPECT_LT(median, 0.01f);
    EXPECT_LT(double(onlyOne), 0.15 * double(both));
}
