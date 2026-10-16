// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - cropProjectionMapsWindowToView: cropProjection() maps the corners of the window to the corners of the view and
 *   leaves the depth unchanged
 * - croppedRenderMatchesFullRender: rendering with a projection cropped to an off-centre window gives the pixels the
 *   full render has in that window, colour and depth, in both backends
 **/

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "klartraum/camera.hpp"
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/gaussian_data_standard.hpp"
#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/offscreen_target.hpp"

using namespace klartraum;

namespace {

CameraMVP perspectiveCamera() {
    CameraMVP camera{};
    camera.model = glm::mat4(1.0f);
    camera.view = glm::lookAt(glm::vec3(0.3f, 0.5f, 2.0f), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    camera.proj = glm::perspective(glm::radians(50.0f), 1.0f, 0.1f, 100.0f);
    camera.proj[1][1] *= -1.0f;
    camera.cameraWorldPos = glm::vec4(0.3f, 0.5f, 2.0f, 1.0f);
    return camera;
}

// A grid of small, opaque Gaussians around the plane z = 0, elongated along the view direction, so their footprint
// depends on where they lie relative to the camera axis.
std::vector<Gaussian3D> gaussianGrid() {
    std::vector<Gaussian3D> gaussians;
    for (int y = -6; y <= 6; ++y) {
        for (int x = -6; x <= 6; ++x) {
            Gaussian3D gaussian{};
            gaussian.position = {0.1f * float(x), 0.1f * float(y), 0.02f * float((x + y) % 3)};
            const float angle = 0.3f * float(x - y);
            gaussian.rotation = {0.0f, 0.0f, std::sin(angle), std::cos(angle)};
            gaussian.scale = {0.012f, 0.01f, 0.08f};
            gaussian.color = {0.2f * float(x % 4), -0.3f + 0.1f * float(y % 5), 0.5f};
            gaussian.alpha = 0.9f;
            gaussians.push_back(gaussian);
        }
    }
    return gaussians;
}

struct Render {
    std::vector<uint8_t> bgra;
    std::vector<float> depth;
};

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

Render render(VulkanContext& vc, GsplatBackend backend, const CameraMVP& matrices, VkExtent2D extent) {
    auto target = std::make_shared<OffscreenTarget>(vc, extent, 1);
    auto camera = std::make_shared<CameraUboType>();
    camera->ubo = matrices;
    GsplatConfig config;
    config.outputDepth = true;
    config.maxMod = 8;
    auto splatting = createGaussianSplatting(vc, backend, target, camera,
                                             std::make_shared<GaussianDataStandard>(vc, gaussianGrid()), config);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(splatting);
    camera->update(0);
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    Render result;
    result.bgra = readImage(vc, target->getImage(0), *target->getFinalLayoutOverride(), extent);
    result.depth.resize(size_t(extent.width) * extent.height);
    getGaussianSplattingDepth(splatting)->getBuffer(0).memcopyTo(result.depth);
    return result;
}

} // namespace

TEST(GaussianSplatCrop, cropProjectionMapsWindowToView) {
    const glm::mat4 proj = perspectiveCamera().proj;
    const glm::vec2 centre(0.625f, 0.375f);
    const float halfExtent = 0.25f;
    const glm::mat4 cropped = cropProjection(proj, centre, halfExtent);
    for (const glm::vec2 corner : {glm::vec2(-1.0f, -1.0f), glm::vec2(1.0f, 1.0f), glm::vec2(1.0f, -1.0f)}) {
        // A point that the full projection maps to the window corner ...
        const glm::vec2 ndc = centre + corner * halfExtent;
        const glm::vec4 view(ndc.x / proj[0][0] * 2.0f, ndc.y / proj[1][1] * 2.0f, -2.0f, 1.0f);
        const glm::vec4 full = proj * view, crop = cropped * view;
        ASSERT_NEAR(full.x / full.w, ndc.x, 1e-5f);
        // ... the cropped projection maps to the view corner, at the same depth.
        EXPECT_NEAR(crop.x / crop.w, corner.x, 1e-5f);
        EXPECT_NEAR(crop.y / crop.w, corner.y, 1e-5f);
        EXPECT_NEAR(crop.z / crop.w, full.z / full.w, 1e-6f);
    }
}

TEST(GaussianSplatCrop, croppedRenderMatchesFullRender) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    // The window [0.375, 0.875] x [0.125, 0.625] in NDC is pixels [88, 120) x [72, 104) of a 128 pixel image.
    const VkExtent2D fullExtent{128, 128}, cropExtent{32, 32};
    const glm::vec2 centre(0.625f, 0.375f);
    const float halfExtent = 0.25f;
    const uint32_t left = 88, top = 72;

    for (GsplatBackend backend : {GsplatBackend::Compute, GsplatBackend::Raster}) {
        SCOPED_TRACE(backend == GsplatBackend::Compute ? "compute" : "raster");
        CameraMVP matrices = perspectiveCamera();
        const Render full = render(vc, backend, matrices, fullExtent);
        matrices.proj = cropProjection(matrices.proj, centre, halfExtent);
        const Render crop = render(vc, backend, matrices, cropExtent);

        double colourDifference = 0.0;
        size_t covered = 0, depthMismatches = 0;
        for (uint32_t y = 0; y < cropExtent.height; ++y) {
            for (uint32_t x = 0; x < cropExtent.width; ++x) {
                const size_t c = size_t(y) * cropExtent.width + x;
                const size_t f = size_t(y + top) * fullExtent.width + (x + left);
                for (int channel = 0; channel < 3; ++channel) {
                    colourDifference += std::abs(int(crop.bgra[4 * c + channel]) - int(full.bgra[4 * f + channel]));
                }
                covered += (full.bgra[4 * f] | full.bgra[4 * f + 1] | full.bgra[4 * f + 2]) != 0 ? 1 : 0;
                const bool same = std::isfinite(crop.depth[c]) == std::isfinite(full.depth[f]) &&
                                  (!std::isfinite(full.depth[f]) || std::abs(crop.depth[c] - full.depth[f]) < 1e-5f);
                depthMismatches += same ? 0 : 1;
            }
        }
        ASSERT_GT(covered, 300u) << "the window must show Gaussians";
        const double meanDifference = colourDifference / (3.0 * cropExtent.width * cropExtent.height);
        std::cout << "mean colour difference " << meanDifference << ", depth mismatches " << depthMismatches
                  << std::endl;
        EXPECT_LT(meanDifference, 2.0);
        EXPECT_LT(depthMismatches, 20u);
    }
}
