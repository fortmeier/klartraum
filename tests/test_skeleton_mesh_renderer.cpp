// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - transparentSkeletonDrawsNothing: with opacity 0 the image stays black
 * - boneDrawnBetweenJoints: a horizontal bone covers the screen line between its projected joints, as thick as its
 *   projected diameter, with a sphere at each joint wider than the bone, and nothing beyond
 * - invisibleJointHidesItsBones: a joint with w = 0 removes its sphere and its bones; the rest is drawn
 * - shadedInTheSkeletonColour: pixels keep the hue of the colour; the bone is brightest where it faces the camera
 * - opacityBlendsOverImage: at opacity 0.5 a pixel is half as bright as when opaque
 * - nearerSkeletonCoversFartherOne: two crossing bones in one pass show the nearer one where they cross, whichever is
 *   drawn first
 * - surfaceInFrontHidesSkeleton: with an occlusion depth buffer whose surface covers half of the image, the skeleton
 *   is drawn only on the free half, and fully when the surface lies behind it
 * - splatOccludesSkeleton: with the depth output of either Gaussian splatting backend, a sphere behind an opaque
 *   Gaussian is hidden and one in front of it is drawn
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/hostvalues.hpp"
#include "klartraum/computegraph/renderpass.hpp"
#include "klartraum/gaussian_data_standard.hpp"
#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/skeleton_mesh_renderer.hpp"
#include "klartraum/vulkan_buffer.hpp"

using namespace klartraum;

namespace {

using JointBuffer = BufferElement<VulkanBuffer<glm::vec4>>;
using DepthBuffer = BufferElement<VulkanBuffer<float>>;
using Params = HostValues<SkeletonGaussianParams>;
using Bones = std::vector<std::pair<uint32_t, uint32_t>>;

constexpr VkExtent2D kExtent{128, 128};

struct Rgb {
    uint8_t r, g, b;
    int sum() const { return int(r) + int(g) + int(b); }
};

struct Image {
    uint32_t width = 0, height = 0;
    std::vector<Rgb> pixels;
    const Rgb& at(uint32_t x, uint32_t y) const { return pixels[size_t(y) * width + x]; }
    bool lit(uint32_t x, uint32_t y) const { return at(x, y).sum() > 0; }
    size_t litCount() const {
        return size_t(std::count_if(pixels.begin(), pixels.end(), [](const Rgb& p) { return p.sum() > 0; }));
    }
};

Image readImage(VulkanContext& vc, VkImage image, VkImageLayout layout, VkExtent2D extent) {
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
    const auto* bgra = static_cast<const uint8_t*>(data);
    Image result{extent.width, extent.height, std::vector<Rgb>(size_t(extent.width) * extent.height)};
    for (size_t i = 0; i < result.pixels.size(); ++i) {
        result.pixels[i] = {bgra[4 * i + 2], bgra[4 * i + 1], bgra[4 * i]};
    }
    vkUnmapMemory(vc.getDevice(), memory);
    vkDestroyBuffer(vc.getDevice(), buffer, nullptr);
    vkFreeMemory(vc.getDevice(), memory, nullptr);
    return result;
}

// Camera at z = 3 looking at the origin (image y follows world y), 50 degrees vertical field of view.
CameraMVP perspectiveCamera() {
    CameraMVP camera{glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f), glm::vec4(0.0f, 0.0f, 3.0f, 1.0f)};
    camera.view = glm::lookAt(glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3(0.0f), glm::vec3(0.0f, -1.0f, 0.0f));
    camera.proj = glm::perspective(glm::radians(50.0f), 1.0f, 0.1f, 100.0f);
    return camera;
}

glm::vec2 projectToPixel(const CameraMVP& camera, glm::vec3 position) {
    const glm::vec4 clip = camera.proj * camera.view * camera.model * glm::vec4(position, 1.0f);
    return (glm::vec2(clip) / clip.w * 0.5f + 0.5f) * glm::vec2(kExtent.width, kExtent.height);
}

float ndcDepth(const CameraMVP& camera, glm::vec3 position) {
    const glm::vec4 clip = camera.proj * camera.view * camera.model * glm::vec4(position, 1.0f);
    return clip.z / clip.w;
}

// Pixels per scene unit at distance 3 from the camera.
float pixelsPerUnit() { return 0.5f * float(kExtent.height) / (3.0f * std::tan(glm::radians(25.0f))); }

SkeletonGaussianParams skeletonParams(glm::vec3 colour, float opacity, float radius = 0.05f) {
    SkeletonGaussianParams params;
    params.colourOpacity = glm::vec4(colour, opacity);
    params.shape = glm::vec4(radius, 1.6f, 0.0f, 0.0f);
    return params;
}

// A skeleton to draw: its joints, bones and parameters.
struct Skeleton {
    std::vector<glm::vec4> joints;
    Bones bones;
    SkeletonGaussianParams params;
};

class SkeletonMeshRendererTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
    }

    // Renders the skeletons, in this order, into one cleared target with a depth buffer.
    Image render(const std::vector<Skeleton>& skeletons, const std::vector<float>* occlusionDepth = nullptr) {
        const CameraMVP cameraMatrices = perspectiveCamera();
        auto target = std::make_shared<OffscreenTarget>(*vc, kExtent, 1);
        auto camera = std::make_shared<CameraUboType>();
        camera->ubo = cameraMatrices;
        auto renderPass = std::make_shared<RenderPass>(vc->getSwapChainImageFormat(), kExtent);
        renderPass->setDepthAttachment(std::make_shared<DepthAttachment>(*vc, kExtent, 1));
        renderPass->setInput(target, 0);
        renderPass->setInput(camera, 1);
        std::shared_ptr<DepthBuffer> depthBuffer;
        if (occlusionDepth) {
            depthBuffer = std::make_shared<DepthBuffer>(*vc, uint32_t(occlusionDepth->size()),
                                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            renderPass->addComputeDependency(depthBuffer);
        }
        std::vector<std::shared_ptr<JointBuffer>> jointBuffers;
        for (const auto& skeleton : skeletons) {
            auto joints = std::make_shared<JointBuffer>(*vc, uint32_t(skeleton.joints.size()),
                                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            auto params = std::make_shared<Params>(*vc, std::vector<SkeletonGaussianParams>{skeleton.params});
            auto renderer = std::make_shared<SkeletonMeshRenderer>(joints, skeleton.bones, params, kExtent);
            if (depthBuffer) {
                renderer->setOcclusionDepth(depthBuffer);
            }
            renderPass->addDrawComponent(renderer);
            renderPass->addComputeDependency(joints);
            renderPass->addComputeDependency(params);
            jointBuffers.push_back(joints);
        }
        ComputeGraph graph(*vc, 1);
        graph.compileFrom(renderPass);
        for (size_t i = 0; i < skeletons.size(); ++i) {
            jointBuffers[i]->getBuffer(0).memcopyFrom(skeletons[i].joints);
        }
        if (depthBuffer) {
            depthBuffer->getBuffer(0).memcopyFrom(*occlusionDepth);
        }
        camera->update(0);
        graph.submitAndWait(vc->getGraphicsQueue(), 0);
        return readImage(*vc, target->getImage(0), *target->getFinalLayoutOverride(), kExtent);
    }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
};

// A horizontal bone from x = -0.6 to 0.6 at depth `z`.
Skeleton horizontalBone(glm::vec3 colour = glm::vec3(1.0f, 0.6f, 0.2f), float opacity = 1.0f, float z = 0.0f) {
    return {{{-0.6f, 0.0f, z, 1.0f}, {0.6f, 0.0f, z, 1.0f}}, {{0, 1}}, skeletonParams(colour, opacity)};
}

// Vertical run of lit pixels in column x.
uint32_t litHeight(const Image& image, uint32_t x) {
    uint32_t count = 0;
    for (uint32_t y = 0; y < image.height; ++y) {
        count += image.lit(x, y) ? 1 : 0;
    }
    return count;
}

} // namespace

TEST_F(SkeletonMeshRendererTest, transparentSkeletonDrawsNothing) {
    const Image image = render({horizontalBone(glm::vec3(1.0f), 0.0f)});
    EXPECT_EQ(image.litCount(), 0u);
}

TEST_F(SkeletonMeshRendererTest, boneDrawnBetweenJoints) {
    const Image image = render({horizontalBone()});
    const CameraMVP camera = perspectiveCamera();
    // The ends on screen, left to right (the camera may mirror x).
    const glm::vec2 a = projectToPixel(camera, {-0.6f, 0.0f, 0.0f});
    const glm::vec2 b = projectToPixel(camera, {0.6f, 0.0f, 0.0f});
    const glm::vec2 left = a.x < b.x ? a : b;
    const glm::vec2 right = a.x < b.x ? b : a;
    const uint32_t row = uint32_t(left.y);
    for (uint32_t x = uint32_t(left.x) + 1; x < uint32_t(right.x); ++x) {
        EXPECT_TRUE(image.lit(x, row)) << "x " << x;
    }
    // Bone diameter 0.1, joint sphere diameter 0.16 (shape.y = 1.6).
    const float boneDiameter = 0.1f * pixelsPerUnit();
    EXPECT_NEAR(float(litHeight(image, kExtent.width / 2)), boneDiameter, 2.0f);
    EXPECT_NEAR(float(litHeight(image, uint32_t(left.x))), 1.6f * boneDiameter, 2.0f);
    EXPECT_GT(litHeight(image, uint32_t(left.x)), litHeight(image, kExtent.width / 2));
    // Nothing beyond the spheres.
    const uint32_t beyond = uint32_t(right.x + 0.08f * pixelsPerUnit()) + 2;
    for (uint32_t x = beyond; x < kExtent.width; ++x) {
        EXPECT_EQ(litHeight(image, x), 0u) << "x " << x;
    }
    EXPECT_FALSE(image.lit(kExtent.width / 2, row + uint32_t(boneDiameter)));
}

TEST_F(SkeletonMeshRendererTest, invisibleJointHidesItsBones) {
    // Three joints in an L; joint 2 is invisible, so only bone 0-1 remains.
    Skeleton skeleton{{{-0.6f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.6f, 0.0f, 0.0f}},
                      {{0, 1}, {1, 2}},
                      skeletonParams(glm::vec3(1.0f), 1.0f)};
    const Image image = render({skeleton});
    const CameraMVP camera = perspectiveCamera();
    const glm::vec2 first = projectToPixel(camera, {-0.3f, 0.0f, 0.0f});
    const glm::vec2 hidden = projectToPixel(camera, {0.0f, 0.4f, 0.0f});
    const glm::vec2 hiddenJoint = projectToPixel(camera, {0.0f, 0.6f, 0.0f});
    EXPECT_TRUE(image.lit(uint32_t(first.x), uint32_t(first.y)));
    EXPECT_FALSE(image.lit(uint32_t(hidden.x), uint32_t(hidden.y)));
    EXPECT_FALSE(image.lit(uint32_t(hiddenJoint.x), uint32_t(hiddenJoint.y)));
}

TEST_F(SkeletonMeshRendererTest, shadedInTheSkeletonColour) {
    const Image image = render({horizontalBone(glm::vec3(1.0f, 0.5f, 0.0f))});
    const uint32_t x = kExtent.width / 2;
    const uint32_t centre = uint32_t(projectToPixel(perspectiveCamera(), {0.0f, 0.0f, 0.0f}).y);
    const Rgb middle = image.at(x, centre);
    EXPECT_GT(middle.r, 200);
    EXPECT_NEAR(float(middle.g), 0.5f * float(middle.r), 2.0f);
    EXPECT_EQ(middle.b, 0);
    // Towards the silhouette the surface turns away from the camera and darkens.
    uint32_t edge = centre;
    while (edge + 1 < kExtent.height && image.lit(x, edge + 1)) {
        ++edge;
    }
    EXPECT_LT(image.at(x, edge).r, middle.r);
    EXPECT_GT(image.at(x, edge).r, 0);
}

TEST_F(SkeletonMeshRendererTest, opacityBlendsOverImage) {
    const Image opaque = render({horizontalBone(glm::vec3(1.0f), 1.0f)});
    const Image half = render({horizontalBone(glm::vec3(1.0f), 0.5f)});
    const uint32_t x = kExtent.width / 2;
    const uint32_t y = uint32_t(projectToPixel(perspectiveCamera(), {0.0f, 0.0f, 0.0f}).y);
    EXPECT_NEAR(float(half.at(x, y).r), 0.5f * float(opaque.at(x, y).r), 1.5f);
}

TEST_F(SkeletonMeshRendererTest, nearerSkeletonCoversFartherOne) {
    // A red horizontal bone in front (z = 0.5), a green vertical one behind (z = -0.5).
    const Skeleton nearBone = horizontalBone(glm::vec3(1.0f, 0.0f, 0.0f), 1.0f, 0.5f);
    const Skeleton farBone{{{0.0f, -0.6f, -0.5f, 1.0f}, {0.0f, 0.6f, -0.5f, 1.0f}},
                           {{0, 1}},
                           skeletonParams(glm::vec3(0.0f, 1.0f, 0.0f), 1.0f)};
    const glm::vec2 crossing = projectToPixel(perspectiveCamera(), {0.0f, 0.0f, 0.5f});
    for (bool nearFirst : {true, false}) {
        SCOPED_TRACE(nearFirst ? "near drawn first" : "far drawn first");
        const Image image =
            render(nearFirst ? std::vector<Skeleton>{nearBone, farBone} : std::vector<Skeleton>{farBone, nearBone});
        const Rgb pixel = image.at(uint32_t(crossing.x), uint32_t(crossing.y));
        EXPECT_GT(pixel.r, 0);
        EXPECT_EQ(pixel.g, 0);
        // The far bone still shows above and below the near one.
        const glm::vec2 above = projectToPixel(perspectiveCamera(), {0.0f, 0.4f, -0.5f});
        EXPECT_GT(image.at(uint32_t(above.x), uint32_t(above.y)).g, 0);
    }
}

TEST_F(SkeletonMeshRendererTest, surfaceInFrontHidesSkeleton) {
    const CameraMVP camera = perspectiveCamera();
    // A surface at z = 0.5 (in front of the bone) over the right half of the image.
    std::vector<float> depth(size_t(kExtent.width) * kExtent.height, std::numeric_limits<float>::infinity());
    const float front = ndcDepth(camera, {0.0f, 0.0f, 0.5f});
    for (uint32_t y = 0; y < kExtent.height; ++y) {
        for (uint32_t x = kExtent.width / 2; x < kExtent.width; ++x) {
            depth[size_t(y) * kExtent.width + x] = front;
        }
    }
    const Image cut = render({horizontalBone()}, &depth);
    const uint32_t row = uint32_t(projectToPixel(camera, {0.0f, 0.0f, 0.0f}).y);
    EXPECT_TRUE(cut.lit(kExtent.width / 2 - 4, row));
    for (uint32_t x = kExtent.width / 2; x < kExtent.width; ++x) {
        EXPECT_EQ(litHeight(cut, x), 0u) << "x " << x;
    }

    // The same surface behind the bone hides nothing.
    const float behind = ndcDepth(camera, {0.0f, 0.0f, -0.5f});
    std::replace(depth.begin(), depth.end(), front, behind);
    const Image full = render({horizontalBone()}, &depth);
    EXPECT_EQ(full.litCount(), render({horizontalBone()}).litCount());
}

TEST_F(SkeletonMeshRendererTest, splatOccludesSkeleton) {
    // An opaque Gaussian disc at the origin facing the camera; one joint sphere behind it, one in front of it beside
    // the other's line of sight.
    const CameraMVP cameraMatrices = perspectiveCamera();
    Gaussian3D gaussian{};
    gaussian.position = {0.0f, 0.0f, 0.0f};
    gaussian.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
    gaussian.scale = {0.3f, 0.3f, 0.05f};
    gaussian.color = {1.0f, 1.0f, 1.0f};
    gaussian.alpha = 0.99f;
    const std::vector<glm::vec4> joints = {{0.0f, 0.0f, -0.5f, 1.0f}, {0.15f, 0.0f, 0.5f, 1.0f}};

    for (GsplatBackend backend : {GsplatBackend::Compute, GsplatBackend::Raster}) {
        SCOPED_TRACE(backend == GsplatBackend::Compute ? "compute" : "raster");
        auto target = std::make_shared<OffscreenTarget>(*vc, kExtent, 1);
        auto camera = std::make_shared<CameraUboType>();
        camera->ubo = cameraMatrices;
        GsplatConfig config;
        config.outputDepth = true;
        config.maxMod = 16;
        auto splatting = createGaussianSplatting(
            *vc, backend, target, camera, std::make_shared<GaussianDataStandard>(*vc, std::vector{gaussian}), config);
        auto jointBuffer =
            std::make_shared<JointBuffer>(*vc, uint32_t(joints.size()), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        // Two zero-length bones: just their joint spheres.
        auto params = std::make_shared<Params>(
            *vc, std::vector<SkeletonGaussianParams>{skeletonParams(glm::vec3(1.0f, 0.0f, 0.0f), 1.0f, 0.03f)});
        auto renderer = std::make_shared<SkeletonMeshRenderer>(jointBuffer, Bones{{0, 0}, {1, 1}}, params, kExtent);
        renderer->setOcclusionDepth(getGaussianSplattingDepth(splatting));
        auto skeletonTarget = std::make_shared<OffscreenTarget>(*vc, kExtent, 1);
        auto renderPass = std::make_shared<RenderPass>(vc->getSwapChainImageFormat(), kExtent);
        renderPass->setDepthAttachment(std::make_shared<DepthAttachment>(*vc, kExtent, 1));
        renderPass->setInput(skeletonTarget, 0);
        renderPass->setInput(camera, 1);
        renderPass->addDrawComponent(renderer);
        renderPass->addComputeDependency(jointBuffer);
        renderPass->addComputeDependency(params);
        renderPass->addComputeDependency(splatting);
        ComputeGraph graph(*vc, 1);
        graph.compileFrom(renderPass);
        jointBuffer->getBuffer(0).memcopyFrom(joints);
        camera->update(0);
        graph.submitAndWait(vc->getGraphicsQueue(), 0);
        const Image image =
            readImage(*vc, skeletonTarget->getImage(0), *skeletonTarget->getFinalLayoutOverride(), kExtent);
        const glm::vec2 hidden = projectToPixel(cameraMatrices, glm::vec3(joints[0]));
        const glm::vec2 shown = projectToPixel(cameraMatrices, glm::vec3(joints[1]));
        EXPECT_FALSE(image.lit(uint32_t(hidden.x), uint32_t(hidden.y)));
        EXPECT_TRUE(image.lit(uint32_t(shown.x), uint32_t(shown.y)));
    }
}
