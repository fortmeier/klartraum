// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - segmentsCoverTheOpenPoseLimbs: the mask's body segments include every OpenPose limb and have positive radii
 * - emptySkeletonLeavesBlackMask: a skeleton whose joints are all invisible masks nothing
 * - boneCapsuleHasProjectedRadius: a single bone is masked as a capsule whose radius in pixels is its radius plus the
 *   margin, projected with the camera at the bone's distance
 * - headIsDiscBetweenEars: the head is a disc around the middle of the ears, with the head radius plus the margin,
 *   not a capsule reaching past the ears
 * - fartherBoneIsThinner: the same bone twice as far away is masked about half as wide
 * - occludedPartsAreCut: with an occlusion depth buffer, the mask keeps exactly the pixels of a bone outside a surface
 *   in front of it
 * - maskCoversSkeletonDrawing: for poses of the character controller, every pixel of the OpenPose drawing lies inside
 *   the character mask
 * - sceneMaskMarksSurfaces: the scene mask is white where the depth buffer holds a surface and black where it holds
 *   +inf
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "klartraum/character_controller.hpp"
#include "klartraum/character_mask.hpp"
#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/buffertographicsbarrier.hpp"
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/renderpass.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/openpose_skeleton_renderer.hpp"

using namespace klartraum;

namespace {

constexpr VkExtent2D kExtent{128, 128};
using Vec4Buffer = BufferElement<VulkanBuffer<glm::vec4>>;
using FloatBuffer = BufferElement<VulkanBuffer<float>>;

// Per pixel the largest channel of a BGRA image: 255 where a mask is set, non-zero where a pose map is drawn.
struct Mask {
    uint32_t width = 0, height = 0;
    std::vector<uint8_t> values;
    uint8_t at(int x, int y) const { return values[size_t(y) * width + size_t(x)]; }
    size_t count() const { return size_t(std::count(values.begin(), values.end(), uint8_t(255))); }
};

Mask readMask(VulkanContext& vc, VkImage image, VkImageLayout layout, VkExtent2D extent) {
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
    Mask mask{extent.width, extent.height, std::vector<uint8_t>(size_t(extent.width) * extent.height)};
    for (size_t i = 0; i < mask.values.size(); ++i) {
        mask.values[i] = std::max({bgra[4 * i], bgra[4 * i + 1], bgra[4 * i + 2]});
    }
    vkUnmapMemory(vc.getDevice(), memory);
    vkDestroyBuffer(vc.getDevice(), buffer, nullptr);
    vkFreeMemory(vc.getDevice(), memory, nullptr);
    return mask;
}

// A perspective camera at z = 3 looking at the origin, in Vulkan clip space.
CameraMVP perspectiveCamera() {
    CameraMVP camera{};
    camera.model = glm::mat4(1.0f);
    camera.view = glm::lookAt(glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    camera.proj = glm::perspective(glm::radians(50.0f), 1.0f, 0.1f, 100.0f);
    camera.proj[1][1] *= -1.0f;
    return camera;
}

glm::vec2 project(const CameraMVP& camera, glm::vec3 position) {
    const glm::vec4 clip = camera.proj * camera.view * camera.model * glm::vec4(position, 1.0f);
    return (glm::vec2(clip) / clip.w * 0.5f + 0.5f) * glm::vec2(kExtent.width, kExtent.height);
}

float ndcDepth(const CameraMVP& camera, glm::vec3 position) {
    const glm::vec4 clip = camera.proj * camera.view * camera.model * glm::vec4(position, 1.0f);
    return clip.z / clip.w;
}

// Pixels per unit length at distance `distance` along the view axis.
float pixelsPerUnit(const CameraMVP& camera, float distance) {
    return std::abs(camera.proj[1][1]) * 0.5f * float(kExtent.height) / distance;
}

std::vector<glm::vec4> invisibleSkeleton() { return std::vector<glm::vec4>(openpose::kJointCount, glm::vec4(0.0f)); }

class CharacterMaskTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
    }

    Mask renderMask(const std::vector<glm::vec4>& joints, const CameraMVP& cameraMatrices,
                    const std::vector<float>* occlusionDepth = nullptr) {
        auto target = std::make_shared<OffscreenTarget>(*vc, kExtent, 1);
        auto camera = std::make_shared<CameraUboType>();
        camera->ubo = cameraMatrices;
        auto jointBuffer = std::make_shared<Vec4Buffer>(*vc, openpose::kJointCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        auto renderer = std::make_shared<CharacterMaskRenderer>(jointBuffer, kExtent);
        std::shared_ptr<FloatBuffer> depth;
        if (occlusionDepth) {
            depth = std::make_shared<FloatBuffer>(*vc, uint32_t(occlusionDepth->size()),
                                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            renderer->setOcclusionDepth(depth, 0.0f);
        }
        auto renderPass = std::make_shared<RenderPass>(vc->getSwapChainImageFormat(), kExtent);
        renderPass->setInput(target, 0);
        renderPass->setInput(camera, 1);
        renderPass->addDrawComponent(renderer);
        renderPass->addComputeDependency(jointBuffer);
        if (depth) {
            renderPass->addComputeDependency(depth);
        }
        ComputeGraph graph(*vc, 1);
        graph.compileFrom(renderPass);
        jointBuffer->getBuffer(0).memcopyFrom(joints);
        if (depth) {
            depth->getBuffer(0).memcopyFrom(*occlusionDepth);
        }
        camera->update(0);
        graph.submitAndWait(vc->getGraphicsQueue(), 0);
        return readMask(*vc, target->getImage(0), *target->getFinalLayoutOverride(), kExtent);
    }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
};

// Joints of a single horizontal bone, neck to right shoulder (segment 0), at depth z.
std::vector<glm::vec4> shoulderBone(float z) {
    auto joints = invisibleSkeleton();
    joints[openpose::Neck] = {-0.4f, 0.0f, z, 1.0f};
    joints[openpose::RightShoulder] = {0.4f, 0.0f, z, 1.0f};
    return joints;
}

// Height of the masked column at x.
int columnHeight(const Mask& mask, int x) {
    int height = 0;
    for (int y = 0; y < int(mask.height); ++y) {
        height += mask.at(x, y) == 255 ? 1 : 0;
    }
    return height;
}

} // namespace

TEST(CharacterMaskTables, segmentsCoverTheOpenPoseLimbs) {
    for (const auto& limb : openpose::kLimbs) {
        const bool found = std::any_of(
            characterMaskSegments().begin(), characterMaskSegments().end(),
            [&](const CharacterMaskSegment& segment) { return segment.first == limb[0] && segment.second == limb[1]; });
        EXPECT_TRUE(found) << limb[0] << "-" << limb[1];
    }
    for (const auto& segment : characterMaskSegments()) {
        EXPECT_GT(segment.radius, 0.0f);
        EXPECT_LT(segment.first, openpose::kJointCount);
        EXPECT_LT(segment.second, openpose::kJointCount);
    }
}

TEST_F(CharacterMaskTest, emptySkeletonLeavesBlackMask) {
    EXPECT_EQ(renderMask(invisibleSkeleton(), perspectiveCamera()).count(), 0u);
}

TEST_F(CharacterMaskTest, boneCapsuleHasProjectedRadius) {
    const CameraMVP camera = perspectiveCamera();
    const float radius = characterMaskSegments()[0].radius + CharacterMaskRenderer::kDefaultMargin;
    const float radiusPixels = radius * pixelsPerUnit(camera, 3.0f);
    const auto mask = renderMask(shoulderBone(0.0f), camera);

    const glm::vec2 middle = project(camera, {0.0f, 0.0f, 0.0f});
    const int x = int(middle.x), y = int(middle.y);
    EXPECT_EQ(mask.at(x, y), 255);
    EXPECT_EQ(mask.at(x, y + int(radiusPixels - 1.5f)), 255);
    EXPECT_EQ(mask.at(x, y - int(radiusPixels - 1.5f)), 255);
    EXPECT_EQ(mask.at(x, y + int(radiusPixels + 1.5f)), 0);
    EXPECT_EQ(mask.at(x, y - int(radiusPixels + 1.5f)), 0);
    EXPECT_NEAR(float(columnHeight(mask, x)), 2.0f * radiusPixels, 2.0f);
    // The ends are round: beyond the shoulder by more than the radius, nothing is masked.
    const glm::vec2 shoulder = project(camera, {0.4f, 0.0f, 0.0f});
    const int beyond = int(shoulder.x + (shoulder.x > middle.x ? 1.0f : -1.0f) * (radiusPixels + 2.0f));
    EXPECT_EQ(columnHeight(mask, beyond), 0);
}

TEST_F(CharacterMaskTest, headIsDiscBetweenEars) {
    const CameraMVP camera = perspectiveCamera();
    auto joints = invisibleSkeleton();
    joints[openpose::RightEar] = {-0.075f, 0.0f, 0.0f, 1.0f};
    joints[openpose::LeftEar] = {0.075f, 0.0f, 0.0f, 1.0f};
    const auto head = std::find_if(characterMaskSegments().begin(), characterMaskSegments().end(),
                                   [](const CharacterMaskSegment& segment) { return segment.disc; });
    ASSERT_NE(head, characterMaskSegments().end());
    const float radiusPixels = (head->radius + CharacterMaskRenderer::kDefaultMargin) * pixelsPerUnit(camera, 3.0f);
    const auto mask = renderMask(joints, camera);

    const glm::vec2 middle = project(camera, {0.0f, 0.0f, 0.0f});
    const int x = int(middle.x), y = int(middle.y);
    for (const glm::ivec2 direction : {glm::ivec2(1, 0), glm::ivec2(-1, 0), glm::ivec2(0, 1), glm::ivec2(0, -1)}) {
        EXPECT_EQ(mask.at(x + direction.x * int(radiusPixels - 1.5f), y + direction.y * int(radiusPixels - 1.5f)), 255);
        EXPECT_EQ(mask.at(x + direction.x * int(radiusPixels + 1.5f), y + direction.y * int(radiusPixels + 1.5f)), 0);
    }
}

TEST_F(CharacterMaskTest, fartherBoneIsThinner) {
    const CameraMVP camera = perspectiveCamera();
    // At z = 0 the bone is 3 units away, at z = -3 six.
    const auto near = renderMask(shoulderBone(0.0f), camera);
    const auto far = renderMask(shoulderBone(-3.0f), camera);
    const int x = int(project(camera, {0.0f, 0.0f, 0.0f}).x);
    EXPECT_NEAR(float(columnHeight(far, x)), 0.5f * float(columnHeight(near, x)), 2.0f);
}

TEST_F(CharacterMaskTest, occludedPartsAreCut) {
    const CameraMVP camera = perspectiveCamera();
    // A surface at z = 0.5 in front of the bone covers the left half of the image.
    std::vector<float> depth(size_t(kExtent.width) * kExtent.height);
    for (uint32_t y = 0; y < kExtent.height; ++y) {
        for (uint32_t x = 0; x < kExtent.width; ++x) {
            depth[size_t(y) * kExtent.width + x] =
                x < kExtent.width / 2 ? ndcDepth(camera, {0.0f, 0.0f, 0.5f}) : std::numeric_limits<float>::infinity();
        }
    }
    const auto occluded = renderMask(shoulderBone(0.0f), camera, &depth);
    const auto open = renderMask(shoulderBone(0.0f), camera);
    size_t left = 0, right = 0, openRight = 0, openLeft = 0;
    for (uint32_t y = 0; y < kExtent.height; ++y) {
        for (uint32_t x = 0; x < kExtent.width; ++x) {
            const bool masked = occluded.at(int(x), int(y)) == 255, openMasked = open.at(int(x), int(y)) == 255;
            (x < kExtent.width / 2 ? left : right) += masked ? 1 : 0;
            (x < kExtent.width / 2 ? openLeft : openRight) += openMasked ? 1 : 0;
        }
    }
    EXPECT_GT(openLeft, 0u);
    EXPECT_EQ(left, 0u);
    EXPECT_EQ(right, openRight);
}

TEST_F(CharacterMaskTest, maskCoversSkeletonDrawing) {
    if (!std::filesystem::exists("data/character/controller.onnx")) {
        GTEST_SKIP() << "Missing data/character/controller.onnx";
    }
    auto controller = std::make_shared<CharacterController>(*vc, "data/character");
    auto camera = std::make_shared<CameraUboType>();
    camera->ubo = perspectiveCamera();
    camera->ubo.view =
        glm::lookAt(glm::vec3(1.5f, 1.2f, 3.5f), glm::vec3(0.0f, 0.9f, 0.5f), glm::vec3(0.0f, 1.0f, 0.0f));
    auto barrier = std::make_shared<BufferToGraphicsBarrier>();
    barrier->addBuffer(controller->getOpenPoseJoints());
    barrier->setInput(controller, 0);

    auto poseTarget = std::make_shared<OffscreenTarget>(*vc, kExtent, 1);
    auto posePass = std::make_shared<RenderPass>(vc->getSwapChainImageFormat(), kExtent);
    posePass->setInput(poseTarget, 0);
    posePass->setInput(camera, 1);
    posePass->addDrawComponent(std::make_shared<OpenPoseSkeletonRenderer>(controller->getOpenPoseJoints(), 1, kExtent));
    posePass->addComputeDependency(barrier);
    auto maskTarget = std::make_shared<OffscreenTarget>(*vc, kExtent, 1);
    auto maskPass = std::make_shared<RenderPass>(vc->getSwapChainImageFormat(), kExtent);
    maskPass->setInput(maskTarget, 0);
    maskPass->setInput(camera, 1);
    maskPass->addDrawComponent(std::make_shared<CharacterMaskRenderer>(controller->getOpenPoseJoints(), kExtent));
    maskPass->addComputeDependency(barrier);
    maskPass->addComputeDependency(posePass);
    ComputeGraph graph(*vc, 1);
    graph.compileFrom(maskPass);
    camera->update(0);

    CharacterControl walk;
    walk.velocity = {0.0f, 1.3f};
    controller->setControl(walk);
    for (int frame = 0; frame < 30; ++frame) {
        graph.submitAndWait(vc->getGraphicsQueue(), 0);
        if (frame % 10 != 9) {
            continue;
        }
        const Mask pose = readMask(*vc, poseTarget->getImage(0), *poseTarget->getFinalLayoutOverride(), kExtent);
        const Mask mask = readMask(*vc, maskTarget->getImage(0), *maskTarget->getFinalLayoutOverride(), kExtent);
        EXPECT_GT(mask.count(), 100u) << "frame " << frame;
        size_t drawnOutside = 0;
        for (size_t i = 0; i < mask.values.size(); ++i) {
            drawnOutside += (pose.values[i] != 0 && mask.values[i] != 255) ? 1 : 0;
        }
        EXPECT_EQ(drawnOutside, 0u) << "frame " << frame;
    }
}

TEST_F(CharacterMaskTest, sceneMaskMarksSurfaces) {
    std::vector<float> values(size_t(kExtent.width) * kExtent.height, std::numeric_limits<float>::infinity());
    for (uint32_t y = 20; y < 60; ++y) {
        for (uint32_t x = 30; x < 90; ++x) {
            values[size_t(y) * kExtent.width + x] = 0.5f + 0.001f * float(x);
        }
    }
    auto depth = std::make_shared<FloatBuffer>(*vc, uint32_t(values.size()), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    auto target = std::make_shared<OffscreenTarget>(*vc, kExtent, 1);
    auto sceneMask = createSceneMask(*vc, depth, target);
    ComputeGraph graph(*vc, 1);
    graph.compileFrom(sceneMask);
    depth->getBuffer(0).memcopyFrom(values);
    graph.submitAndWait(vc->getGraphicsQueue(), 0);
    const Mask mask = readMask(*vc, target->getImage(0), VK_IMAGE_LAYOUT_GENERAL, kExtent);
    for (uint32_t y = 0; y < kExtent.height; ++y) {
        for (uint32_t x = 0; x < kExtent.width; ++x) {
            const bool surface = std::isfinite(values[size_t(y) * kExtent.width + x]);
            ASSERT_EQ(mask.at(int(x), int(y)), surface ? 255 : 0) << x << ", " << y;
        }
    }
}
