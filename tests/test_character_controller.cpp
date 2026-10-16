// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - describesItsSkeletonAndStyles: the controller lists 24 joints and the 12 training styles, and its OpenPose
 *   keypoints refer to existing joints
 * - loadsInitialState: after construction the state buffer holds initial_state_f32.bin, and setState(), getState()
 *   and reset() round-trip
 * - assembledInputsMatchReference: one submission writes the motion and gating inputs that the Python reference
 *   (scripts/character_animation/controller.py) builds for the first frame of the rollout
 * - singleStepMatchesOnnxRuntime: the network output and the resulting world joints of the first frame match ONNX
 *   Runtime and the Python integration
 * - rolloutMatchesReference: 50 frames driven by the reference controls (stand, walk, turn right) stay close to the
 *   Python rollout: below 1 cm joint error after the first frame and below 5 cm after the last
 * - rolloutIsPlausible: over the same rollout no value is NaN, the hips stay at walking height, the limb bones keep
 *   their length (within 1 %), the character walks forward and ends up turned towards +x
 * - openPoseKeypointsFollowJoints: the COCO-18 keypoints are the matching controller joints, the neck is the middle
 *   of the shoulders, and every keypoint is visible
 * - drivesSkeletonRendererOnGpu: a single graph runs the controller and draws its keypoints with
 *   OpenPoseSkeletonRenderer through a BufferToGraphicsBarrier; the pose maps of three consecutive frames equal
 *   pose maps drawn from CPU copies of the same keypoints, and they change from frame to frame
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "klartraum/character_controller.hpp"
#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/buffertographicsbarrier.hpp"
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/renderpass.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/openpose_skeleton_renderer.hpp"

using namespace klartraum;

namespace {

const std::filesystem::path kModelDirectory = "data/character";
constexpr uint32_t kFrames = 50;
constexpr uint32_t kMotionInputSize = 198;
constexpr uint32_t kGatingInputSize = 42;
constexpr uint32_t kOutputSize = 165;

std::vector<float> readFloats(const std::string& name) {
    const auto path = kModelDirectory / name;
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Could not open " + path.string());
    }
    std::vector<float> values(static_cast<size_t>(input.tellg()) / sizeof(float));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    return values;
}

float maximumError(const float* actual, const float* expected, size_t count) {
    float result = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(actual[i])) {
            return std::numeric_limits<float>::infinity();
        }
        result = std::max(result, std::abs(actual[i] - expected[i]));
    }
    return result;
}

// Largest distance between the controller's world joints and the reference joints of `frame`.
float jointError(const std::vector<glm::vec4>& joints, const std::vector<float>& reference, uint32_t frame) {
    float result = 0.0f;
    for (uint32_t joint = 0; joint < CharacterController::kJointCount; ++joint) {
        const float* expected = &reference[(size_t(frame) * CharacterController::kJointCount + joint) * 3];
        const glm::vec3 difference = glm::vec3(joints[joint]) - glm::vec3(expected[0], expected[1], expected[2]);
        result = std::max(result, glm::length(difference));
    }
    return result;
}

CharacterControl controlOfFrame(const std::vector<float>& controls, uint32_t frame) {
    const float* values = &controls[size_t(frame) * CharacterController::kControlSize];
    CharacterControl control;
    control.velocity = {values[0], values[1]};
    control.facing = {values[2], values[3]};
    control.style = static_cast<uint32_t>(values[4]);
    control.responsiveness = values[5];
    return control;
}

uint32_t jointIndex(const std::string& name) {
    const auto& names = CharacterController::kJointNames;
    return static_cast<uint32_t>(std::find(names.begin(), names.end(), name) - names.begin());
}

class CharacterControllerTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(kModelDirectory / "controller.onnx")) {
            GTEST_SKIP() << "Missing " << (kModelDirectory / "controller.onnx").string()
                         << "; run scripts/character_animation (see its README)";
        }
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
        controller = std::make_shared<CharacterController>(*vc, kModelDirectory);
        controls = readFloats("rollout_controls_f32.bin");
    }

    void compile() {
        graph = std::make_unique<ComputeGraph>(*vc, 1);
        graph->compileFrom(controller);
    }

    void step(uint32_t frame) {
        controller->setControl(controlOfFrame(controls, frame));
        graph->submitAndWait(vc->getGraphicsQueue(), 0);
    }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
    std::shared_ptr<CharacterController> controller;
    std::unique_ptr<ComputeGraph> graph;
    std::vector<float> controls;
};

} // namespace

TEST(CharacterControllerTables, describesItsSkeletonAndStyles) {
    EXPECT_EQ(CharacterController::kJointNames.size(), CharacterController::kJointCount);
    EXPECT_EQ(CharacterController::kStyles.size(), 12u);
    EXPECT_EQ(CharacterController::kStyles.front(), "Neutral");
    for (const char* name : {"Hips", "Nose", "RightWrist", "LeftAnkle", "RightEar", "LeftToe"}) {
        EXPECT_LT(jointIndex(name), CharacterController::kJointCount) << name;
    }
}

TEST_F(CharacterControllerTest, loadsInitialState) {
    const auto initial = readFloats("initial_state_f32.bin");
    ASSERT_EQ(initial.size(), CharacterController::kStateSize);
    EXPECT_EQ(controller->getState(), initial);

    std::vector<float> changed(initial.size(), 0.25f);
    controller->setState(changed);
    EXPECT_EQ(controller->getState(), changed);
    controller->reset();
    EXPECT_EQ(controller->getState(), initial);
}

TEST_F(CharacterControllerTest, assembledInputsMatchReference) {
    compile();
    step(0);
    const auto motionReference = readFloats("rollout_motion_inputs_f32.bin");
    const auto gatingReference = readFloats("rollout_gating_inputs_f32.bin");
    std::vector<float> motion(kMotionInputSize), gating(kGatingInputSize);
    controller->getMotionInput()->getDataBuffer(0).memcopyTo(motion);
    controller->getGatingInput()->getDataBuffer(0).memcopyTo(gating);
    EXPECT_LT(maximumError(motion.data(), motionReference.data(), kMotionInputSize), 1e-5f);
    EXPECT_LT(maximumError(gating.data(), gatingReference.data(), kGatingInputSize), 1e-5f);
}

TEST_F(CharacterControllerTest, singleStepMatchesOnnxRuntime) {
    compile();
    step(0);
    auto output = std::dynamic_pointer_cast<TensorElement<float>>(controller->getNetwork()->getOutputElement("output"));
    ASSERT_TRUE(output);
    ASSERT_EQ(output->getDataElementCount(), kOutputSize);
    std::vector<float> values(kOutputSize);
    output->getDataBuffer(0).memcopyTo(values);
    const auto reference = readFloats("rollout_outputs_f32.bin");
    EXPECT_LT(maximumError(values.data(), reference.data(), kOutputSize), 1e-3f);
    EXPECT_LT(jointError(controller->readJoints(), readFloats("rollout_joints_f32.bin"), 0), 1e-4f);
}

TEST_F(CharacterControllerTest, rolloutMatchesReference) {
    compile();
    const auto reference = readFloats("rollout_joints_f32.bin");
    ASSERT_EQ(reference.size(), size_t(kFrames) * CharacterController::kJointCount * 3);
    float first = 0.0f, last = 0.0f;
    for (uint32_t frame = 0; frame < kFrames; ++frame) {
        step(frame);
        const float error = jointError(controller->readJoints(), reference, frame);
        if (frame == 0) {
            first = error;
        }
        last = error;
    }
    EXPECT_LT(first, 0.01f);
    EXPECT_LT(last, 0.05f);
}

TEST_F(CharacterControllerTest, rolloutIsPlausible) {
    compile();
    const std::vector<std::pair<std::string, std::string>> bones = {
        {"RightHip", "RightKnee"},     {"RightKnee", "RightAnkle"},     {"LeftHip", "LeftKnee"},
        {"LeftKnee", "LeftAnkle"},     {"RightShoulder", "RightElbow"}, {"RightElbow", "RightWrist"},
        {"LeftShoulder", "LeftElbow"}, {"LeftElbow", "LeftWrist"},
    };
    std::vector<float> firstLengths;
    std::vector<glm::vec4> joints;
    for (uint32_t frame = 0; frame < kFrames; ++frame) {
        step(frame);
        joints = controller->readJoints();
        for (const auto& joint : joints) {
            ASSERT_TRUE(std::isfinite(joint.x) && std::isfinite(joint.y) && std::isfinite(joint.z)) << frame;
        }
        const float hips = joints[jointIndex("Hips")].y;
        EXPECT_GT(hips, 0.75f) << frame;
        EXPECT_LT(hips, 1.15f) << frame;
        for (size_t bone = 0; bone < bones.size(); ++bone) {
            const float length = glm::length(glm::vec3(joints[jointIndex(bones[bone].first)]) -
                                             glm::vec3(joints[jointIndex(bones[bone].second)]));
            if (frame == 0) {
                firstLengths.push_back(length);
            } else {
                EXPECT_NEAR(length / firstLengths[bone], 1.0f, 0.01f) << bones[bone].first << " frame " << frame;
            }
        }
    }
    // The controls walk forward (+z) and then turn right, towards +x.
    const glm::vec4 hips = joints[jointIndex("Hips")];
    EXPECT_GT(hips.z, 0.3f);
    EXPECT_GT(hips.x, -0.3f);
    const std::vector<float> state = controller->getState();
    EXPECT_GT(state[2], 0.1f) << "final facing angle";
}

TEST_F(CharacterControllerTest, openPoseKeypointsFollowJoints) {
    compile();
    step(0);
    const auto joints = controller->readJoints();
    const auto keypoints = controller->readOpenPoseJoints();
    const std::vector<std::pair<uint32_t, std::string>> pairs = {
        {openpose::Nose, "Nose"},
        {openpose::RightShoulder, "RightShoulder"},
        {openpose::RightWrist, "RightWrist"},
        {openpose::LeftElbow, "LeftElbow"},
        {openpose::RightHip, "RightHip"},
        {openpose::LeftAnkle, "LeftAnkle"},
        {openpose::RightEye, "RightEye"},
        {openpose::LeftEar, "LeftEar"},
    };
    for (const auto& [keypoint, joint] : pairs) {
        EXPECT_EQ(glm::vec3(keypoints[keypoint]), glm::vec3(joints[jointIndex(joint)])) << joint;
    }
    const glm::vec3 neck =
        0.5f * (glm::vec3(joints[jointIndex("RightShoulder")]) + glm::vec3(joints[jointIndex("LeftShoulder")]));
    EXPECT_LT(glm::length(glm::vec3(keypoints[openpose::Neck]) - neck), 1e-6f);
    for (const auto& keypoint : keypoints) {
        EXPECT_EQ(keypoint.w, 1.0f);
    }
}

namespace {

// Camera looking at the character's hips from the front, at a distance of 4 m.
CameraMVP frontCamera() {
    CameraMVP camera{};
    camera.model = glm::mat4(1.0f);
    const glm::vec3 eye(0.0f, 1.0f, 4.0f), centre(0.0f, 0.9f, 0.0f);
    const glm::vec3 forward = glm::normalize(centre - eye);
    const glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));
    const glm::vec3 down = glm::cross(forward, right);
    // Vulkan clip space: x right, y down, z forward.
    camera.view = glm::mat4(glm::vec4(right.x, down.x, forward.x, 0.0f), glm::vec4(right.y, down.y, forward.y, 0.0f),
                            glm::vec4(right.z, down.z, forward.z, 0.0f),
                            glm::vec4(-glm::dot(right, eye), -glm::dot(down, eye), -glm::dot(forward, eye), 1.0f));
    const float focal = 2.5f;
    camera.proj = glm::mat4(glm::vec4(focal, 0, 0, 0), glm::vec4(0, focal, 0, 0), glm::vec4(0, 0, 0, 1),
                            glm::vec4(0, 0, 0.1f, 0));
    return camera;
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

} // namespace

TEST_F(CharacterControllerTest, drivesSkeletonRendererOnGpu) {
    const VkExtent2D extent{128, 128};
    auto camera = std::make_shared<CameraUboType>();
    camera->ubo = frontCamera();

    // Controller and renderer in one graph.
    auto target = std::make_shared<OffscreenTarget>(*vc, extent, 1);
    auto barrier = std::make_shared<BufferToGraphicsBarrier>();
    barrier->addBuffer(controller->getOpenPoseJoints());
    barrier->setInput(controller, 0);
    auto renderer = std::make_shared<OpenPoseSkeletonRenderer>(controller->getOpenPoseJoints(), 1, extent);
    auto renderPass = std::make_shared<RenderPass>(vc->getSwapChainImageFormat(), extent);
    renderPass->setInput(target, 0);
    renderPass->setInput(camera, 1);
    renderPass->addDrawComponent(renderer);
    renderPass->addComputeDependency(barrier);
    ComputeGraph combined(*vc, 1);
    combined.compileFrom(renderPass);

    // The same renderer fed from a CPU copy of the keypoints.
    auto copyTarget = std::make_shared<OffscreenTarget>(*vc, extent, 1);
    auto copiedJoints = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(
        *vc, openpose::kJointCount, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    auto copyRenderer = std::make_shared<OpenPoseSkeletonRenderer>(copiedJoints, 1, extent);
    auto copyPass = std::make_shared<RenderPass>(vc->getSwapChainImageFormat(), extent);
    copyPass->setInput(copyTarget, 0);
    copyPass->setInput(camera, 1);
    copyPass->addDrawComponent(copyRenderer);
    copyPass->addComputeDependency(copiedJoints);
    ComputeGraph copyGraph(*vc, 1);
    copyGraph.compileFrom(copyPass);
    camera->update(0);

    std::vector<uint8_t> previous;
    for (uint32_t frame = 0; frame < 3; ++frame) {
        controller->setControl(controlOfFrame(controls, frame + kFrames / 5));
        combined.submitAndWait(vc->getGraphicsQueue(), 0);
        const auto image = readImage(*vc, target->getImage(0), *target->getFinalLayoutOverride(), extent);

        copiedJoints->getBuffer(0).memcopyFrom(controller->readOpenPoseJoints());
        copyGraph.submitAndWait(vc->getGraphicsQueue(), 0);
        const auto expected = readImage(*vc, copyTarget->getImage(0), *copyTarget->getFinalLayoutOverride(), extent);

        EXPECT_EQ(image, expected) << "frame " << frame;
        size_t drawn = 0;
        for (size_t i = 0; i < image.size(); i += 4) {
            drawn += (image[i] | image[i + 1] | image[i + 2]) != 0 ? 1 : 0;
        }
        EXPECT_GT(drawn, 200u) << "frame " << frame;
        if (!previous.empty()) {
            EXPECT_NE(image, previous) << "frame " << frame;
        }
        previous = image;
    }
}
