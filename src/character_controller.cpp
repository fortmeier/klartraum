// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include <fstream>
#include <stdexcept>

#include "klartraum/character_controller.hpp"

namespace klartraum {

namespace {

// Sizes of the network inputs; see scripts/character_animation/features.py.
constexpr uint32_t kMotionInputSize = 198;
constexpr uint32_t kGatingInputSize = 42;

std::vector<float> readFloats(const std::filesystem::path& path, size_t count) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || static_cast<size_t>(input.tellg()) != count * sizeof(float)) {
        throw std::runtime_error("CharacterController: missing or wrongly sized " + path.string());
    }
    input.seekg(0);
    std::vector<float> values(count);
    input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(float)));
    return values;
}

} // namespace

const std::vector<std::string> CharacterController::kStyles = {
    "Neutral",  "Proud",    "Old",        "Elated", "Strutting", "Rushed",
    "BigSteps", "Heavyset", "ArmsBySide", "Angry",  "Depressed", "March",
};

const std::vector<std::string> CharacterController::kJointNames = {
    "Hips",     "Chest2",    "Neck",          "Head",       "HeadTop",    "Nose",         "RightEye",  "LeftEye",
    "RightEar", "LeftEar",   "RightShoulder", "RightElbow", "RightWrist", "LeftShoulder", "LeftElbow", "LeftWrist",
    "RightHip", "RightKnee", "RightAnkle",    "RightToe",   "LeftHip",    "LeftKnee",     "LeftAnkle", "LeftToe",
};

CharacterController::CharacterController(VulkanContext& vulkanContext, const std::filesystem::path& modelDirectory)
    : initialState(readFloats(modelDirectory / "initial_state_f32.bin", kStateSize)) {
    constexpr VkBufferUsageFlags storage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    state = std::make_shared<FloatBuffer>(vulkanContext, kStateSize, storage);
    control = std::make_shared<FloatBuffer>(vulkanContext, kControlSize, storage);
    joints = std::make_shared<Vec4Buffer>(vulkanContext, kJointCount, storage);
    openPoseJoints = std::make_shared<Vec4Buffer>(vulkanContext, kOpenPoseJointCount, storage);
    state->getBuffer().memcopyFrom(initialState);
    setControl(CharacterControl{});
    state->setName("CharacterState");
    control->setName("CharacterControl");
    joints->setName("CharacterJoints");
    openPoseJoints->setName("CharacterOpenPoseJoints");

    motionInput = vulkanContext.create<TensorElement<float>>(std::vector<uint32_t>{1, kMotionInputSize}, storage);
    gatingInput = vulkanContext.create<TensorElement<float>>(std::vector<uint32_t>{1, kGatingInputSize}, storage);

    assemble = vulkanContext.create<GeneralComputation<>>("shaders/character/character_assemble.comp.spv");
    assemble->setName("CharacterAssemble");
    assemble->setInput(state, 0);
    assemble->setInput(control, 1);
    assemble->setInput(motionInput, 2);
    assemble->setInput(gatingInput, 3);

    network = vulkanContext.create<OnnxNetwork>((modelDirectory / "controller.onnx").string());
    network->setInputTensor("motion_input", assemble, 2);
    network->setInputTensor("gating_input", assemble, 3);

    integrate = vulkanContext.create<GeneralComputation<>>("shaders/character/character_integrate.comp.spv");
    integrate->setName("CharacterIntegrate");
    integrate->setInput(assemble, 0, 0);
    integrate->setInput(network, 1, 0);
    integrate->setInput(joints, 2);
    integrate->setInput(openPoseJoints, 3);

    outputElements[0] = integrate;
}

void CharacterController::setState(const std::vector<float>& values) {
    if (values.size() != kStateSize) {
        throw std::runtime_error("CharacterController: a state has " + std::to_string(kStateSize) + " values");
    }
    state->getBuffer().memcopyFrom(values);
}

std::vector<float> CharacterController::getState() {
    std::vector<float> values(kStateSize);
    state->getBuffer().memcopyTo(values);
    return values;
}

void CharacterController::reset() { setState(initialState); }

void CharacterController::setControl(const CharacterControl& value) {
    const glm::vec2 facing = glm::length(value.facing) > 0.0f ? glm::normalize(value.facing) : glm::vec2(0.0f, 1.0f);
    const std::vector<float> values = {value.velocity.x,
                                       value.velocity.y,
                                       facing.x,
                                       facing.y,
                                       static_cast<float>(value.style),
                                       value.responsiveness,
                                       0.0f,
                                       0.0f};
    control->getBuffer().memcopyFrom(values);
}

std::vector<glm::vec4> CharacterController::read(const std::shared_ptr<Vec4Buffer>& buffer, uint32_t count) {
    std::vector<glm::vec4> values(count);
    buffer->getBuffer().memcopyTo(values);
    return values;
}

} // namespace klartraum
