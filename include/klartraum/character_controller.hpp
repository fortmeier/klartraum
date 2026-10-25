// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_CHARACTER_CONTROLLER_HPP
#define KLARTRAUM_CHARACTER_CONTROLLER_HPP

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/computegraphgroup.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/onnx/onnx_network.hpp"
#include "klartraum/vulkan_buffer.hpp"

namespace klartraum {

/// What the character should do; set with CharacterController::setControl() before submitting a frame.
struct CharacterControl {
    glm::vec2 velocity{0.0f};     ///< desired root velocity on the ground (world x, z) in m/s
    glm::vec2 facing{0.0f, 1.0f}; ///< desired facing direction on the ground (world x, z)
    uint32_t style = 0;           ///< index into CharacterController::kStyles
    float responsiveness = 0.6f;  ///< 0 follows the network's own trajectory prediction, 1 follows the control
};

/**
 * @brief Neural character animation in the compute graph: a mixture-of-experts locomotion controller.
 *
 * The network follows the mixture-of-experts motion controllers of Sebastian
 * Starke and colleagues (mode-adaptive neural networks, Neural State Machine,
 * Local Motion Phases): a gating network blends eight expert networks that
 * predict the next pose of a character from its current pose and a trajectory
 * of its root. It is trained on the 100STYLE dataset by
 * scripts/character_animation and loaded as `controller.onnx`.
 *
 * Every submission of the graph advances the character by one frame at 30 fps,
 * entirely on the GPU: an assemble shader builds the network input from the
 * controller state and the control, the network runs, and an integrate shader
 * updates the state and writes the joints. The CPU only writes the control.
 * Poses are joint positions; getOpenPoseJoints() holds them as the COCO-18
 * keypoints that OpenPoseSkeletonRenderer draws (place a
 * BufferToGraphicsBarrier between the controller and the render pass).
 *
 * The state and control live in one buffer shared by all paths, so the graph
 * must submit its paths one after another.
 */
class CharacterController : public ComputeGraphGroup {
public:
    static constexpr uint32_t kJointCount = 24;
    static constexpr uint32_t kOpenPoseJointCount = 18;
    static constexpr uint32_t kStateSize = 345;
    static constexpr uint32_t kControlSize = 8;
    static constexpr uint32_t kFramesPerSecond = 30;

    /// Styles of the training data, in the order of CharacterControl::style.
    static const std::vector<std::string> kStyles;
    /// Controller joints, in the order of getJoints().
    static const std::vector<std::string> kJointNames;

    /**
     * @param vulkanContext the context the controller network and its buffers are created in
     * @param modelDirectory directory with `controller.onnx` and `initial_state_f32.bin`
     *        (written by scripts/character_animation)
     */
    CharacterController(VulkanContext& vulkanContext, const std::filesystem::path& modelDirectory);

    virtual const char* getType() const override { return "CharacterController"; }

    /// Writes the state (kStateSize floats); call after the graph is compiled. reset() restores the initial one.
    void setState(const std::vector<float>& state);
    std::vector<float> getState();
    void reset();
    const std::vector<float>& getInitialState() const { return initialState; }

    /// Writes the control for the next submission; call after the graph is compiled.
    void setControl(const CharacterControl& control);

    /// World joint positions (vec4, w = 1) of the frame the last submission produced, in the order of kJointNames.
    std::shared_ptr<BufferElementInterface> getJoints() { return joints; }
    /// The same joints as COCO-18 OpenPose keypoints (vec4, w = visibility).
    std::shared_ptr<BufferElementInterface> getOpenPoseJoints() { return openPoseJoints; }
    std::shared_ptr<OnnxNetwork> getNetwork() { return network; }
    std::shared_ptr<TensorElement<float>> getMotionInput() { return motionInput; }
    std::shared_ptr<TensorElement<float>> getGatingInput() { return gatingInput; }

    std::vector<glm::vec4> readJoints() { return read(joints, kJointCount); }
    std::vector<glm::vec4> readOpenPoseJoints() { return read(openPoseJoints, kOpenPoseJointCount); }

private:
    using FloatBuffer = BufferElementSinglePath<VulkanBuffer<float>>;
    using Vec4Buffer = BufferElementSinglePath<VulkanBuffer<glm::vec4>>;

    static std::vector<glm::vec4> read(const std::shared_ptr<Vec4Buffer>& buffer, uint32_t count);

    std::vector<float> initialState;
    std::shared_ptr<FloatBuffer> state;
    std::shared_ptr<FloatBuffer> control;
    std::shared_ptr<Vec4Buffer> joints;
    std::shared_ptr<Vec4Buffer> openPoseJoints;
    std::shared_ptr<TensorElement<float>> motionInput;
    std::shared_ptr<TensorElement<float>> gatingInput;
    std::shared_ptr<GeneralComputation<>> assemble;
    std::shared_ptr<OnnxNetwork> network;
    std::shared_ptr<GeneralComputation<>> integrate;
};

} // namespace klartraum

#endif // KLARTRAUM_CHARACTER_CONTROLLER_HPP
