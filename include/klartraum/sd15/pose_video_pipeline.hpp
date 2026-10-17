// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_SD15_POSE_VIDEO_PIPELINE_HPP
#define KLARTRAUM_SD15_POSE_VIDEO_PIPELINE_HPP

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "klartraum/camera.hpp"
#include "klartraum/character_controller.hpp"
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/draw_component.hpp"
#include "klartraum/onnx/onnx_network.hpp"
#include "klartraum/sd15/controlnet_denoiser.hpp"
#include "klartraum/sd15/temporal_lcm_sampler.hpp"

namespace klartraum {

/**
 * @brief The SD1.5 CLIP embedding [77, 768] of `prompt`.
 *
 * Uses the text encoder and tokenizer exported by scripts/sd15_onnx/export_denoiser.py
 * into `sdDirectory`; the encoder is loaded only for this call.
 */
std::vector<float> encodeSd15Prompt(VulkanContext& vulkanContext, const std::filesystem::path& sdDirectory,
                                    const std::string& prompt);

/**
 * @brief Neural character animation rendered as photorealistic frames with SD1.5 ControlNet-OpenPose.
 *
 * Three compute graphs share their tensors on the GPU:
 * - the pose graph advances a CharacterController by one step, draws its
 *   keypoints with OpenPoseSkeletonRenderer, and converts the pose map into
 *   the pose tensor of the denoiser (stepCharacter());
 * - the denoising graph, ControlNetDenoiser, runs once per LCM step
 *   (sampleFrame(), driven by a TemporalLcmSampler);
 * - the decoding graph runs the SD1.5 VAE decoder (decode()).
 *
 * Models: the controller from scripts/character_animation (`characterDirectory`),
 * ControlNet, the LCM UNet and the LCM schedule and noise from
 * scripts/sd15_onnx/export_controlnet.py (`controlDirectory`), and the CLIP text
 * encoder, tokenizer and VAE decoder from scripts/sd15_onnx/export_denoiser.py
 * (`sdDirectory`), all exported for the same image size.
 */
class PoseVideoPipeline {
public:
    struct Directories {
        std::filesystem::path character = "data/character";
        std::filesystem::path sd;      ///< default: data/onnx/sd15_denoiser_<size>
        std::filesystem::path control; ///< default: data/onnx/sd15_controlnet_<size>
    };

    /// @param size image size in pixels, a multiple of 64
    PoseVideoPipeline(VulkanContext& vulkanContext, Directories directories, uint32_t size);
    ~PoseVideoPipeline();

    /// Encodes `prompt` with the CLIP text encoder; the encoder is loaded only for this call.
    void setPrompt(const std::string& prompt);
    /// Sets the prompt embedding [77, 768] directly.
    void setPromptEmbedding(const std::vector<float>& embedding);
    std::vector<float> getPromptEmbedding();

    /// Advances the character by one controller step with `control` and draws its pose map seen by `camera`.
    void stepCharacter(const CharacterControl& control, const CameraMVP& camera);

    /// The pose map of the last stepCharacter(), [3, size, size] with RGB in [0, 1].
    std::vector<float> readPoseMap();

    /// A sampler over the exported LCM schedule and noise.
    TemporalLcmSampler makeSampler(const TemporalLcmSampler::Settings& settings) const;

    /// Samples the latents of a frame conditioned on the current pose map.
    std::vector<float> sampleFrame(TemporalLcmSampler& sampler);

    /// Decodes latents into an image [3, size, size] with RGB in [-1, 1].
    std::vector<float> decode(const std::vector<float>& latents);

    /**
     * @brief A perspective camera at `target + offset` looking at `target`, y up, in Vulkan clip space.
     *
     * With the default offset it looks at the character from the front right,
     * slightly from above; the character walks towards +z at first.
     */
    static CameraMVP lookAt(glm::vec3 target, glm::vec3 offset = {2.0f, 0.35f, 3.4f}, float verticalFovDegrees = 40.0f);

    CharacterController& getController() { return *controller; }
    uint32_t getSize() const { return size; }
    /// Number of ControlNet and UNet evaluations so far.
    uint64_t getEvaluations() const { return evaluations; }

private:
    VulkanContext& vulkanContext;
    Directories directories;
    uint32_t size;
    uint32_t latentSize;
    uint64_t evaluations = 0;

    std::shared_ptr<ControlNetDenoiser> denoiser;
    std::shared_ptr<CharacterController> controller;
    std::shared_ptr<CameraUboType> camera;
    std::shared_ptr<OnnxNetwork> decoder;
    std::shared_ptr<TensorElement<float>> decoderInput;
    std::unique_ptr<ComputeGraph> poseGraph;
    std::unique_ptr<ComputeGraph> denoiseGraph;
    std::unique_ptr<ComputeGraph> decodeGraph;
};

} // namespace klartraum

#endif // KLARTRAUM_SD15_POSE_VIDEO_PIPELINE_HPP
