// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

// A neural animated character walks once around the lantern of the Gaussian
// splatting scene, drawn as an OpenPose skeleton over the scene, headless.
//
//   lantern_walk_example --size 512 --out-dir build/TestingOutput/lantern_walk
//
// The lantern is measured from its Gaussians (measureUprightObject), and the
// character is placed and scaled so that it is about half as tall as the
// lantern (characterToScene). A CirclePathFollower steers the
// CharacterController in a full circle on the moss, between the lantern's
// footprint and the edge of the moss. Each frame, one graph advances the
// controller, splats the scene with its depth, and draws the skeleton over it;
// a second render pass draws the skeleton alone, the pose map for ControlNet.
// Joints behind the lantern are hidden in both, using the splatting's depth.
// For inpainting with ControlNet it also renders the character mask (the
// region of the body around the skeleton, cut by the scene in front of it)
// and the scene mask (where the splatting found a surface). Writes per frame,
// seen from a fixed camera: frame_XXXX.ppm (scene and skeleton), pose_XXXX.ppm,
// character_mask_XXXX.ppm, scene_mask_XXXX.ppm, and panels_XXXX.ppm with the
// scene and skeleton, the pose map, the character mask, and both masks (scene
// in blue-grey, character in white) in a 2 x 2 grid.
//
// With --generate it also renders the person with SD1.5, ControlNet-OpenPose
// and LCM-LoRA into the scene, as generated_XXXX.ppm: the scene without the
// character is rendered once and encoded with the SD1.5 VAE; every frame is
// then inpainted inside the character mask by latent blending (outside the
// mask, every step keeps the scene's latent) and pasted into the scene render
// with a slightly feathered mask, so the lantern stays as rendered, also where
// it is in front of the person. The noise is cut from a larger noise field at
// an offset that follows the character, so the person is always generated from
// the same noise. The third-party models are exported by scripts/sd15_onnx.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "klartraum/camera.hpp"
#include "klartraum/character_controller.hpp"
#include "klartraum/character_mask.hpp"
#include "klartraum/computegraph/buffertographicsbarrier.hpp"
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/computegraph/renderpass.hpp"
#include "klartraum/rendering/gaussian_data_standard.hpp"
#include "klartraum/rendering/gaussian_splatting_factory.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/interface_camera_orbit.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/openpose_skeleton_renderer.hpp"
#include "klartraum/scene_placement.hpp"
#include "klartraum/onnx/onnx_network.hpp"
#include "klartraum/sd15/controlnet_denoiser.hpp"
#include "klartraum/sd15/latent_inpainting.hpp"
#include "klartraum/sd15/lcm_scheduler.hpp"
#include "klartraum/sd15/pose_video_pipeline.hpp"

namespace {

// Height of the controller's character from the ground to the top of the head, in metres.
constexpr float kCharacterHeight = 1.75f;

struct Options {
    std::string scene = "data/lantern.spz";
    std::filesystem::path characterDirectory = "data/character";
    std::filesystem::path outDirectory = "build/TestingOutput/lantern_walk";
    uint32_t size = 512;
    uint32_t controllerSteps = 2;
    float relativeHeight = 0.5f;
    float pathPosition = 0.5f;
    float speed = 1.2f;
    std::string style = "Neutral";
    glm::vec3 cameraPosition{0.55f, 0.48f, 0.69f};
    float distance = 0.0f;
    bool generate = false;
    std::string prompt = "a photograph of a woman in a dark blue kimono walking in a japanese moss garden, full body, "
                         "natural light, realistic";
    std::filesystem::path sdDirectory;      // CLIP, tokenizer; default data/onnx/sd15_denoiser_<size>
    std::filesystem::path vaeDirectory;     // VAE encoder and decoder; default data/onnx/sd15_<size>
    std::filesystem::path controlDirectory; // ControlNet, LCM UNet, schedule; default data/onnx/sd15_controlnet_<size>
    uint32_t seed = 5;
    float orbitDegrees = 20.0f;   // how far the camera orbits around the lantern during the walk
    float pushIn = 0.12f;         // how much closer (fraction of the distance) the camera gets during the walk
    float pasteMargin = 0.0f;     // margin of the body region that is pasted into the scene, in metres
    float turns = 1.0f;           // how many times the character walks around the lantern
    float ringWeight = 0.5f;      // how freely SD may paint the margin around the body (1: freely, 0: keep the scene)
    bool latentComposite = false; // the close-up is the decoded latent as a whole, without pasting in pixel space
};

constexpr float kVaeScalingFactor = 0.18215f;
constexpr VkBufferUsageFlags kTensorUsage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

// RGB of a BGRA image as a [3, h, w] tensor with values from `low` to `high`.
std::vector<float> toTensor(const std::vector<uint8_t>& bgra, size_t pixels, float low, float high) {
    std::vector<float> tensor(3 * pixels);
    for (size_t i = 0; i < pixels; ++i) {
        for (size_t channel = 0; channel < 3; ++channel) {
            tensor[channel * pixels + i] = low + (high - low) * float(bgra[4 * i + 2 - channel]) / 255.0f;
        }
    }
    return tensor;
}

std::vector<uint8_t> readImage(klartraum::VulkanContext& vc, VkImage image, VkImageLayout layout, VkExtent2D extent) {
    const VkDeviceSize bytes = VkDeviceSize(extent.width) * extent.height * 4;
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
    std::vector<uint8_t> result(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + bytes);
    vkUnmapMemory(vc.getDevice(), memory);
    vkDestroyBuffer(vc.getDevice(), buffer, nullptr);
    vkFreeMemory(vc.getDevice(), memory, nullptr);
    return result;
}

// A feathered version of a mask (non-zero: set): its coverage of a 5 x 5 box, from 0 to 1.
std::vector<float> feather(const std::vector<uint8_t>& mask, int width, int height) {
    std::vector<float> result(mask.size(), 0.0f);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float coverage = 0.0f;
            for (int dy = -2; dy <= 2; ++dy) {
                for (int dx = -2; dx <= 2; ++dx) {
                    const int sx = std::clamp(x + dx, 0, width - 1), sy = std::clamp(y + dy, 0, height - 1);
                    coverage += mask[size_t(sy) * size_t(width) + size_t(sx)] != 0 ? 1.0f : 0.0f;
                }
            }
            result[size_t(y) * size_t(width) + size_t(x)] = coverage / 25.0f;
        }
    }
    return result;
}

// The red channel of a BGRA image.
std::vector<uint8_t> redChannel(const std::vector<uint8_t>& bgra) {
    std::vector<uint8_t> result(bgra.size() / 4);
    for (size_t i = 0; i < result.size(); ++i) {
        result[i] = bgra[4 * i + 2];
    }
    return result;
}

/**
 * Inpaints a person into the scene with SD1.5, ControlNet-OpenPose and LCM-LoRA (see the comment at the top).
 *
 * The person is small in the full view, too small for SD1.5. Every frame is therefore generated in a close-up: a
 * crop camera (cropProjection()) zooms onto a square window around the character, and a graph of its own renders the
 * scene with its depth, the pose map and the character mask through it at the generation size. The close-up is
 * inpainted and pasted into the full frame through the full view's character mask.
 */
class Generator {
public:
    Generator(klartraum::VulkanContext& vc, const Options& options, VkExtent2D extent,
              std::shared_ptr<klartraum::GaussianDataStandard> model,
              std::shared_ptr<klartraum::BufferElementInterface> joints, const glm::mat4& characterToScene)
        : vc(vc),
          extent(extent),
          latentSize(extent.width / 8),
          characterToScene(characterToScene),
          ringWeight(options.ringWeight),
          latentComposite(options.latentComposite),
          scheduler(klartraum::LcmScheduler::load(options.controlDirectory / "lcm_schedule_f32.bin")),
          noise(klartraum::NoiseField(scheduler.size(), 4, latentSize, latentSize, options.seed)
                    .window(0, 0, latentSize, latentSize)) {
        // The close-up graph: scene and depth, pose map and character mask through the crop camera.
        sceneCamera = std::make_shared<klartraum::CameraUboType>();
        skeletonCamera = std::make_shared<klartraum::CameraUboType>();
        sceneTarget = std::make_shared<klartraum::OffscreenTarget>(vc, extent, 1u);
        klartraum::GsplatConfig config;
        config.outputDepth = true;
        auto splatting = klartraum::createGaussianSplatting(vc, klartraum::GsplatBackend::Compute, sceneTarget,
                                                            sceneCamera, model, config);
        const auto depth = klartraum::getGaussianSplattingDepth(splatting);
        poseTarget = std::make_shared<klartraum::OffscreenTarget>(vc, extent, 1u);
        auto skeleton = std::make_shared<klartraum::OpenPoseSkeletonRenderer>(joints, 1, extent);
        skeleton->setOcclusionDepth(depth);
        auto posePass = std::make_shared<klartraum::RenderPass>(vc.getSwapChainImageFormat(), extent);
        posePass->setInput(poseTarget, 0);
        posePass->setInput(skeletonCamera, 1);
        posePass->addDrawComponent(skeleton);
        posePass->addComputeDependency(joints);
        posePass->addComputeDependency(splatting);
        maskTarget = std::make_shared<klartraum::OffscreenTarget>(vc, extent, 1u);
        auto mask = std::make_shared<klartraum::CharacterMaskRenderer>(joints, extent);
        mask->setOcclusionDepth(depth);
        auto maskPass = std::make_shared<klartraum::RenderPass>(vc.getSwapChainImageFormat(), extent);
        maskPass->setInput(maskTarget, 0);
        maskPass->setInput(skeletonCamera, 1);
        maskPass->addDrawComponent(mask);
        maskPass->addComputeDependency(posePass);
        // The body without margin: the region pasted into the scene.
        pasteTarget = std::make_shared<klartraum::OffscreenTarget>(vc, extent, 1u);
        auto pasteMask = std::make_shared<klartraum::CharacterMaskRenderer>(joints, extent);
        pasteMask->setMargin(options.pasteMargin);
        pasteMask->setOcclusionDepth(depth);
        auto pastePass = std::make_shared<klartraum::RenderPass>(vc.getSwapChainImageFormat(), extent);
        pastePass->setInput(pasteTarget, 0);
        pastePass->setInput(skeletonCamera, 1);
        pastePass->addDrawComponent(pasteMask);
        pastePass->addComputeDependency(maskPass);
        closeUpGraph = std::make_unique<klartraum::ComputeGraph>(vc, 1);
        closeUpGraph->compileFrom(pastePass);

        encoder = vc.create<klartraum::OnnxNetwork>((options.vaeDirectory / "sd15_vae_encoder.onnx").string());
        encoderInput = vc.create<klartraum::TensorElement<float>>(
            std::vector<uint32_t>{1, 3, extent.height, extent.width}, kTensorUsage);
        encoder->setInputTensor("input", encoderInput);
        encodeGraph = std::make_unique<klartraum::ComputeGraph>(vc, 1);
        encodeGraph->compileFrom(encoder);

        denoiser = std::make_shared<klartraum::ControlNetDenoiser>(vc, options.controlDirectory, latentSize);
        denoiseGraph = std::make_unique<klartraum::ComputeGraph>(vc, 1);
        denoiseGraph->compileFrom(denoiser);
        denoiser->getEmbeddings()->setData(0, klartraum::encodeSd15Prompt(vc, options.sdDirectory, options.prompt));

        decoder = vc.create<klartraum::OnnxNetwork>((options.vaeDirectory / "sd15_vae_decoder.onnx").string());
        decoderInput = vc.create<klartraum::TensorElement<float>>(std::vector<uint32_t>{1, 4, latentSize, latentSize},
                                                                  kTensorUsage);
        decoder->setInputTensor("input", decoderInput);
        decodeGraph = std::make_unique<klartraum::ComputeGraph>(vc, 1);
        decodeGraph->compileFrom(decoder);
    }

    struct Result {
        std::vector<uint8_t> full;         ///< the scene with the person, as seen by the full view (BGRA)
        std::vector<uint8_t> closeUp;      ///< the close-up the person was generated in (BGRA)
        std::vector<uint8_t> closeUpPose;  ///< the close-up pose map ControlNet saw (BGRA)
        std::vector<uint8_t> closeUpMasks; ///< generated region in grey, pasted region in white (BGRA)
        double ringDifference = 0.0;       ///< mean |generated - scene| in the margin ring, of 255
    };

    /**
     * The frame for the character's current joints. `scene` and `characterMask` are the full view's scene (without
     * the character) and character mask, seen with the camera `matrices`; `openPoseJoints` the joints in the
     * character's world.
     */
    Result frame(const std::vector<uint8_t>& scene, const std::vector<uint8_t>& characterMask,
                 const std::vector<glm::vec4>& openPoseJoints, const klartraum::CameraMVP& matrices) {
        sceneMatrices = matrices;
        updateWindow(openPoseJoints);
        sceneCamera->ubo = sceneMatrices;
        sceneCamera->ubo.proj = klartraum::cropProjection(sceneMatrices.proj, centre, halfExtent);
        skeletonCamera->ubo = sceneCamera->ubo;
        skeletonCamera->ubo.model = characterToScene;
        sceneCamera->update(0);
        skeletonCamera->update(0);
        closeUpGraph->submitAndWait(vc.getGraphicsQueue(), 0);
        const auto closeUpScene =
            readImage(vc, sceneTarget->getImage(0), *sceneTarget->getFinalLayoutOverride(), extent);
        const auto closeUpPose = readImage(vc, poseTarget->getImage(0), *poseTarget->getFinalLayoutOverride(), extent);
        const auto closeUpMask =
            redChannel(readImage(vc, maskTarget->getImage(0), *maskTarget->getFinalLayoutOverride(), extent));
        const auto pasteMask =
            redChannel(readImage(vc, pasteTarget->getImage(0), *pasteTarget->getFinalLayoutOverride(), extent));
        std::vector<uint8_t> masks(closeUpScene.size(), 0);
        for (size_t i = 0; i < closeUpMask.size(); ++i) {
            const uint8_t value = pasteMask[i] ? 255 : closeUpMask[i] ? 110 : 0;
            masks[4 * i] = masks[4 * i + 1] = masks[4 * i + 2] = value;
            masks[4 * i + 3] = 255;
        }
        if (std::none_of(closeUpMask.begin(), closeUpMask.end(), [](uint8_t value) { return value != 0; })) {
            return {scene, closeUpScene, closeUpPose, masks}; // the character is hidden
        }

        const size_t pixels = size_t(extent.width) * extent.height;
        encoderInput->setData(0, toTensor(closeUpScene, pixels, -1.0f, 1.0f));
        encodeGraph->submitAndWait(vc.getGraphicsQueue(), 0);
        std::vector<float> known(4 * size_t(latentSize) * latentSize);
        std::dynamic_pointer_cast<klartraum::TensorElement<float>>(encoder->getOutputElement("output"))
            ->getDataBuffer(0)
            .memcopyTo(known);
        for (auto& value : known) {
            value *= kVaeScalingFactor;
        }

        denoiser->getPose()->setData(0, toTensor(closeUpPose, pixels, 0.0f, 1.0f));
        // Generated freely where the body is; in the margin ring around it only half, so SD stays close to the scene
        // there but clothing can still reach into it.
        const auto bodyLatent = klartraum::latentMask(pasteMask, extent.width, extent.height, 8, 1);
        auto latentMask = klartraum::latentMask(closeUpMask, extent.width, extent.height, 8, 1);
        for (size_t i = 0; i < latentMask.size(); ++i) {
            latentMask[i] = bodyLatent[i] > 0.0f ? 1.0f : latentMask[i] * ringWeight;
        }
        const klartraum::LatentInpainting inpainting{latentMask, known};
        auto latents = klartraum::sampleInpainted(
            scheduler, noise, inpainting, [&](const std::vector<float>& sample, int64_t timestep) {
                denoiser->getSample()->setData(0, sample);
                denoiser->getTimestep()->setData(0, std::vector<int64_t>{timestep});
                denoiseGraph->submitAndWait(vc.getGraphicsQueue(), 0);
                std::vector<float> prediction(known.size());
                denoiser->getNoisePrediction()->getDataBuffer(0).memcopyTo(prediction);
                return prediction;
            });
        for (auto& value : latents) {
            value /= kVaeScalingFactor;
        }
        decoderInput->setData(0, latents);
        decodeGraph->submitAndWait(vc.getGraphicsQueue(), 0);
        std::vector<float> generated(3 * pixels);
        std::dynamic_pointer_cast<klartraum::TensorElement<float>>(decoder->getOutputElement("output"))
            ->getDataBuffer(0)
            .memcopyTo(generated);

        // The close-up: the generated person pasted into the close-up scene through the feathered body mask, or with
        // latentComposite the decoded latent as a whole, where scene and person were only combined in latent space.
        const int width = int(extent.width), height = int(extent.height);
        Result result{scene, closeUpScene, closeUpPose, masks};
        const auto closeUpAlpha =
            latentComposite ? std::vector<float>(pixels, 1.0f) : feather(pasteMask, width, height);
        double ringSum = 0.0;
        size_t ringPixels = 0;
        for (size_t i = 0; i < pixels; ++i) {
            if (closeUpMask[i] && !pasteMask[i]) {
                for (size_t channel = 0; channel < 3; ++channel) {
                    const float value = (std::clamp(generated[channel * pixels + i], -1.0f, 1.0f) + 1.0f) * 127.5f;
                    ringSum += std::abs(value - float(closeUpScene[4 * i + 2 - channel]));
                }
                ++ringPixels;
            }
        }
        result.ringDifference = ringPixels ? ringSum / (3.0 * double(ringPixels)) : 0.0;
        for (size_t i = 0; i < pixels; ++i) {
            for (size_t channel = 0; channel < 3; ++channel) {
                const float value = (std::clamp(generated[channel * pixels + i], -1.0f, 1.0f) + 1.0f) * 127.5f;
                const float original = float(closeUpScene[4 * i + 2 - channel]);
                result.closeUp[4 * i + 2 - channel] =
                    uint8_t(closeUpAlpha[i] * value + (1.0f - closeUpAlpha[i]) * original + 0.5f);
            }
        }

        // The full view: the close-up and its paste mask, resampled into the window. The full view's character mask
        // only limits the work to where the character can be. With latentComposite the close-up is taken inside the
        // whole region SD painted, as both views have their own resolution and cannot share a latent.
        const auto fullAlpha = latentComposite ? feather(closeUpMask, width, height) : closeUpAlpha;
        std::vector<uint8_t> alphaImage(4 * pixels, 0);
        for (size_t i = 0; i < pixels; ++i) {
            alphaImage[4 * i] = uint8_t(fullAlpha[i] * 255.0f + 0.5f);
        }
        const auto region = feather(redChannel(characterMask), width, height);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const size_t i = size_t(y) * size_t(width) + size_t(x);
                if (region[i] == 0.0f) {
                    continue;
                }
                // The pixel's NDC in the full view, then in the close-up, then its close-up pixel.
                const glm::vec2 ndc((float(x) + 0.5f) / float(width) * 2.0f - 1.0f,
                                    (float(y) + 0.5f) / float(height) * 2.0f - 1.0f);
                const glm::vec2 closeUpNdc = (ndc - centre) / halfExtent;
                const glm::vec2 source = (closeUpNdc * 0.5f + 0.5f) * glm::vec2(width, height) - 0.5f;
                const float alpha = sample(alphaImage, width, height, source, 0) / 255.0f;
                for (size_t channel = 0; channel < 3; ++channel) {
                    const float value = sample(result.closeUp, width, height, source, 2 - channel);
                    result.full[4 * i + 2 - channel] =
                        uint8_t(alpha * value + (1.0f - alpha) * float(scene[4 * i + 2 - channel]) + 0.5f);
                }
            }
        }
        return result;
    }

private:
    // Bilinear sample of a BGRA channel at continuous pixel coordinates (pixel centres at integers).
    static float sample(const std::vector<uint8_t>& bgra, int width, int height, glm::vec2 position, size_t channel) {
        const float x = std::clamp(position.x, 0.0f, float(width - 1)),
                    y = std::clamp(position.y, 0.0f, float(height - 1));
        const int x0 = int(x), y0 = int(y), x1 = std::min(x0 + 1, width - 1), y1 = std::min(y0 + 1, height - 1);
        const float fx = x - float(x0), fy = y - float(y0);
        auto at = [&](int px, int py) { return float(bgra[4 * (size_t(py) * size_t(width) + size_t(px)) + channel]); };
        return (1.0f - fy) * ((1.0f - fx) * at(x0, y0) + fx * at(x1, y0)) +
               fy * ((1.0f - fx) * at(x0, y1) + fx * at(x1, y1));
    }

    // A square window around the character's joints in the full view's NDC, with room for the body around the
    // skeleton and for context. The centre follows the character exactly; the size, which swinging limbs change, is
    // smoothed from frame to frame so the close-up does not pump, but never falls below 90 % of the body's.
    void updateWindow(const std::vector<glm::vec4>& openPoseJoints) {
        glm::vec2 lower(1e9f), upper(-1e9f);
        for (const auto& joint : openPoseJoints) {
            const glm::vec4 clip =
                sceneMatrices.proj * sceneMatrices.view * characterToScene * glm::vec4(glm::vec3(joint), 1.0f);
            const glm::vec2 ndc = glm::vec2(clip) / clip.w;
            lower = glm::min(lower, ndc);
            upper = glm::max(upper, ndc);
        }
        const glm::vec2 targetCentre = 0.5f * (lower + upper);
        const float targetHalfExtent = std::max(0.08f, 0.8f * std::max(upper.x - lower.x, upper.y - lower.y));
        centre = targetCentre;
        halfExtent = initialized ? halfExtent + 0.35f * (targetHalfExtent - halfExtent) : targetHalfExtent;
        halfExtent = std::max(halfExtent, 0.9f * targetHalfExtent);
        initialized = true;
    }

    klartraum::VulkanContext& vc;
    VkExtent2D extent;
    uint32_t latentSize;
    klartraum::CameraMVP sceneMatrices;
    glm::mat4 characterToScene;
    klartraum::LcmScheduler scheduler;
    std::vector<std::vector<float>> noise;
    float ringWeight;
    bool latentComposite;
    glm::vec2 centre{0.0f};
    float halfExtent = 1.0f;
    bool initialized = false;

    std::shared_ptr<klartraum::CameraUboType> sceneCamera, skeletonCamera;
    std::shared_ptr<klartraum::OffscreenTarget> sceneTarget, poseTarget, maskTarget, pasteTarget;
    std::unique_ptr<klartraum::ComputeGraph> closeUpGraph;
    std::shared_ptr<klartraum::OnnxNetwork> encoder;
    std::shared_ptr<klartraum::TensorElement<float>> encoderInput;
    std::unique_ptr<klartraum::ComputeGraph> encodeGraph;
    std::shared_ptr<klartraum::ControlNetDenoiser> denoiser;
    std::unique_ptr<klartraum::ComputeGraph> denoiseGraph;
    std::shared_ptr<klartraum::OnnxNetwork> decoder;
    std::shared_ptr<klartraum::TensorElement<float>> decoderInput;
    std::unique_ptr<klartraum::ComputeGraph> decodeGraph;
};

// Four BGRA images of `extent` in a 2 x 2 grid, top left to bottom right.
std::vector<uint8_t> panels(const std::vector<const std::vector<uint8_t>*>& images, VkExtent2D extent) {
    const size_t width = extent.width, height = extent.height;
    std::vector<uint8_t> result(4 * width * height * 4);
    for (size_t panel = 0; panel < 4; ++panel) {
        const size_t left = (panel % 2) * width, top = (panel / 2) * height;
        for (size_t y = 0; y < height; ++y) {
            std::copy_n(images[panel]->begin() + std::ptrdiff_t(4 * y * width), 4 * width,
                        result.begin() + std::ptrdiff_t(4 * ((top + y) * 2 * width + left)));
        }
    }
    return result;
}

// Both masks in one BGRA image: the scene mask in blue-grey, the character mask in white over it.
std::vector<uint8_t> combineMasks(const std::vector<uint8_t>& sceneMask, const std::vector<uint8_t>& characterMask) {
    std::vector<uint8_t> result(sceneMask.size());
    for (size_t i = 0; i < result.size(); i += 4) {
        const bool character = characterMask[i + 2] != 0, scene = sceneMask[i + 2] != 0;
        const uint8_t blue = character ? 255
                             : scene   ? 160
                                       : 0,
                      green = character ? 255
                              : scene   ? 100
                                        : 0,
                      red = character ? 255
                            : scene   ? 70
                                      : 0;
        result[i] = blue;
        result[i + 1] = green;
        result[i + 2] = red;
        result[i + 3] = 255;
    }
    return result;
}

void writePpm(const std::filesystem::path& path, const std::vector<uint8_t>& bgra, VkExtent2D extent) {
    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << extent.width << " " << extent.height << "\n255\n";
    for (size_t i = 0; i < bgra.size(); i += 4) {
        const char rgb[3] = {char(bgra[i + 2]), char(bgra[i + 1]), char(bgra[i])};
        out.write(rgb, 3);
    }
}

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + argument);
            }
            return argv[++i];
        };
        if (argument == "--file") {
            options.scene = next();
        } else if (argument == "--character-dir") {
            options.characterDirectory = next();
        } else if (argument == "--out-dir") {
            options.outDirectory = next();
        } else if (argument == "--size") {
            options.size = static_cast<uint32_t>(std::stoul(next()));
        } else if (argument == "--controller-steps") {
            options.controllerSteps = static_cast<uint32_t>(std::stoul(next()));
        } else if (argument == "--relative-height") {
            options.relativeHeight = std::stof(next());
        } else if (argument == "--path-position") {
            options.pathPosition = std::stof(next());
        } else if (argument == "--speed") {
            options.speed = std::stof(next());
        } else if (argument == "--style") {
            options.style = next();
        } else if (argument == "--camera-position" && i + 3 < argc) {
            options.cameraPosition = {std::stof(argv[i + 1]), std::stof(argv[i + 2]), std::stof(argv[i + 3])};
            i += 3;
        } else if (argument == "--distance") {
            options.distance = std::stof(next());
        } else if (argument == "--generate") {
            options.generate = true;
        } else if (argument == "--prompt") {
            options.prompt = next();
        } else if (argument == "--sd-dir") {
            options.sdDirectory = next();
        } else if (argument == "--vae-dir") {
            options.vaeDirectory = next();
        } else if (argument == "--control-dir") {
            options.controlDirectory = next();
        } else if (argument == "--seed") {
            options.seed = static_cast<uint32_t>(std::stoul(next()));
        } else if (argument == "--orbit-degrees") {
            options.orbitDegrees = std::stof(next());
        } else if (argument == "--push-in") {
            options.pushIn = std::stof(next());
        } else if (argument == "--paste-margin") {
            options.pasteMargin = std::stof(next());
        } else if (argument == "--turns") {
            options.turns = std::stof(next());
        } else if (argument == "--latent-composite") {
            options.latentComposite = true;
        } else if (argument == "--ring-weight") {
            options.ringWeight = std::stof(next());
        } else {
            throw std::runtime_error("Unknown or incomplete argument: " + argument);
        }
    }
    if (options.size == 0 || options.controllerSteps == 0 || !(options.relativeHeight > 0.0f) ||
        !(options.turns > 0.0f)) {
        throw std::runtime_error("--size, --controller-steps, --relative-height and --turns must be positive");
    }
    if (options.generate && options.size % 64 != 0) {
        throw std::runtime_error("--generate needs a --size that is a multiple of 64");
    }
    const std::string size = std::to_string(options.size);
    if (options.sdDirectory.empty()) {
        options.sdDirectory = "data/onnx/sd15_denoiser_" + size;
    }
    if (options.vaeDirectory.empty()) {
        options.vaeDirectory = "data/onnx/sd15_" + size;
    }
    if (options.controlDirectory.empty()) {
        options.controlDirectory = "data/onnx/sd15_controlnet_" + size;
    }
    return options;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        const auto& styles = klartraum::CharacterController::kStyles;
        const auto styleIt = std::find(styles.begin(), styles.end(), options.style);
        if (styleIt == styles.end()) {
            throw std::runtime_error("Unknown --style " + options.style);
        }
        std::filesystem::create_directories(options.outDirectory);

        // Measure the lantern and place the character: half as tall, on a circle between the lantern's footprint
        // (path position 0) and the edge of the moss (1).
        const auto gaussians = klartraum::loadGaussiansSpz(options.scene, true);
        const auto lantern = klartraum::measureUprightObject(gaussians, 0.04f);
        const float scale = options.relativeHeight * lantern.height / kCharacterHeight;
        const float pathRadius =
            lantern.footprintRadius + options.pathPosition * (lantern.groundRadius - lantern.footprintRadius);
        const float radiusMetres = pathRadius / scale;
        // The character starts at the origin of its world, facing +z: on the circle, going counter-clockwise
        // around a centre on its left (-x).
        const glm::vec3 start = lantern.base + glm::vec3(radiusMetres * scale, 0.0f, 0.0f);
        const glm::mat4 characterToScene =
            klartraum::characterToScene(start, options.relativeHeight * lantern.height, kCharacterHeight);
        klartraum::CirclePathFollower follower({-radiusMetres, 0.0f}, radiusMetres, options.speed, false,
                                               static_cast<uint32_t>(styleIt - styles.begin()));
        std::cout << "lantern: base " << lantern.base.x << " " << lantern.base.y << " " << lantern.base.z << ", height "
                  << lantern.height << ", footprint radius " << lantern.footprintRadius << ", ground radius "
                  << lantern.groundRadius << std::endl;
        std::cout << "character: " << scale << " scene units per metre, circle of " << radiusMetres << " m ("
                  << radiusMetres * scale << " scene units)" << std::endl;

        klartraum::HeadlessFrontend frontend;
        auto& vc = frontend.getKlartraumEngine().getVulkanContext();
        const VkExtent2D extent{options.size, options.size};

        auto sceneCamera = std::make_shared<klartraum::CameraUboType>();
        auto skeletonCamera = std::make_shared<klartraum::CameraUboType>();
        klartraum::InterfaceCameraOrbit orbit(klartraum::InterfaceCameraOrbit::UpDirection::Y);
        orbit.initialize(vc);
        const glm::vec3 target = lantern.base + glm::vec3(0.0f, 0.4f * lantern.height, 0.0f);
        orbit.setPosition(-target);
        orbit.setAzimuth(std::atan2(options.cameraPosition.z, options.cameraPosition.x));
        orbit.setElevation(-std::asin(options.cameraPosition.y / glm::length(options.cameraPosition)));
        orbit.setDistance(options.distance > 0.0f ? options.distance : glm::length(options.cameraPosition));
        orbit.setProjectionAspectRatio(1.0f);
        orbit.update(sceneCamera->ubo);
        skeletonCamera->ubo = sceneCamera->ubo;
        skeletonCamera->ubo.model = characterToScene;

        auto controller = std::make_shared<klartraum::CharacterController>(vc, options.characterDirectory);
        auto barrier = std::make_shared<klartraum::BufferToGraphicsBarrier>();
        barrier->addBuffer(controller->getOpenPoseJoints());
        barrier->setInput(controller, 0);

        // The scene, then the skeleton drawn over it.
        auto frameTarget = std::make_shared<klartraum::OffscreenTarget>(vc, extent, 1u);
        auto model = std::make_shared<klartraum::GaussianDataStandard>(vc, gaussians);
        klartraum::GsplatConfig splatConfig;
        splatConfig.outputDepth = true;
        auto splatting = klartraum::createGaussianSplatting(vc, klartraum::GsplatBackend::Compute, frameTarget,
                                                            sceneCamera, model, splatConfig);
        // The skeleton camera shares the scene camera's view and projection, so the depths compare directly.
        const auto sceneDepth = klartraum::getGaussianSplattingDepth(splatting);
        auto makeRenderer = [&]() {
            auto renderer =
                std::make_shared<klartraum::OpenPoseSkeletonRenderer>(controller->getOpenPoseJoints(), 1, extent);
            renderer->setOcclusionDepth(sceneDepth);
            return renderer;
        };
        // Splatting leaves the target ready for transfers; drawing over it needs GENERAL (see RenderPass).
        auto toGeneral = std::make_shared<klartraum::ImageViewSrcTransition>(
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
        toGeneral->setInput(splatting, 0, 0);
        auto framePass = std::make_shared<klartraum::RenderPass>(vc.getSwapChainImageFormat(), extent);
        framePass->setLoadExisting(true);
        framePass->setInput(toGeneral, 0, 0);
        framePass->setInput(skeletonCamera, 1);
        framePass->addDrawComponent(makeRenderer());
        framePass->addComputeDependency(barrier);

        // The skeleton alone: the pose map.
        auto poseTarget = std::make_shared<klartraum::OffscreenTarget>(vc, extent, 1u);
        auto posePass = std::make_shared<klartraum::RenderPass>(vc.getSwapChainImageFormat(), extent);
        posePass->setInput(poseTarget, 0);
        posePass->setInput(skeletonCamera, 1);
        posePass->addDrawComponent(makeRenderer());
        posePass->addComputeDependency(barrier);
        posePass->addComputeDependency(framePass);

        // The masks for inpainting: the character's body region and the scene's surfaces.
        auto characterMaskTarget = std::make_shared<klartraum::OffscreenTarget>(vc, extent, 1u);
        auto characterMask =
            std::make_shared<klartraum::CharacterMaskRenderer>(controller->getOpenPoseJoints(), extent);
        characterMask->setOcclusionDepth(sceneDepth);
        auto characterMaskPass = std::make_shared<klartraum::RenderPass>(vc.getSwapChainImageFormat(), extent);
        characterMaskPass->setInput(characterMaskTarget, 0);
        characterMaskPass->setInput(skeletonCamera, 1);
        characterMaskPass->addDrawComponent(characterMask);
        characterMaskPass->addComputeDependency(barrier);
        characterMaskPass->addComputeDependency(posePass);
        auto sceneMaskTarget = std::make_shared<klartraum::OffscreenTarget>(vc, extent, 1u);
        auto sceneMask = klartraum::createSceneMask(vc, sceneDepth, sceneMaskTarget, splatting);
        characterMaskPass->addComputeDependency(sceneMask);

        klartraum::ComputeGraph graph(vc, 1);
        graph.compileFrom(characterMaskPass);
        sceneCamera->update(0);
        skeletonCamera->update(0);

        std::unique_ptr<Generator> generator;
        std::shared_ptr<klartraum::OffscreenTarget> sceneOnlyTarget;
        std::unique_ptr<klartraum::ComputeGraph> sceneOnlyGraph;
        if (options.generate) {
            // The full view of the scene without the character, rendered for every frame as the camera moves.
            sceneOnlyTarget = std::make_shared<klartraum::OffscreenTarget>(vc, extent, 1u);
            auto sceneSplatting = klartraum::createGaussianSplatting(vc, klartraum::GsplatBackend::Compute,
                                                                     sceneOnlyTarget, sceneCamera, model);
            sceneOnlyGraph = std::make_unique<klartraum::ComputeGraph>(vc, 1);
            sceneOnlyGraph->compileFrom(sceneSplatting);
            generator = std::make_unique<Generator>(vc, options, extent, model, controller->getOpenPoseJoints(),
                                                    characterToScene);
        }

        // The camera orbits slowly around the lantern and moves in a little over the whole walk, eased in and out.
        const double startAzimuth = std::atan2(options.cameraPosition.z, options.cameraPosition.x);
        const float startDistance = options.distance > 0.0f ? options.distance : glm::length(options.cameraPosition);
        auto moveCamera = [&](float progress) {
            const float eased = 0.5f - 0.5f * std::cos(3.14159265f * std::clamp(progress, 0.0f, 1.0f));
            orbit.setAzimuth(startAzimuth + double(glm::radians(options.orbitDegrees) * eased));
            orbit.setDistance(startDistance * (1.0f - options.pushIn * eased));
            orbit.update(sceneCamera->ubo);
            skeletonCamera->ubo = sceneCamera->ubo;
            skeletonCamera->ubo.model = characterToScene;
            sceneCamera->update(0);
            skeletonCamera->update(0);
        };

        uint32_t frame = 0, step = 0;
        const uint32_t maximumSteps = uint32_t(3000.0f * std::max(1.0f, options.turns));
        while (follower.completedTurns() < options.turns && step < maximumSteps) {
            const glm::vec4 hips = controller->readJoints()[0];
            controller->setControl(follower.control({hips.x, hips.z}));
            moveCamera(follower.completedTurns() / options.turns);
            graph.submitAndWait(vc.getGraphicsQueue(), 0);
            if (++step % options.controllerSteps == 0) {
                const auto combined =
                    readImage(vc, frameTarget->getImage(0), *frameTarget->getFinalLayoutOverride(), extent);
                const auto pose = readImage(vc, poseTarget->getImage(0), *poseTarget->getFinalLayoutOverride(), extent);
                const auto character = readImage(vc, characterMaskTarget->getImage(0),
                                                 *characterMaskTarget->getFinalLayoutOverride(), extent);
                // The scene mask's compute pass leaves its target in GENERAL.
                const auto scene = readImage(vc, sceneMaskTarget->getImage(0), VK_IMAGE_LAYOUT_GENERAL, extent);
                const auto masks = combineMasks(scene, character);
                auto write = [&](const char* prefix, const std::vector<uint8_t>& bgra, VkExtent2D size) {
                    char name[48];
                    std::snprintf(name, sizeof(name), "%s_%04u.ppm", prefix, frame);
                    writePpm(options.outDirectory / name, bgra, size);
                };
                write("frame", combined, extent);
                write("pose", pose, extent);
                write("character_mask", character, extent);
                write("scene_mask", scene, extent);
                write("panels", panels({&combined, &pose, &character, &masks}, extent),
                      {2 * extent.width, 2 * extent.height});
                if (generator) {
                    sceneOnlyGraph->submitAndWait(vc.getGraphicsQueue(), 0);
                    const auto sceneOnly =
                        readImage(vc, sceneOnlyTarget->getImage(0), *sceneOnlyTarget->getFinalLayoutOverride(), extent);
                    const auto generated =
                        generator->frame(sceneOnly, character, controller->readOpenPoseJoints(), sceneCamera->ubo);
                    write("generated", generated.full, extent);
                    write("closeup", generated.closeUp, extent);
                    write("generated_panels",
                          panels({&generated.full, &generated.closeUp, &generated.closeUpPose, &generated.closeUpMasks},
                                 extent),
                          {2 * extent.width, 2 * extent.height});
                    std::cout << "generated frame " << frame << ", ring difference " << generated.ringDifference
                              << std::endl;
                }
                ++frame;
            }
        }
        if (follower.completedTurns() < options.turns) {
            throw std::runtime_error("the character did not get around the lantern within " +
                                     std::to_string(maximumSteps) + " steps");
        }
        std::cout << "walked around the lantern in " << step << " controller steps (" << float(step) / 30.0f
                  << " s); wrote " << frame << " frames to " << options.outDirectory.string() << std::endl;
    } catch (const std::exception& error) {
        std::cerr << "lantern_walk_example: " << error.what() << std::endl;
        return 1;
    }
    return 0;
}
