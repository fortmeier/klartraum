// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <fstream>
#include <stdexcept>

#include <glm/gtc/matrix_transform.hpp>

#include "klartraum/computegraph/buffertographicsbarrier.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/renderpass.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/openpose_skeleton_renderer.hpp"
#include "klartraum/sd15/clip_tokenizer.hpp"
#include "klartraum/sd15/pose_video_pipeline.hpp"

namespace klartraum {

namespace {

constexpr float kVaeScalingFactor = 0.18215f;
constexpr size_t kEmbeddingElements = 77 * 768;
constexpr VkBufferUsageFlags kTensorUsage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

struct ImageTensorPushConstants {
    uint32_t width;
    uint32_t height;
};

std::vector<float> readFloats(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("PoseVideoPipeline: could not open " + path.string());
    }
    std::vector<float> values(static_cast<size_t>(input.tellg()) / sizeof(float));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    return values;
}

} // namespace

PoseVideoPipeline::PoseVideoPipeline(VulkanContext& vulkanContext, Directories directories, uint32_t size)
    : vulkanContext(vulkanContext),
      directories(std::move(directories)),
      size(size),
      latentSize(size / 8) {
    if (size == 0 || size % 64 != 0) {
        throw std::runtime_error("PoseVideoPipeline: the image size must be a positive multiple of 64");
    }
    if (this->directories.sd.empty()) {
        this->directories.sd = "data/onnx/sd15_denoiser_" + std::to_string(size);
    }
    if (this->directories.control.empty()) {
        this->directories.control = "data/onnx/sd15_controlnet_" + std::to_string(size);
    }

    denoiser = std::make_shared<ControlNetDenoiser>(vulkanContext, this->directories.control, latentSize);
    controller = std::make_shared<CharacterController>(vulkanContext, this->directories.character);
    camera = std::make_shared<CameraUboType>();

    // Pose graph: controller step, barrier, pose map, and its conversion into the denoiser's pose tensor.
    const VkExtent2D extent{size, size};
    auto target = std::make_shared<OffscreenTarget>(vulkanContext, extent, 1u);
    auto barrier = std::make_shared<BufferToGraphicsBarrier>();
    barrier->addBuffer(controller->getOpenPoseJoints());
    barrier->setInput(controller, 0);
    auto renderer = std::make_shared<OpenPoseSkeletonRenderer>(controller->getOpenPoseJoints(), 1, extent);
    auto renderPass = std::make_shared<RenderPass>(vulkanContext.getSwapChainImageFormat(), extent);
    renderPass->setInput(target, 0);
    renderPass->setInput(camera, 1);
    renderPass->addDrawComponent(renderer);
    renderPass->addComputeDependency(barrier);
    auto imageToTensor =
        vulkanContext.create<GeneralComputation<ImageTensorPushConstants>>("shaders/onnx/image_to_tensor.comp.spv");
    imageToTensor->setPushConstants({{size, size}});
    imageToTensor->setGroupCount((size + 7) / 8, (size + 7) / 8, 1);
    imageToTensor->setInput(renderPass, 0);
    imageToTensor->setInput(denoiser->getPose(), 1);
    imageToTensor->setImageLayoutTransition(0, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                                            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                            VK_ACCESS_MEMORY_READ_BIT, VK_ACCESS_SHADER_READ_BIT);
    poseGraph = std::make_unique<ComputeGraph>(vulkanContext, 1);
    poseGraph->compileFrom(imageToTensor);

    denoiseGraph = std::make_unique<ComputeGraph>(vulkanContext, 1);
    denoiseGraph->compileFrom(denoiser);

    decoder = vulkanContext.create<OnnxNetwork>((this->directories.sd / "sd15_vae_decoder.onnx").string());
    decoderInput =
        vulkanContext.create<TensorElement<float>>(std::vector<uint32_t>{1, 4, latentSize, latentSize}, kTensorUsage);
    decoder->setInputTensor("input", decoderInput);
    decodeGraph = std::make_unique<ComputeGraph>(vulkanContext, 1);
    decodeGraph->compileFrom(decoder);
}

PoseVideoPipeline::~PoseVideoPipeline() = default;

CameraMVP PoseVideoPipeline::lookAt(glm::vec3 target, glm::vec3 offset, float verticalFovDegrees) {
    CameraMVP result{};
    result.model = glm::mat4(1.0f);
    const glm::vec3 eye = target + offset;
    result.view = glm::lookAt(eye, target, glm::vec3(0.0f, 1.0f, 0.0f));
    result.proj = glm::perspective(glm::radians(verticalFovDegrees), 1.0f, 0.1f, 100.0f);
    // Vulkan's clip space points y down.
    result.proj[1][1] *= -1.0f;
    result.cameraWorldPos = glm::vec4(eye, 1.0f);
    return result;
}

std::vector<float> encodeSd15Prompt(VulkanContext& vulkanContext, const std::filesystem::path& sdDirectory,
                                    const std::string& prompt) {
    // The exported encoder takes a pair (negative, positive) of prompts; the positive half is used.
    ClipTokenizer tokenizer(sdDirectory);
    const auto tokens = tokenizer.encodePair("", prompt);
    auto encoder = vulkanContext.create<OnnxNetwork>((sdDirectory / "sd15_text_encoder.onnx").string());
    auto inputIds = vulkanContext.create<TensorElement<int64_t>>(std::vector<uint32_t>{2, 77}, kTensorUsage);
    auto attention = vulkanContext.create<TensorElement<int64_t>>(std::vector<uint32_t>{2, 77}, kTensorUsage);
    encoder->setInputTensor("input_ids", inputIds);
    encoder->setInputTensor("attention_mask", attention);
    ComputeGraph graph(vulkanContext, 1);
    graph.compileFrom(encoder);
    inputIds->setData(0, tokens);
    attention->setData(0, tokenizer.attentionMask(tokens));
    graph.submitAndWait(vulkanContext.getGraphicsQueue(), 0);
    auto output = std::dynamic_pointer_cast<TensorElement<float>>(encoder->getOutputElement("last_hidden_state"));
    if (!output || output->getDataElementCount() != 2 * kEmbeddingElements) {
        throw std::runtime_error("encodeSd15Prompt: unexpected CLIP text encoder output");
    }
    std::vector<float> both(2 * kEmbeddingElements);
    output->getDataBuffer(0).memcopyTo(both);
    return std::vector<float>(both.begin() + kEmbeddingElements, both.end());
}

void PoseVideoPipeline::setPrompt(const std::string& prompt) {
    setPromptEmbedding(encodeSd15Prompt(vulkanContext, directories.sd, prompt));
}

void PoseVideoPipeline::setPromptEmbedding(const std::vector<float>& embedding) {
    if (embedding.size() != kEmbeddingElements) {
        throw std::runtime_error("PoseVideoPipeline: a prompt embedding has 77 x 768 values");
    }
    denoiser->getEmbeddings()->setData(0, embedding);
}

std::vector<float> PoseVideoPipeline::getPromptEmbedding() {
    std::vector<float> embedding(kEmbeddingElements);
    denoiser->getEmbeddings()->getDataBuffer(0).memcopyTo(embedding);
    return embedding;
}

void PoseVideoPipeline::stepCharacter(const CharacterControl& control, const CameraMVP& cameraMatrices) {
    controller->setControl(control);
    camera->ubo = cameraMatrices;
    camera->update(0);
    poseGraph->submitAndWait(vulkanContext.getGraphicsQueue(), 0);
}

std::vector<float> PoseVideoPipeline::readPoseMap() {
    std::vector<float> pose(3 * size_t(size) * size);
    denoiser->getPose()->getDataBuffer(0).memcopyTo(pose);
    return pose;
}

TemporalLcmSampler PoseVideoPipeline::makeSampler(const TemporalLcmSampler::Settings& settings) const {
    auto scheduler = LcmScheduler::load(directories.control / "lcm_schedule_f32.bin");
    const auto values = readFloats(directories.control / "lcm_noise_f32.bin");
    const size_t latentElements = 4 * size_t(latentSize) * latentSize;
    if (values.size() != scheduler.size() * latentElements) {
        throw std::runtime_error("PoseVideoPipeline: lcm_noise_f32.bin does not match the schedule and image size");
    }
    std::vector<std::vector<float>> noise;
    for (size_t step = 0; step < scheduler.size(); ++step) {
        noise.emplace_back(values.begin() + step * latentElements, values.begin() + (step + 1) * latentElements);
    }
    return TemporalLcmSampler(std::move(scheduler), std::move(noise), settings);
}

std::vector<float> PoseVideoPipeline::sampleFrame(TemporalLcmSampler& sampler) {
    const size_t latentElements = 4 * size_t(latentSize) * latentSize;
    return sampler.sampleFrame([&](const std::vector<float>& latents, int64_t timestep) {
        denoiser->getSample()->setData(0, latents);
        denoiser->getTimestep()->setData(0, std::vector<int64_t>{timestep});
        denoiseGraph->submitAndWait(vulkanContext.getGraphicsQueue(), 0);
        ++evaluations;
        std::vector<float> prediction(latentElements);
        denoiser->getNoisePrediction()->getDataBuffer(0).memcopyTo(prediction);
        return prediction;
    });
}

std::vector<float> PoseVideoPipeline::decode(const std::vector<float>& latents) {
    std::vector<float> scaled(latents.size());
    std::transform(latents.begin(), latents.end(), scaled.begin(), [](float v) { return v / kVaeScalingFactor; });
    decoderInput->setData(0, scaled);
    decodeGraph->submitAndWait(vulkanContext.getGraphicsQueue(), 0);
    auto output = std::dynamic_pointer_cast<TensorElement<float>>(decoder->getOutputElement("output"));
    if (!output) {
        throw std::runtime_error("PoseVideoPipeline: the VAE decoder has no float output named output");
    }
    std::vector<float> image(3 * size_t(size) * size);
    output->getDataBuffer(0).memcopyTo(image);
    return image;
}

} // namespace klartraum
