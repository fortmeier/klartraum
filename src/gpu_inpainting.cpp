// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/sd15/gpu_inpainting.hpp"

#include <cmath>
#include <stdexcept>

namespace klartraum {

namespace {

constexpr uint32_t kGroupSize = 256;

uint32_t low(int64_t value) { return uint32_t(uint64_t(value) & 0xffffffffu); }
uint32_t high(int64_t value) { return uint32_t(uint64_t(value) >> 32); }

uint32_t groups(uint32_t size, uint32_t group) { return (size + group - 1) / group; }

// The element reads input `index`, an image in `layout`, in GENERAL.
template <typename P>
void readsImage(GeneralComputation<P>& element, int index, VkImageLayout layout) {
    element.setImageLayoutTransition(index, layout, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
                                     VK_ACCESS_SHADER_READ_BIT);
}

} // namespace

LcmInpaintingBuffers LcmInpaintingBuffers::create(VulkanContext& vulkanContext, uint32_t channels, uint32_t plane,
                                                  const std::vector<std::vector<float>>& noise) {
    if (noise.empty()) {
        throw std::runtime_error("LcmInpaintingBuffers: no noise");
    }
    LcmInpaintingBuffers buffers;
    buffers.channels = channels;
    buffers.plane = plane;
    const size_t count = size_t(channels) * plane;
    buffers.known = std::make_shared<Floats>(vulkanContext, count);
    buffers.known->setName("KnownLatent");
    buffers.mask = std::make_shared<Floats>(vulkanContext, plane);
    buffers.mask->setName("LatentMask");
    buffers.noise = std::make_shared<SharedFloats>(vulkanContext, count * noise.size());
    buffers.noise->setName("LcmNoise");
    std::vector<float> all;
    all.reserve(count * noise.size());
    for (const auto& latent : noise) {
        if (latent.size() != count) {
            throw std::runtime_error("LcmInpaintingBuffers: a noise latent has the wrong size");
        }
        all.insert(all.end(), latent.begin(), latent.end());
    }
    buffers.noise->getBuffer().memcopyFrom(all);
    return buffers;
}

std::shared_ptr<GeneralComputation<LcmInitPushConstants>>
createLcmInit(VulkanContext& vulkanContext, const LcmScheduler& scheduler, float scale, ComputeGraphElementPtr encoded,
              const LcmInpaintingBuffers& buffers, ComputeGraphElementPtr latents, ComputeGraphElementPtr timestep) {
    const auto& first = scheduler.getSteps().front();
    auto element =
        std::make_shared<GeneralComputation<LcmInitPushConstants>>(vulkanContext, "shaders/sd15/lcm_init.comp.spv");
    element->setName("LcmInit");
    element->setInput(encoded, 0);
    element->setInput(buffers.mask, 1);
    element->setInput(buffers.noise, 2);
    element->setInput(buffers.known, 3);
    element->setInput(latents, 4);
    element->setInput(timestep, 5);
    element->setPushConstants({{buffers.count(), buffers.plane, scale, std::sqrt(first.alpha),
                                std::sqrt(1.0f - first.alpha), low(first.timestep), high(first.timestep)}});
    element->setGroupCount((buffers.count() + kGroupSize - 1) / kGroupSize, 1, 1);
    return element;
}

std::shared_ptr<GeneralComputation<LcmStepPushConstants>>
createLcmStep(VulkanContext& vulkanContext, const LcmScheduler& scheduler, size_t index, float scale,
              ComputeGraphElementPtr latents, ComputeGraphElementPtr prediction, const LcmInpaintingBuffers& buffers,
              ComputeGraphElementPtr timestep, ComputeGraphElementPtr decoderInput) {
    const auto& steps = scheduler.getSteps();
    const auto& current = steps.at(index);
    const bool last = index + 1 == steps.size();
    const auto& next = last ? current : steps[index + 1];
    auto element =
        std::make_shared<GeneralComputation<LcmStepPushConstants>>(vulkanContext, "shaders/sd15/lcm_step.comp.spv");
    element->setName("LcmStep" + std::to_string(index));
    element->setInput(latents, 0);
    element->setInput(prediction, 1);
    element->setInput(buffers.known, 2);
    element->setInput(buffers.mask, 3);
    element->setInput(buffers.noise, 4);
    element->setInput(timestep, 5);
    element->setInput(decoderInput, 6);
    element->setPushConstants({{buffers.count(), buffers.plane, uint32_t(last ? index : index + 1), last ? 1u : 0u,
                                std::sqrt(current.alpha), std::sqrt(1.0f - current.alpha), std::sqrt(current.alphaNext),
                                std::sqrt(1.0f - current.alphaNext), current.cSkip, current.cOut, std::sqrt(next.alpha),
                                std::sqrt(1.0f - next.alpha), scale, low(next.timestep), high(next.timestep)}});
    element->setGroupCount((buffers.count() + kGroupSize - 1) / kGroupSize, 1, 1);
    return element;
}

std::shared_ptr<GeneralComputation<ImageTensorRangePushConstants>>
createImageToTensor(VulkanContext& vulkanContext, ComputeGraphElementPtr image, VkImageLayout layout,
                    ComputeGraphElementPtr tensor, VkExtent2D extent, float low, float high) {
    auto element = std::make_shared<GeneralComputation<ImageTensorRangePushConstants>>(
        vulkanContext, "shaders/sd15/image_to_tensor_range.comp.spv");
    element->setName("ImageToTensor");
    element->setInput(image, 0);
    element->setInput(tensor, 1);
    readsImage(*element, 0, layout);
    element->setPushConstants({{extent.width, extent.height, low, high}});
    element->setGroupCount(groups(extent.width, 8), groups(extent.height, 8), 1);
    return element;
}

std::shared_ptr<GeneralComputation<LatentMaskPushConstants>>
createCloseUpLatentMask(VulkanContext& vulkanContext, ComputeGraphElementPtr marginImage, VkImageLayout marginLayout,
                        ComputeGraphElementPtr bodyImage, VkImageLayout bodyLayout, ComputeGraphElementPtr latentMask,
                        VkExtent2D extent, uint32_t factor, uint32_t dilation) {
    if (factor == 0 || extent.width % factor != 0 || extent.height % factor != 0) {
        throw std::runtime_error("createCloseUpLatentMask: the image size must be a multiple of the factor");
    }
    auto element = std::make_shared<GeneralComputation<LatentMaskPushConstants>>(vulkanContext,
                                                                                 "shaders/sd15/latent_mask.comp.spv");
    element->setName("LatentMask");
    element->setInput(marginImage, 0);
    element->setInput(bodyImage, 1);
    element->setInput(latentMask, 2);
    readsImage(*element, 0, marginLayout);
    readsImage(*element, 1, bodyLayout);
    element->setPushConstants({{extent.width, extent.height, factor, dilation}});
    element->setGroupCount(groups(extent.width / factor, 8), groups(extent.height / factor, 8), 1);
    return element;
}

void CloseUpWindowElement::_record(VkCommandBuffer commandBuffer, uint32_t pathId) {
    if (!sceneCamera || !skeletonCamera || sceneCamera->getDescriptorSets().empty() ||
        skeletonCamera->getDescriptorSets().empty()) {
        throw std::runtime_error("CloseUpWindowElement: both cameras must be part of the compiled graph");
    }
    GeneralComputation<CloseUpWindowPushConstants>::_record(commandBuffer, pathId);
    VkMemoryBarrier toTransfer{};
    toTransfer.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    toTransfer.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                         &toTransfer, 0, nullptr, 0, nullptr);
    auto cameras = std::dynamic_pointer_cast<BufferElementInterface>(getInputElement(3));
    VkBufferCopy copy{0, 0, sizeof(CameraMVP)};
    vkCmdCopyBuffer(commandBuffer, cameras->getVkBuffer(pathId), sceneCamera->getVkBuffer(pathId), 1, &copy);
    copy.srcOffset = sizeof(CameraMVP);
    vkCmdCopyBuffer(commandBuffer, cameras->getVkBuffer(pathId), skeletonCamera->getVkBuffer(pathId), 1, &copy);
}

CloseUpWindowElements createCloseUpWindow(VulkanContext& vulkanContext, ComputeGraphElementPtr joints,
                                          uint32_t jointCount, std::shared_ptr<CameraUboType> fullCamera,
                                          const glm::mat4& characterToScene, float aspect,
                                          std::shared_ptr<CameraUboType> sceneCamera,
                                          std::shared_ptr<CameraUboType> skeletonCamera) {
    static_assert(sizeof(CameraMVP) == 208, "CameraMVP must match the shader's Camera struct");
    CloseUpWindowElements result;
    result.state = std::make_shared<BufferElementSinglePath<VulkanBuffer<float>>>(vulkanContext, 8);
    result.state->setName("CloseUpWindowState");
    result.state->getBuffer().memcopyFrom(std::vector<float>(8, 0.0f));
    auto cameras = std::make_shared<BufferElement<VulkanBuffer<float>>>(
        vulkanContext, uint32_t(2 * sizeof(CameraMVP) / sizeof(float)), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    cameras->setName("CloseUpCameras");
    result.element = std::make_shared<CloseUpWindowElement>(vulkanContext, "shaders/sd15/closeup_window.comp.spv");
    result.element->setName("CloseUpWindow");
    result.element->setInput(joints, 0);
    result.element->setInput(fullCamera, 1);
    result.element->setInput(result.state, 2);
    result.element->setInput(cameras, 3);
    result.element->setCameras(sceneCamera, skeletonCamera);
    result.element->setPushConstants({{characterToScene, aspect, jointCount}});
    result.element->setGroupCount(1, 1, 1);
    sceneCamera->addDependency(result.element);
    skeletonCamera->addDependency(result.element);
    return result;
}

std::shared_ptr<GeneralComputation<CloseUpPastePushConstants>>
createCloseUpPaste(VulkanContext& vulkanContext, ComputeGraphElementPtr generated, ComputeGraphElementPtr sceneImage,
                   VkImageLayout sceneLayout, ComputeGraphElementPtr bodyImage, VkImageLayout bodyLayout,
                   ComputeGraphElementPtr closeUp, uint32_t size) {
    auto element = std::make_shared<GeneralComputation<CloseUpPastePushConstants>>(
        vulkanContext, "shaders/sd15/closeup_paste.comp.spv");
    element->setName("CloseUpPaste");
    element->setInput(generated, 0);
    element->setInput(sceneImage, 1);
    element->setInput(bodyImage, 2);
    element->setInput(closeUp, 3);
    readsImage(*element, 1, sceneLayout);
    readsImage(*element, 2, bodyLayout);
    element->setPushConstants({{size}});
    element->setGroupCount(groups(size, 8), groups(size, 8), 1);
    return element;
}

std::shared_ptr<GeneralComputation<CloseUpCompositePushConstants>>
createCloseUpComposite(VulkanContext& vulkanContext, ComputeGraphElementPtr sceneImage, VkImageLayout sceneLayout,
                       ComputeGraphElementPtr maskImage, VkImageLayout maskLayout, ComputeGraphElementPtr closeUp,
                       ComputeGraphElementPtr state, ComputeGraphElementPtr frame, VkExtent2D extent, uint32_t size) {
    auto element = std::make_shared<GeneralComputation<CloseUpCompositePushConstants>>(
        vulkanContext, "shaders/sd15/closeup_composite.comp.spv");
    element->setName("CloseUpComposite");
    element->setInput(sceneImage, 0);
    element->setInput(maskImage, 1);
    element->setInput(closeUp, 2);
    element->setInput(state, 3);
    element->setInput(frame, 4);
    readsImage(*element, 0, sceneLayout);
    readsImage(*element, 1, maskLayout);
    element->setPushConstants({{extent.width, extent.height, size}});
    element->setGroupCount(groups(extent.width, 8), groups(extent.height, 8), 1);
    return element;
}

} // namespace klartraum
