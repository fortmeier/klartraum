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

} // namespace klartraum
