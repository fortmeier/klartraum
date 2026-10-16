// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include <stdexcept>
#include <string>
#include <vector>

#include "klartraum/sd15/controlnet_denoiser.hpp"

namespace klartraum {

namespace {

constexpr VkBufferUsageFlags kInputUsage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

std::string residualName(uint32_t index) {
    return index + 1 < ControlNetDenoiser::kResidualCount ? "down_residual_" + std::to_string(index) : "mid_residual";
}

} // namespace

ControlNetDenoiser::ControlNetDenoiser(VulkanContext& vulkanContext, const std::filesystem::path& modelDirectory,
                                       uint32_t latentSize)
    : latentSize(latentSize) {
    const auto controlNetPath = modelDirectory / "sd15_controlnet.onnx";
    const auto unetPath = modelDirectory / "sd15_unet_control.onnx";
    for (const auto& path : {controlNetPath, unetPath}) {
        if (!std::filesystem::exists(path)) {
            throw std::runtime_error("Missing " + path.string() + "; run scripts/sd15_onnx/export_controlnet.py");
        }
    }

    sample =
        vulkanContext.create<TensorElement<float>>(std::vector<uint32_t>{1, 4, latentSize, latentSize}, kInputUsage);
    timestep = vulkanContext.create<TensorElement<int64_t>>(std::vector<uint32_t>{1}, kInputUsage);
    embeddings = vulkanContext.create<TensorElement<float>>(std::vector<uint32_t>{1, 77, 768}, kInputUsage);
    pose = vulkanContext.create<TensorElement<float>>(std::vector<uint32_t>{1, 3, latentSize * 8, latentSize * 8},
                                                      kInputUsage);

    controlNet = vulkanContext.create<OnnxNetwork>(controlNetPath.string());
    controlNet->setInputTensor("sample", sample);
    controlNet->setInputTensor("timestep", timestep);
    controlNet->setInputTensor("encoder_hidden_states", embeddings);
    controlNet->setInputTensor("controlnet_cond", pose);

    unet = vulkanContext.create<OnnxNetwork>(unetPath.string());
    unet->setInputTensor("sample", sample);
    unet->setInputTensor("timestep", timestep);
    unet->setInputTensor("encoder_hidden_states", embeddings);
    for (uint32_t index = 0; index < kResidualCount; ++index) {
        unet->setInputTensor(residualName(index), controlNet, static_cast<int>(index));
    }
    outputElements[0] = unet;
}

std::shared_ptr<TensorElement<float>> ControlNetDenoiser::getNoisePrediction() {
    auto output = std::dynamic_pointer_cast<TensorElement<float>>(unet->getOutputElement("noise_prediction"));
    if (!output) {
        throw std::runtime_error("ControlNetDenoiser: the UNet has no float output noise_prediction");
    }
    return output;
}

} // namespace klartraum
