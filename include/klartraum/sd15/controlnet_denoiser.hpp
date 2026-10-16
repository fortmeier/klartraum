// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_SD15_CONTROLNET_DENOISER_HPP
#define KLARTRAUM_SD15_CONTROLNET_DENOISER_HPP

#include <cstdint>
#include <filesystem>
#include <memory>

#include "klartraum/computegraph/computegraphgroup.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/onnx/onnx_network.hpp"

namespace klartraum {

/**
 * @brief One denoising evaluation of SD1.5 conditioned on a pose map: ControlNet-OpenPose followed by the UNet.
 *
 * Loads `sd15_controlnet.onnx` and `sd15_unet_control.onnx` exported by
 * scripts/sd15_onnx/export_controlnet.py. The ControlNet's twelve down-block
 * residuals and its mid-block residual feed the UNet inputs directly, in the
 * same graph. The UNet has LCM-LoRA fused in, so a few steps of LcmScheduler
 * without classifier-free guidance (batch one) produce an image.
 *
 * Inputs: sample [1, 4, h, w], timestep (int64) [1], encoder_hidden_states
 * [1, 77, 768] and the pose map [1, 3, 8h, 8w] with RGB in [0, 1]. Output:
 * getNoisePrediction() [1, 4, h, w]. The inputs are tensors the graph does not
 * write, so other graphs can fill them on the GPU: e.g. a graph with an
 * OpenPoseSkeletonRenderer pass followed by image_to_tensor can write the pose
 * map once per frame, while this graph runs once per denoising step.
 */
class ControlNetDenoiser : public ComputeGraphGroup {
public:
    static constexpr uint32_t kResidualCount = 13;

    ControlNetDenoiser(VulkanContext& vulkanContext, const std::filesystem::path& modelDirectory, uint32_t latentSize);

    virtual const char* getType() const override { return "ControlNetDenoiser"; }

    std::shared_ptr<TensorElement<float>> getSample() { return sample; }
    std::shared_ptr<TensorElement<int64_t>> getTimestep() { return timestep; }
    std::shared_ptr<TensorElement<float>> getEmbeddings() { return embeddings; }
    std::shared_ptr<TensorElement<float>> getPose() { return pose; }
    std::shared_ptr<TensorElement<float>> getNoisePrediction();

    std::shared_ptr<OnnxNetwork> getControlNet() { return controlNet; }
    std::shared_ptr<OnnxNetwork> getUnet() { return unet; }

    uint32_t getLatentSize() const { return latentSize; }

private:
    uint32_t latentSize;
    std::shared_ptr<TensorElement<float>> sample;
    std::shared_ptr<TensorElement<int64_t>> timestep;
    std::shared_ptr<TensorElement<float>> embeddings;
    std::shared_ptr<TensorElement<float>> pose;
    std::shared_ptr<OnnxNetwork> controlNet;
    std::shared_ptr<OnnxNetwork> unet;
};

} // namespace klartraum

#endif // KLARTRAUM_SD15_CONTROLNET_DENOISER_HPP
