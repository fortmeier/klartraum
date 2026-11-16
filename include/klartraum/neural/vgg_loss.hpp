// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_NEURAL_VGG_LOSS_HPP
#define KLARTRAUM_NEURAL_VGG_LOSS_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "klartraum/computegraph/computegraphelement.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/vulkan_context.hpp"

namespace klartraum::neural {

/**
 * @brief The perceptual loss of the photo enhancer (train_photo_enhancer.Vgg): images and targets in [0, 1], normalized
 * as for ImageNet, through VGG-16's convolutions up to relu4_3 (frozen, scripts/neural/export_vgg16.py), and the mean
 * absolute differences of their relu1_2, relu2_2, relu3_3 and relu4_3 features weighted 1/32, 1/16, 1/8 and 1/4,
 * all times `weight`. Its gradient is added to the images' gradient, through VGG's ReLUs, max-pools and convolutions.
 *
 * The elements go into the owner's graph in the order of elements(), after the image and the image gradient's first
 * writer; call upload() once the graph is compiled.
 */
class VggLoss {
public:
    VggLoss(VulkanContext& vulkanContext, const std::string& directory, uint32_t batch, uint32_t height, uint32_t width,
            float weight, std::shared_ptr<TensorElement<float>> images, std::shared_ptr<TensorElement<float>> targets,
            std::shared_ptr<TensorElement<float>> imageGradient);

    const std::vector<ComputeGraphElementPtr>& elements() const { return sequence; }
    /// Uploads VGG's weights; the graph must be compiled.
    void upload();
    /// The loss of the last submission.
    float loss() const;

private:
    struct Path;

    std::shared_ptr<TensorElement<float>> tensor(const std::vector<uint32_t>& shape);
    void forward(Path& path, std::shared_ptr<TensorElement<float>> images);

    VulkanContext& vulkanContext;
    std::string directory;
    uint32_t batch, height, width;
    std::vector<ComputeGraphElementPtr> sequence;
    /// The convolutions' weights and biases, and their channels and whether a max-pool follows.
    std::vector<std::shared_ptr<TensorElement<float>>> weights, biases;
    std::vector<uint32_t> inputChannels, outputChannels;
    std::vector<bool> pooled;
    /// The four features' shares of the loss and their sums.
    std::vector<std::shared_ptr<TensorElement<float>>> totals;
};

} // namespace klartraum::neural

#endif // KLARTRAUM_NEURAL_VGG_LOSS_HPP
