// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_TREE_TREE_ENHANCER_HPP
#define KLARTRAUM_TREE_TREE_ENHANCER_HPP

#include <memory>
#include <string>

#include "klartraum/computegraph/computegraphelement.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/onnx/onnx_network.hpp"
#include "klartraum/vulkan_context.hpp"

namespace klartraum {

/**
 * @brief Runs the neural enhancer of distant trees (scripts/neural/train_tree_enhancer.py) after a Gaussian
 *        rendering: its colour image and feature images (BonsaiRenderer::getFeatureTarget()) in, the enhanced
 *        image out.
 *
 * `tree_enhancer_input.comp` packs the three images into the network's input tensor (8 planes, or 9 with feature B's
 * third channel, the splatted noise, for networks whose input has 9 channels), the ONNX network
 * runs, and `tensor_to_image.comp` writes its output into `output`. The images are OffscreenTargets of the
 * network's input size; after the enhancer they are all in TRANSFER_SRC_OPTIMAL again. `producer` is the element
 * that last writes the images (BonsaiRenderer::getOutput()); compile a graph from getOutput().
 */
class TreeEnhancer {
public:
    TreeEnhancer(VulkanContext& vulkanContext, const std::string& modelPath, std::shared_ptr<OffscreenTarget> color,
                 std::shared_ptr<OffscreenTarget> featureA, std::shared_ptr<OffscreenTarget> featureB,
                 std::shared_ptr<OffscreenTarget> output, ComputeGraphElementPtr producer);

    ComputeGraphElementPtr getOutput() const { return done; }
    /// The network, e.g. to read its output tensor ("output").
    std::shared_ptr<OnnxNetwork> getNetwork() const { return network; }

    struct Push {
        uint32_t width;
        uint32_t height;
    };
    struct InputPush {
        uint32_t width;
        uint32_t height;
        uint32_t channels;
        uint32_t pad;
    };

private:
    std::shared_ptr<TensorElement<float>> tensor;
    std::shared_ptr<GeneralComputation<InputPush>> input;
    std::shared_ptr<OnnxNetwork> network;
    std::shared_ptr<GeneralComputation<Push>> toImage;
    ComputeGraphElementPtr done;
};

} // namespace klartraum

#endif // KLARTRAUM_TREE_TREE_ENHANCER_HPP
