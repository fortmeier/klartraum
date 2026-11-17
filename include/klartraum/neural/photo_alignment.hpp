// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_NEURAL_PHOTO_ALIGNMENT_HPP
#define KLARTRAUM_NEURAL_PHOTO_ALIGNMENT_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/vulkan_context.hpp"

namespace klartraum {
class OnnxNetwork;
}

namespace klartraum::neural {

/**
 * @brief How well a generated photograph follows the structure of its rendering
 * (scripts/neural/check_photo_alignment.py in Klartraum): the Spearman rank correlation between the rendering's inverse
 * depth and the inverse depth that Depth Anything V2 Metric Outdoor Small (an ONNX graph of
 * scripts/landscape/export_depth.py, which normalizes its input) estimates from the photograph, over the pixels the
 * rendering covers (every 7th of them). A photograph that invents its own scene scores low.
 */
class PhotoAlignment {
public:
    PhotoAlignment(
        VulkanContext& vulkanContext,
        const std::string& model = "build/models/depth/onnx/depth_anything_v2_metric_outdoor_small_518x924.onnx",
        uint32_t networkHeight = 518, uint32_t networkWidth = 924);
    ~PhotoAlignment();

    /**
     * @brief The score of a photograph [3][height][width] in [0, 1] against the rendering's stretched inverse depth
     * [height][width] (0 where it covers nothing); -1 if it covers fewer than 1000 pixels.
     */
    float score(const std::vector<float>& photo, const std::vector<float>& renderedDepth, uint32_t width,
                uint32_t height);

private:
    VulkanContext& vulkanContext;
    uint32_t networkHeight, networkWidth;
    std::shared_ptr<OnnxNetwork> network;
    std::shared_ptr<TensorElement<float>> input, output;
    std::unique_ptr<ComputeGraph> graph;
};

/// The Spearman rank correlation of two series (ranks by order, ties in order of appearance, as numpy's argsort).
float spearman(const std::vector<float>& a, const std::vector<float>& b);

} // namespace klartraum::neural

#endif // KLARTRAUM_NEURAL_PHOTO_ALIGNMENT_HPP
