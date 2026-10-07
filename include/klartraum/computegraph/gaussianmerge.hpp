// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_COMPUTEGRAPH_GAUSSIANMERGE_HPP
#define KLARTRAUM_COMPUTEGRAPH_GAUSSIANMERGE_HPP

#include <cstdint>
#include <memory>

#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/vulkan_gaussian_splatting_types.hpp"

namespace klartraum {

struct GaussianMergePushConstants {
    uint32_t countA;
    uint32_t countB;
};

/**
 * @brief Concatenates two sets of Gaussians every time the graph runs.
 *
 * A's Gaussians come first, so one backend renders and depth-sorts both
 * (shaders/gaussians/merge_gaussians.comp). The output is per-path.
 */
class GaussianMerge : public GeneralComputation<GaussianMergePushConstants> {
public:
    GaussianMerge(VulkanContext& vulkanContext)
        : GeneralComputation<GaussianMergePushConstants>(vulkanContext, "shaders/gaussians/merge_gaussians.comp.spv") {}

    const char* getType() const override { return "GaussianMerge"; }
};

struct GaussianMergeResult {
    std::shared_ptr<GaussianMerge> element;
    GaussianSoABuffers output; // inputs 14 .. 20 of the element
};

inline GaussianMergeResult createGaussianMerge(VulkanContext& vulkanContext, const GaussianSoABuffers& a,
                                               const GaussianSoABuffers& b) {
    auto element = std::make_shared<GaussianMerge>(vulkanContext);
    connectGaussians(*element, a, 0);
    connectGaussians(*element, b, 7);
    const uint32_t count = a.count + b.count;
    GaussianMergeResult result{element, addGaussianOutputs(vulkanContext, element, count, 14)};
    element->setPushConstants({{a.count, b.count}});
    element->setGroupCount((count + 255) / 256, 1, 1);
    return result;
}

} // namespace klartraum

#endif // KLARTRAUM_COMPUTEGRAPH_GAUSSIANMERGE_HPP
