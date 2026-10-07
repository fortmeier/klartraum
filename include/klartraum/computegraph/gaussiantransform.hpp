// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_COMPUTEGRAPH_GAUSSIANTRANSFORM_HPP
#define KLARTRAUM_COMPUTEGRAPH_GAUSSIANTRANSFORM_HPP

#include <cstdint>
#include <memory>

#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/vulkan_gaussian_splatting_types.hpp"

namespace klartraum {

struct GaussianTransformPushConstants {
    uint32_t count;
};

/**
 * @brief Writes transformed copies of Gaussians every time the graph runs.
 *
 * Positions, orientations and sizes move by a transform buffer (see
 * TransformBuffer), SH coefficients rotate with the scene
 * (shaders/gaussians/transform_gaussians.comp). The output is per-path, so a
 * frame in flight keeps its Gaussians while the next one is written.
 */
class GaussianTransform : public GeneralComputation<GaussianTransformPushConstants> {
public:
    GaussianTransform(VulkanContext& vulkanContext)
        : GeneralComputation<GaussianTransformPushConstants>(vulkanContext,
                                                             "shaders/gaussians/transform_gaussians.comp.spv") {}

    const char* getType() const override { return "GaussianTransform"; }
};

/**
 * @brief Result of createGaussianTransform(): the element and the Gaussians it outputs.
 */
struct GaussianTransformResult {
    std::shared_ptr<GaussianTransform> element;
    GaussianSoABuffers output; // inputs 8 .. 14 of the element
};

/**
 * @brief Creates a GaussianTransform that applies @p transform to the Gaussians in @p source.
 */
inline GaussianTransformResult createGaussianTransform(VulkanContext& vulkanContext, const GaussianSoABuffers& source,
                                                       const BufferRef& transform) {
    auto element = std::make_shared<GaussianTransform>(vulkanContext);
    connectGaussians(*element, source, 0);
    transform.connectTo(*element, 7);
    GaussianTransformResult result{element, addGaussianOutputs(vulkanContext, element, source.count, 8)};
    element->setPushConstants({{source.count}});
    element->setGroupCount((source.count + 255) / 256, 1, 1);
    return result;
}

} // namespace klartraum

#endif // KLARTRAUM_COMPUTEGRAPH_GAUSSIANTRANSFORM_HPP
