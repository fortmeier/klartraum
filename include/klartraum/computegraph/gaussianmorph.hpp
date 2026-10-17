// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_COMPUTEGRAPH_GAUSSIANMORPH_HPP
#define KLARTRAUM_COMPUTEGRAPH_GAUSSIANMORPH_HPP

#include <cstdint>
#include <memory>
#include <vector>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/gaussian_morph.hpp"
#include "klartraum/vulkan_gaussian_splatting_types.hpp"

namespace klartraum {

struct GaussianMorphPushConstants {
    uint32_t count;
};

/**
 * @brief Blends one set of Gaussians into another every time the graph runs.
 *
 * Gaussian i of the start set turns into Gaussian i of the end set along the
 * path of a MorphVariant (shaders/gaussians/morph_gaussians.comp). The time
 * and the variant come from a MorphParams buffer (e.g. HostFloat), so the
 * graph is recorded once and only the values change per frame. At progress 0
 * the output is the start set, at 1 the end set, exactly. The output is
 * per-path.
 */
class GaussianMorph : public GeneralComputation<GaussianMorphPushConstants> {
public:
    GaussianMorph(VulkanContext& vulkanContext)
        : GeneralComputation<GaussianMorphPushConstants>(vulkanContext, "shaders/gaussians/morph_gaussians.comp.spv") {}

    const char* getType() const override { return "GaussianMorph"; }
};

/**
 * @brief Result of createGaussianMorph(): the element and the Gaussians it outputs.
 */
struct GaussianMorphResult {
    std::shared_ptr<GaussianMorph> element;
    GaussianSoABuffers output; // inputs 16 .. 22 of the element
};

/**
 * @brief Uploads `values` once into a device-local buffer shared by all paths.
 */
inline BufferRef uploadFloats(VulkanContext& vulkanContext, const std::vector<float>& values, const char* name) {
    auto element = std::make_shared<BufferElementSinglePath<VulkanBuffer<float>>>(
        vulkanContext, uint32_t(values.size()), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    element->setName(name);
    element->getBuffer().memcopyFrom(values);
    return BufferRef{element};
}

/**
 * @brief Creates a GaussianMorph from @p start to @p end.
 *
 * @p params holds MorphParams::kSize floats (MorphParams::values()),
 * @p delays one float per Gaussian in [0, 1] (see TextMorphStart::delays),
 * used by MorphVariant::Wave.
 */
inline GaussianMorphResult createGaussianMorph(VulkanContext& vulkanContext, const GaussianSoABuffers& start,
                                               const GaussianSoABuffers& end, const BufferRef& params,
                                               const BufferRef& delays) {
    if (start.count != end.count) {
        throw std::invalid_argument("createGaussianMorph: start and end need the same number of Gaussians");
    }
    auto element = std::make_shared<GaussianMorph>(vulkanContext);
    connectGaussians(*element, start, 0);
    connectGaussians(*element, end, 7);
    params.connectTo(*element, 14);
    delays.connectTo(*element, 15);
    GaussianMorphResult result{element, addGaussianOutputs(vulkanContext, element, start.count, 16)};
    element->setPushConstants({{start.count}});
    element->setGroupCount((start.count + 255) / 256, 1, 1);
    return result;
}

} // namespace klartraum

#endif // KLARTRAUM_COMPUTEGRAPH_GAUSSIANMORPH_HPP
