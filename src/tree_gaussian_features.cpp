// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/tree/tree_gaussian_features.hpp"

#include <glm/glm.hpp>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"

namespace klartraum {

GaussianFeatures splatGaussianFeatures(VulkanContext& vulkanContext, const GaussianSoABuffers& splats,
                                       std::shared_ptr<CameraUboType> camera, VkExtent2D extent, uint32_t numberPaths,
                                       GaussianFeaturePush ranges, ComputeGraphElementPtr after,
                                       GsplatBackend backend) {
    const uint32_t count = splats.count;
    auto features = std::make_shared<GeneralComputation<GaussianFeaturePush>>(
        vulkanContext, "shaders/tree/tree_gaussian_features.comp.spv");
    features->setName("GaussianFeatures");
    splats.pos.connectTo(*features, 0);
    splats.scale.connectTo(*features, 1);
    splats.colAlpha.connectTo(*features, 2);
    features->setInput(camera, 3);
    for (int k = 0; k < 2; ++k) {
        features->setInput(
            std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(
                vulkanContext, count, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
            4 + k);
    }
    features->setGroupCountX((count + 255) / 256);
    ranges.count = count;
    ranges.halfHeight = 0.5f * float(extent.height);
    features->setPushConstants({ranges});

    GaussianFeatures result;
    GsplatConfig featureConfig;
    featureConfig.shDegree = 0;
    ComputeGraphElementPtr previous = after;
    for (int k = 0; k < 2; ++k) {
        result.targets[k] = std::make_shared<OffscreenTarget>(vulkanContext, extent, numberPaths);
        GaussianSoABuffers featureSplats = splats;
        featureSplats.colAlpha = BufferRef{features, 4 + k};
        auto splatting =
            createGaussianSplatting(vulkanContext, backend, result.targets[k], camera, featureSplats, featureConfig);
        if (previous) {
            splatting->addDependency(previous);
        }
        previous = splatting;
    }
    result.done = previous;
    return result;
}

} // namespace klartraum
