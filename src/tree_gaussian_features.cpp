// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/tree/tree_gaussian_features.hpp"

#include <glm/glm.hpp>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/gaussian_splatting_factory.hpp"

namespace klartraum {

namespace {

/// The push constants of tree_pixel_noise.comp.
struct PixelNoisePush {
    uint32_t width;
    uint32_t height;
    float cellUnits;
    uint32_t pad;
};

/// The finest cell of the noise per pixel, in metres.
constexpr float kPixelNoiseCellMeters = 0.02f;

} // namespace

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
        // Feature B's splatting also gives the depth for the noise per pixel, of translucent foliage too.
        GsplatConfig config = featureConfig;
        config.outputDepth = k == 1;
        config.depthOpacity = 0.1f;
        auto splatting =
            createGaussianSplatting(vulkanContext, backend, result.targets[k], camera, featureSplats, config);
        if (previous) {
            splatting->addDependency(previous);
        }
        previous = splatting;
    }

    // Noise per pixel of the world positions (tree_pixel_noise.comp) into feature B's third channel.
    auto noise =
        std::make_shared<GeneralComputation<PixelNoisePush>>(vulkanContext, "shaders/tree/tree_pixel_noise.comp.spv");
    noise->setName("TreePixelNoise");
    noise->setInput(getGaussianSplattingDepth(previous), 0);
    noise->setInput(camera, 1);
    noise->setInput(result.targets[1], 2);
    noise->setImageLayoutTransition(2, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                                    VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                    VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    noise->setPushConstants({PixelNoisePush{extent.width, extent.height, kPixelNoiseCellMeters * ranges.depthUnit, 0}});
    noise->setGroupCount((extent.width + 7) / 8, (extent.height + 7) / 8, 1);
    noise->addDependency(previous);
    auto back = std::make_shared<ImageViewSrcTransition>(VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    back->setName("TreePixelNoiseDone");
    back->setInput(noise, 0, 2);
    back->addDependency(noise);
    result.done = back;
    return result;
}

} // namespace klartraum
