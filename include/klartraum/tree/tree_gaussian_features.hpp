// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_TREE_TREE_GAUSSIAN_FEATURES_HPP
#define KLARTRAUM_TREE_TREE_GAUSSIAN_FEATURES_HPP

#include <array>
#include <cstdint>
#include <memory>

#include "klartraum/computegraph/computegraphelement.hpp"
#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/vulkan_context.hpp"
#include "klartraum/vulkan_gaussian_splatting_types.hpp"

namespace klartraum {

/// The push constants of tree_gaussian_features.comp.
struct GaussianFeaturePush {
    uint32_t count = 0;
    uint32_t treeBegin = 0; ///< the generated trees' slots [treeBegin, treeEnd)
    uint32_t treeEnd = 0;
    uint32_t copyBegin = 0; ///< the library copies' slots [copyBegin, copyEnd)
    uint32_t copyEnd = 0;
    float halfHeight = 0.0f; ///< half the image's height in pixels
    float depthUnit = 1.0f;  ///< units in which the depth is encoded (a metre)
    uint32_t pad = 0;
};

struct GaussianFeatures {
    /// 0: coverage, tree, level of detail; 1: Gaussian size, depth (see tree_gaussian_features.comp).
    std::array<std::shared_ptr<OffscreenTarget>, 2> targets;
    ComputeGraphElementPtr done; ///< the last feature splatting, after `after`
};

/**
 * @brief Splats `splats` again with feature colours (tree_gaussian_features.comp) into two images of `extent`,
 *        after the element `after` (e.g. the splatting of their colours).
 *
 * `ranges` gives the slots of the generated trees and the library copies and the depth unit; its count and
 * halfHeight are filled in.
 */
GaussianFeatures splatGaussianFeatures(VulkanContext& vulkanContext, const GaussianSoABuffers& splats,
                                       std::shared_ptr<CameraUboType> camera, VkExtent2D extent, uint32_t numberPaths,
                                       GaussianFeaturePush ranges, ComputeGraphElementPtr after,
                                       GsplatBackend backend = GsplatBackend::Raster);

} // namespace klartraum

#endif // KLARTRAUM_TREE_TREE_GAUSSIAN_FEATURES_HPP
