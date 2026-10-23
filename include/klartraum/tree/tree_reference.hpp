// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_TREE_TREE_REFERENCE_HPP
#define KLARTRAUM_TREE_TREE_REFERENCE_HPP

#include <cstdint>
#include <memory>
#include <vector>

#include <glm/glm.hpp>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/offscreen_target.hpp"
#include "klartraum/tree/bonsai_renderer.hpp"
#include "klartraum/tree/tree_leaf_renderer.hpp"
#include "klartraum/tree/tree_stem_renderer.hpp"

namespace klartraum {

struct TreeReferenceConfig {
    uint32_t batch = 256;        ///< trees generated per batch
    float rangeMeters = 1500.0f; ///< trees farther away are left out
    float minPixels = 1.0f;      ///< trees shorter on screen are left out
    /// The trees' seeds as the Gaussian renderer's (TreeGaussianForestConfig::variants), so that both draw the
    /// same trees.
    uint32_t variants = 8;
};

/**
 * @brief A high-fidelity rendering of a landscape's view: the classical renderer with every tree in view
 *        generated in full, for training and judging cheaper renderings.
 *
 * A classical BonsaiRenderer draws the view without its far trees (BonsaiRenderConfig::farTrees off, no forest
 * canopy) and keeps its depth buffer. The trees beyond its near ones that stand in the view, within rangeMeters
 * and at least minPixels tall, are then generated `batch` at a time with the classical stems and leaves (leaves
 * not thinned out) and drawn over the image with that depth buffer, one submission per batch, so that the batches
 * compose like one pass. The far trees are seeded as the Gaussian renderer's (`variants`) and lit without the near
 * trees' light grid.
 */
class TreeReferenceRenderer {
public:
    TreeReferenceRenderer(VulkanContext& vulkanContext, const BonsaiScene& scene, VkExtent2D extent,
                          TreeReferenceConfig config = {});

    struct Frame {
        std::vector<uint8_t> image; ///< BGRA
        std::vector<uint8_t> mask;  ///< per pixel 255 where the far trees changed the image, else 0
        uint32_t trees = 0;         ///< far trees drawn
        uint32_t batches = 0;
    };

    /// The view from `eye` towards `lookAt` (vertical field of view in degrees), in season `season` at `timeOfDay`,
    /// without wind.
    Frame render(glm::vec3 eye, glm::vec3 lookAt, float fovDegrees, float nearPlane, float farPlane, float season,
                 float timeOfDay);

    BonsaiRenderer& getBase() { return *base; }
    VkExtent2D getExtent() const { return extent; }

private:
    VulkanContext& vulkanContext;
    const BonsaiScene& scene;
    VkExtent2D extent;
    TreeReferenceConfig config;
    std::shared_ptr<OffscreenTarget> target;
    std::unique_ptr<BonsaiRenderer> base;
    std::unique_ptr<ComputeGraph> baseGraph;
    std::shared_ptr<TreeGenerator> generator;
    std::shared_ptr<HostValues<TreeRenderSettings>> settings;
    std::unique_ptr<TreeStemRenderer> stems;
    std::unique_ptr<TreeLeafRenderer> leaves;
    std::shared_ptr<RenderPass> pass;
    std::unique_ptr<ComputeGraph> batchGraph;
};

} // namespace klartraum

#endif // KLARTRAUM_TREE_TREE_REFERENCE_HPP
