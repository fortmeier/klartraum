// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_TREE_TREE_LEAF_RENDERER_HPP
#define KLARTRAUM_TREE_TREE_LEAF_RENDERER_HPP

#include <memory>
#include <vector>

#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/hostvalues.hpp"
#include "klartraum/storage_draw_component.hpp"
#include "klartraum/tree/tree_generator.hpp"

namespace klartraum {

/// Indices into TreeLeafRenderer's statistics buffer.
enum TreeLeafStat : uint32_t {
    kTreeLeafStatDrawn = 0,     ///< leaves drawn
    kTreeLeafStatTriangles = 1, ///< triangles of the drawn lobes
    kTreeLeafStatCulled = 2,    ///< leaves outside the view or thinned out
};

/**
 * @brief Draws the leaves of a TreeGenerator: Bezier lobes with exact edges and a continuous level of detail.
 *
 * `tree_leaf_classify.comp` culls the leaves and sorts them into a full and a
 * simplified draw bucket (see TreeRenderSettings::leafLod); `tree_leaf.vert`
 * builds the lobes and `tree_leaf.frag` cuts their curved edges. Use it like
 * TreeStemRenderer.
 */
class TreeLeafRenderer {
public:
    TreeLeafRenderer(VulkanContext& vulkanContext, TreeGenerator& generator, const std::vector<TreeTypeParams>& types,
                     std::shared_ptr<CameraUboType> camera, std::shared_ptr<HostValues<TreeRenderSettings>> settings,
                     VkExtent2D extent);

    ComputeGraphElementPtr getClassify() const { return classify; }
    std::shared_ptr<StorageDrawComponent> getDrawComponent() const { return draw; }

    /// Reads the statistics (TreeLeafStat) of path `pathId` after its submission finished.
    std::vector<uint32_t> readStats(uint32_t pathId = 0);

    /// The most lobes a leaf can have.
    static constexpr uint32_t kMaxLobes = 7;

private:
    std::shared_ptr<GeneralComputation<glm::uvec2>> classify;
    std::shared_ptr<BufferElement<VulkanBuffer<uint32_t>>> lists;
    std::shared_ptr<DrawIndirectCommandBufferElement> drawArgs;
    std::shared_ptr<BufferElement<VulkanBuffer<uint32_t>>> stats;
    std::shared_ptr<StorageDrawComponent> draw;
};

} // namespace klartraum

#endif // KLARTRAUM_TREE_TREE_LEAF_RENDERER_HPP
