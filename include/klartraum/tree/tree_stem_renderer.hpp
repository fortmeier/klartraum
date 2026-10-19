// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_TREE_TREE_STEM_RENDERER_HPP
#define KLARTRAUM_TREE_TREE_STEM_RENDERER_HPP

#include <memory>
#include <vector>

#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/hostvalues.hpp"
#include "klartraum/storage_draw_component.hpp"
#include "klartraum/tree/tree_generator.hpp"

namespace klartraum {

/// Indices into TreeStemRenderer's statistics buffer.
enum TreeStemStat : uint32_t {
    kTreeStemStatDrawn = 0,     ///< segments drawn
    kTreeStemStatTriangles = 1, ///< triangles of their tessellation (without the degenerate ones of the grid)
    kTreeStemStatCulled = 2,    ///< segments outside the view or thinner than the minimum radius
};

/**
 * @brief Draws the stem segments of a TreeGenerator as spline tubes, tessellated for the current view.
 *
 * `tree_stem_classify.comp` culls the segments and sorts the others into five
 * buckets by the number of quads they need (1x1 up to 32x16), writing one
 * indirect draw per bucket; the draw component (`tree_stem.vert`) builds each
 * segment's grid from its record. Add getClassify() as a compute dependency
 * and getDrawComponent() as a draw component of a RenderPass with a depth
 * attachment.
 */
class TreeStemRenderer {
public:
    TreeStemRenderer(VulkanContext& vulkanContext, TreeGenerator& generator, std::shared_ptr<CameraUboType> camera,
                     std::shared_ptr<HostValues<TreeRenderSettings>> settings, VkExtent2D extent);

    ComputeGraphElementPtr getClassify() const { return classify; }
    std::shared_ptr<StorageDrawComponent> getDrawComponent() const { return draw; }

    /// Reads the statistics (TreeStemStat) of path `pathId` after its submission finished.
    std::vector<uint32_t> readStats(uint32_t pathId = 0);

private:
    std::shared_ptr<GeneralComputation<uint32_t>> classify;
    std::shared_ptr<BufferElement<VulkanBuffer<uint32_t>>> lists;
    std::shared_ptr<DrawIndirectCommandBufferElement> drawArgs;
    std::shared_ptr<BufferElement<VulkanBuffer<uint32_t>>> stats;
    std::shared_ptr<StorageDrawComponent> draw;
};

} // namespace klartraum

#endif // KLARTRAUM_TREE_TREE_STEM_RENDERER_HPP
