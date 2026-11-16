// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_TREE_TREE_LIBRARY_HPP
#define KLARTRAUM_TREE_TREE_LIBRARY_HPP

#include <array>
#include <memory>
#include <vector>

#include "klartraum/tree/tree_moss_lod.hpp"
#include "klartraum/tree/tree_proxies.hpp"

namespace klartraum {

/**
 * @brief Gaussians of each tree type per season sample, for drawing trees far away as copies (TreeGaussianForest).
 *
 * Patch `(type * variants + v) * seasons.size() + k` is a tree of type `type` (reduced, treeReducedType() with 8 %
 * of the leaves, when `reduced`) grown from seed v + 1 at scale 1 in season seasons[k], upright with its foot at
 * the origin: a flat Gaussian per leaf, coloured with its seasonal albedo, and Gaussians along the stems thicker
 * than a thousandth of the tree's height. Their colours are albedos (Gaussian3D::color), lit every frame like the
 * trees (TreeLodInstance::ids.w). `heights` receives each type's height, the largest of its variants.
 */
std::vector<MossPatch> buildTreeGaussianLibrary(const std::vector<TreeTypeParams>& types,
                                                const std::vector<float>& seasons, std::vector<float>& heights,
                                                bool reduced = true, uint32_t variants = 1);

/**
 * @brief The four levels of detail of a generated tree (its foot at the origin) as Gaussians, each within its
 *        budget, for TreeMossLod.
 *
 * A level spends up to a quarter of its budget on the stems thicker than a thousandth of the tree's height,
 * thickest first, with fewer, longer Gaussians on coarser levels. The rest goes to the leaves: those in a cell of
 * the finest grid whose occupied cells fit become one Gaussian at their middle, as wide as their spread (at most
 * three times as wide one way as the other, so nothing streaks), half as thick towards the outside of the crown
 * (its normal, for the lighting), with their mean albedo and as opaque as their area covers its face.
 */
GaussianLods buildTreeGaussianLevels(const TreeRecords& records, const TreeTypeParams& type,
                                     std::array<uint32_t, 4> budgets);

/// buildTreeGaussianLevels() of the full trees of buildTreeGaussianLibrary() (same order of patches); `heights`
/// receives each type's height, the largest of its variants.
std::vector<GaussianLods> buildTreeGaussianLibraryLevels(const std::vector<TreeTypeParams>& types,
                                                         const std::vector<float>& seasons, std::vector<float>& heights,
                                                         uint32_t variants, std::array<uint32_t, 4> budgets);

struct TreeGaussianForestConfig {
    std::vector<float> seasons{0.2f, 1.2f, 2.0f, 3.0f}; ///< at most 4
    /// Copies per frame, at most: the largest visible trees (TreeSizeCut), the others fade out by size.
    uint32_t trees = 32768;
    /// Trees per type, grown from the seeds 1 to `variants`; the near trees take the same seeds (treeSiteSeed()),
    /// so that a tree keeps its shape as it becomes a copy.
    uint32_t variants = 8;
    uint32_t seed = 7; ///< of the scene, for treeSiteSeed()
    /// Gaussians of a library tree's levels of detail 0 to 3 (buildTreeGaussianLevels()).
    std::array<uint32_t, 4> budgets{2560, 640, 160, 40};
    /// Pixel diameters (of the crown) below which a tree uses its library levels 1, 2 and 3, and the crossfade band.
    glm::vec4 boundaries{480.0f, 160.0f, 60.0f, 0.15f};
};

/**
 * @brief The visible trees of TreeProxies as copies of a Gaussian library: each tree the patch of its type and the
 *        season sample nearest to the current season, turned and scaled to its site, with a level of detail by its
 *        size on screen (TreeMossLod) and faded as TreeProxies fades it.
 *
 * Every frame `tree_library_histogram.comp` counts the visible trees by size,
 * `tree_library_cut.comp` finds the cut-off above which they fit `trees`
 * copies, and `tree_library_instances.comp` turns the visible sites above it
 * into TreeLodInstance, one slot per entry of the visible list; the copies go
 * into `poolSlots` slots from `poolOffset` of `source`, after its writers.
 * Which trees are copied thus depends on their sizes only, not on the order
 * of the visible list. Call initialize() once after the graph is compiled.
 */
class TreeGaussianForest {
public:
    TreeGaussianForest(VulkanContext& vulkanContext, TreeProxies& proxies, const GaussianSoABuffers& source,
                       uint32_t poolOffset, uint32_t poolSlots, const std::vector<TreeTypeParams>& types,
                       std::shared_ptr<CameraUboType> camera, std::shared_ptr<HostValues<TreeRenderSettings>> settings,
                       BufferRef light = {}, TreeGaussianForestConfig config = {});

    const GaussianSoABuffers& getGaussians() const { return lod->getGaussians(); }
    ComputeGraphElementPtr getElement() const { return lod->getElement(); }
    TreeMossLod& getLod() { return *lod; }
    const std::vector<float>& getHeights() const { return heights; }

    /// Starts the cut-off anew in every path; call after compiling.
    void initialize(uint32_t numberPaths);
    /// After a submission of `pathId`: the cut-off and the histogram it came from.
    TreeSizeCut readCut(uint32_t pathId = 0);
    /// After a submission of `pathId`: the copies, one per entry of the visible list (scale 0: none).
    std::vector<TreeLodInstance> readInstances(uint32_t pathId = 0);

private:
    std::vector<float> heights;
    std::shared_ptr<BufferElement<VulkanBuffer<uint32_t>>> state;
    std::shared_ptr<BufferElement<VulkanBuffer<TreeLodInstance>>> instances;
    std::shared_ptr<GeneralComputation<uint32_t>> histogram;
    std::shared_ptr<GeneralComputation<glm::uvec2>> cutoff;
    std::shared_ptr<GeneralComputation<uint32_t>> convert;
    std::unique_ptr<TreeMossLod> lod;
};

} // namespace klartraum

#endif // KLARTRAUM_TREE_TREE_LIBRARY_HPP
