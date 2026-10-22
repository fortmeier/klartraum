// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_TREE_TREE_LIBRARY_HPP
#define KLARTRAUM_TREE_TREE_LIBRARY_HPP

#include <memory>
#include <vector>

#include "klartraum/tree/tree_moss_lod.hpp"
#include "klartraum/tree/tree_proxies.hpp"

namespace klartraum {

/**
 * @brief Gaussians of each tree type per season sample, for drawing trees far away as copies (TreeGaussianForest).
 *
 * Patch `type * seasons.size() + k` is a tree of type `type` (reduced, treeReducedType() with 8 % of the leaves,
 * when `reduced`) at scale
 * 1 grown in season seasons[k], upright with its foot at the origin: a flat Gaussian per leaf, coloured with its
 * seasonal albedo, and Gaussians along the stems thicker than a thousandth of the tree's height, all lit by
 * their colours are albedos (Gaussian3D::color), lit every frame like the trees (TreeLodInstance::ids.w). `heights`
 * receives each type's height.
 */
std::vector<MossPatch> buildTreeGaussianLibrary(const std::vector<TreeTypeParams>& types,
                                                const std::vector<float>& seasons, std::vector<float>& heights,
                                                bool reduced = true);

/// The cut-off (log2 pixels, the lower edge of a size bin) from which on the trees counted in `bins` (TREE_SIZE_BINS,
/// treeSizeBin()) fit `copies`: the lowest bin edge for which the bins at or above it hold at most `copies` trees.
/// `taken` receives that number. What tree_library_cut.comp computes as its target.
float treeForestCutTarget(const std::vector<uint32_t>& bins, uint32_t copies, uint32_t* taken = nullptr);

/// The cut-off of the library copies after a frame (TreeGaussianForest::readCut()).
struct TreeForestCut {
    float cut = 0.0f;    ///< log2 pixels: copies fade in above it (treeCopyFade())
    float target = 0.0f; ///< this frame's target (treeForestCutTarget()), which the cut-off follows
    uint32_t taken = 0;  ///< trees at or above the target's bin
    std::vector<uint32_t> bins;
};

struct TreeGaussianForestConfig {
    std::vector<float> seasons{0.2f, 1.2f, 2.0f, 3.0f}; ///< at most 4
    /// Copies per frame, at most: the largest visible trees (TreeForestCut), the others fade out by size.
    uint32_t trees = 32768;
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
    TreeForestCut readCut(uint32_t pathId = 0);
    /// After a submission of `pathId`: the copies, one per entry of the visible list (scale 0: none).
    std::vector<TreeLodInstance> readInstances(uint32_t pathId = 0);

private:
    std::vector<float> heights;
    std::shared_ptr<BufferElement<VulkanBuffer<uint32_t>>> state;
    std::shared_ptr<BufferElement<VulkanBuffer<TreeLodInstance>>> instances;
    std::shared_ptr<GeneralComputation<uint32_t>> histogram;
    std::shared_ptr<GeneralComputation<uint32_t>> cutoff;
    std::shared_ptr<GeneralComputation<uint32_t>> convert;
    std::unique_ptr<TreeMossLod> lod;
};

} // namespace klartraum

#endif // KLARTRAUM_TREE_TREE_LIBRARY_HPP
