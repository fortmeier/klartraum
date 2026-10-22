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
 * defaultTreeRenderSettings() (TreeMossLod tints them for the current light). `heights` receives each type's
 * height.
 */
std::vector<MossPatch> buildTreeGaussianLibrary(const std::vector<TreeTypeParams>& types,
                                                const std::vector<float>& seasons, std::vector<float>& heights,
                                                bool reduced = true);

struct TreeGaussianForestConfig {
    std::vector<float> seasons{0.2f, 1.2f, 2.0f, 3.0f}; ///< at most 4
    uint32_t trees = 32768;                             ///< copies per frame, at most
    /// Pixel diameters (of the crown) below which a tree uses its library levels 1, 2 and 3, and the crossfade band.
    glm::vec4 boundaries{480.0f, 160.0f, 60.0f, 0.15f};
};

/**
 * @brief The visible trees of TreeProxies as copies of a Gaussian library: each tree the patch of its type and the
 *        season sample nearest to the current season, turned and scaled to its site, with a level of detail by its
 *        size on screen (TreeMossLod) and faded as TreeProxies fades it.
 *
 * `tree_library_instances.comp` turns the visible sites into TreeLodInstance
 * every frame; the copies go into `poolSlots` slots from `poolOffset` of
 * `source`, after its writers.
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

private:
    std::vector<float> heights;
    std::shared_ptr<GeneralComputation<uint32_t>> convert;
    std::unique_ptr<TreeMossLod> lod;
};

} // namespace klartraum

#endif // KLARTRAUM_TREE_TREE_LIBRARY_HPP
