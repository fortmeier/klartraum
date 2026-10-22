// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_TREE_TREE_FLOOR_HPP
#define KLARTRAUM_TREE_TREE_FLOOR_HPP

#include <memory>
#include <vector>

#include "klartraum/tree/tree_moss_lod.hpp"

namespace klartraum {

/// The forest floor near the camera, in metres of the landscape.
struct TreeFloorConfig {
    float radiusMeters = 12.0f; ///< no copies beyond
    float fadeFrom = 0.7f;      ///< share of the radius from which the copies fade out
    float spacing = 1.3f;       ///< cell size in patch radii (below 2 the copies overlap)
    uint32_t seed = 5;
};

/// The largest distance of a patch's Gaussians from its centre along the ground, in capture units.
float mossPatchRadius(const std::vector<MossPatch>& patches);

/// The parameters of treeFloorCell() for `scene`: copies of `patchCount` patches of `patchRadius` capture units,
/// enlarged by `scale` (as the planting's copies, with the lantern).
TreeFloorSetup makeTreeFloor(const BonsaiScene& scene, uint32_t patchCount, float patchRadius, float scale,
                             TreeFloorConfig config = {});

/// The first cell of the grid around `eye` along x and z.
glm::ivec2 treeFloorGridOrigin(const TreeFloorSetup& floor, glm::vec3 eye);

/// The copies of the cells around `eye` (the grid TreeForestFloor evaluates: counts.x cells per side, rows along x),
/// with scale 0 where there is none.
std::vector<TreeLodInstance> treeFloorAround(const TreeFloorSetup& floor, glm::vec3 eye);

/**
 * @brief Copies of captured moss patches on the forest floor near the camera, placed on the GPU every frame
 *        (treeFloorCell(), `tree_floor_place.comp`) and drawn by a TreeMossLod into `poolSlots` slots from
 *        `poolOffset` of `source`.
 */
class TreeForestFloor {
public:
    TreeForestFloor(VulkanContext& vulkanContext, const TreeFloorSetup& floor, const GaussianSoABuffers& source,
                    const std::vector<MossPatch>& patches, std::shared_ptr<CameraUboType> camera,
                    std::shared_ptr<HostValues<TreeRenderSettings>> settings, BufferRef light, glm::vec4 boundaries,
                    uint32_t poolSlots, uint32_t poolOffset);

    const GaussianSoABuffers& getGaussians() const { return lod->getGaussians(); }
    ComputeGraphElementPtr getElement() const { return lod->getElement(); }
    TreeMossLod& getLod() { return *lod; }
    const TreeFloorSetup& getSetup() const { return setup; }
    /// The camera of the next frames is at `eye`: no copies are placed while it is farther above the ground than
    /// the radius.
    void setView(glm::vec3 eye);
    /// After a submission of `pathId`: the copies of the grid (scale 0 where there is none).
    std::vector<TreeLodInstance> readInstances(uint32_t pathId = 0);

private:
    TreeFloorSetup setup;
    std::shared_ptr<GeneralComputation<uint32_t>> place;
    std::shared_ptr<BufferElement<VulkanBuffer<TreeLodInstance>>> instances;
    std::shared_ptr<HostValues<glm::vec4>> frame; ///< [0] x 1 while the floor is in reach
    std::unique_ptr<TreeMossLod> lod;
};

} // namespace klartraum

#endif // KLARTRAUM_TREE_TREE_FLOOR_HPP
