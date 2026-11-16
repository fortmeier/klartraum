// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_TREE_TREE_TERRAIN_GAUSSIANS_HPP
#define KLARTRAUM_TREE_TREE_TERRAIN_GAUSSIANS_HPP

#include <memory>
#include <vector>

#include "klartraum/computegraph/hostvalues.hpp"
#include "klartraum/tree/tree_terrain.hpp"
#include "klartraum/rendering/vulkan_gaussian_splatting_types.hpp"

namespace klartraum {

/**
 * @brief The selected nodes of a TreeTerrain as Gaussians, every frame: a flat Gaussian on each block of 2 x 2
 *        quads of a node's grid, coloured with the classical terrain's materials and lit like the trees' Gaussians.
 *
 * They go into `poolSlots` slots from `poolOffset` of `gaussians` (the
 * extra slots of a TreeGaussians), which `tree_lod_clear.comp` makes
 * transparent first, so slots no node takes stay invisible. Nodes that do
 * not fit ((quads / 2)^2 slots each) are left out and counted (readOverflow()).
 * getGaussians() are the Gaussians as the write pass leaves them.
 */
class TreeTerrainGaussians {
public:
    TreeTerrainGaussians(VulkanContext& vulkanContext, TreeTerrain& terrain, const GaussianSoABuffers& gaussians,
                         uint32_t poolOffset, uint32_t poolSlots,
                         std::shared_ptr<HostValues<TreeRenderSettings>> settings,
                         std::shared_ptr<CameraUboType> camera, BufferRef light = {});

    const GaussianSoABuffers& getGaussians() const { return gaussians; }
    ComputeGraphElementPtr getElement() const { return write; }

    /// After a submission of `pathId`: the nodes left out for lack of room.
    uint32_t readOverflow(uint32_t pathId = 0);

private:
    std::shared_ptr<GeneralComputation<glm::uvec2>> clear;
    std::shared_ptr<GeneralComputation<glm::uvec2>> write;
    std::shared_ptr<BufferElement<VulkanBuffer<uint32_t>>> stats;
    GaussianSoABuffers gaussians;
};

} // namespace klartraum

#endif // KLARTRAUM_TREE_TREE_TERRAIN_GAUSSIANS_HPP
