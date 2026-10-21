// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_TREE_TREE_TERRAIN_RENDERER_HPP
#define KLARTRAUM_TREE_TREE_TERRAIN_RENDERER_HPP

#include <memory>

#include "klartraum/computegraph/hostvalues.hpp"
#include "klartraum/storage_draw_component.hpp"
#include "klartraum/tree/tree_render_inputs.hpp"
#include "klartraum/tree/tree_terrain.hpp"

namespace klartraum {

/**
 * @brief Draws the selected nodes of a TreeTerrain classically: a grid per node inside the soil's outline, its
 *        materials blended by the land cover (treeTerrainAlbedo()), lit like the other scene surfaces.
 *
 * Add the terrain's getElement() as a compute dependency and
 * getDrawComponent() as a draw component of a RenderPass with a depth
 * attachment. Reads its `inputs` like TreeStemRenderer.
 */
class TreeTerrainRenderer {
public:
    TreeTerrainRenderer(VulkanContext& vulkanContext, TreeTerrain& terrain,
                        std::shared_ptr<HostValues<TreeRenderSettings>> settings, VkExtent2D extent,
                        TreeRenderInputs inputs = {});

    std::shared_ptr<StorageDrawComponent> getDrawComponent() const { return draw; }

private:
    std::shared_ptr<StorageDrawComponent> draw;
};

} // namespace klartraum

#endif // KLARTRAUM_TREE_TREE_TERRAIN_RENDERER_HPP
