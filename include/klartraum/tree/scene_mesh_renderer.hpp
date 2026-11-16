// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_TREE_SCENE_MESH_RENDERER_HPP
#define KLARTRAUM_TREE_SCENE_MESH_RENDERER_HPP

#include <memory>
#include <vector>

#include "klartraum/computegraph/hostvalues.hpp"
#include "klartraum/storage_draw_component.hpp"
#include "klartraum/tree/tree_model.hpp"
#include "klartraum/tree/tree_render_inputs.hpp"

namespace klartraum {

/**
 * @brief Draws a static triangle list of TreeSceneVertex with the lighting and materials of the tree renderers.
 *
 * The vertices are uploaded once into a buffer all paths share. Reads its
 * `inputs` like TreeStemRenderer.
 */
class SceneMeshRenderer {
public:
    SceneMeshRenderer(VulkanContext& vulkanContext, const std::vector<TreeSceneVertex>& vertices,
                      std::shared_ptr<HostValues<TreeRenderSettings>> settings, VkExtent2D extent,
                      TreeRenderInputs inputs = {});

    std::shared_ptr<StorageDrawComponent> getDrawComponent() const { return draw; }

private:
    std::shared_ptr<StorageDrawComponent> draw;
};

} // namespace klartraum

#endif // KLARTRAUM_TREE_SCENE_MESH_RENDERER_HPP
