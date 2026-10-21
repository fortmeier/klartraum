// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/tree/tree_terrain_renderer.hpp"

namespace klartraum {

TreeTerrainRenderer::TreeTerrainRenderer(VulkanContext& vulkanContext, TreeTerrain& terrain,
                                         std::shared_ptr<HostValues<TreeRenderSettings>> settings, VkExtent2D extent,
                                         TreeRenderInputs inputs, glm::vec4 color) {
    StorageDrawConfig config;
    config.vertexShader = "shaders/tree/tree_terrain.vert.spv";
    config.fragmentShader = "shaders/tree/tree_terrain.frag.spv";
    config.extent = extent;
    config.pushConstantSize = sizeof(glm::vec4);
    draw = std::make_shared<StorageDrawComponent>(config, std::vector<BufferRef>{
                                                              terrain.getVertices(),
                                                              {settings, -1},
                                                              bufferOrPlaceholder(vulkanContext, inputs.sceneDepth),
                                                              bufferOrPlaceholder(vulkanContext, inputs.light),
                                                              terrain.getNodes(),
                                                              terrain.getConfigBuffer(),
                                                          });
    StorageDrawComponent::Draw d;
    d.indirect = terrain.getDrawArgs();
    d.pushConstants = StorageDrawComponent::bytes(color);
    draw->addDraw(std::move(d));
}

} // namespace klartraum
