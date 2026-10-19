// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/tree/scene_mesh_renderer.hpp"

#include <stdexcept>

namespace klartraum {

SceneMeshRenderer::SceneMeshRenderer(VulkanContext& vulkanContext, const std::vector<TreeSceneVertex>& vertices,
                                     std::shared_ptr<HostValues<TreeRenderSettings>> settings, VkExtent2D extent,
                                     TreeRenderInputs inputs) {
    if (vertices.empty() || vertices.size() % 3 != 0) {
        throw std::invalid_argument("SceneMeshRenderer: needs a non-empty triangle list");
    }
    auto buffer = std::make_shared<BufferElementSinglePath<VulkanBuffer<TreeSceneVertex>>>(vulkanContext,
                                                                                           uint32_t(vertices.size()));
    buffer->setName("SceneMeshVertices");
    buffer->getBuffer().memcopyFrom(vertices);

    StorageDrawConfig config;
    config.vertexShader = "shaders/tree/scene_mesh.vert.spv";
    config.fragmentShader = "shaders/tree/scene_mesh.frag.spv";
    config.extent = extent;
    draw = std::make_shared<StorageDrawComponent>(
        config, std::vector<BufferRef>{{buffer, -1},
                                       {settings, -1},
                                       bufferOrPlaceholder(vulkanContext, inputs.sceneDepth),
                                       bufferOrPlaceholder(vulkanContext, inputs.light)});
    StorageDrawComponent::Draw d;
    d.vertexCount = uint32_t(vertices.size());
    draw->addDraw(std::move(d));
}

} // namespace klartraum
