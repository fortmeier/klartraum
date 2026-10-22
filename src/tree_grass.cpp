// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/tree/tree_grass.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace klartraum {

TreeGrassParams makeTreeGrass(const BonsaiScene& scene, TreeGrassConfig config) {
    if (config.cellMeters <= 0.0f || config.radiusMeters < config.cellMeters || config.bladesNear < 1.0f) {
        throw std::invalid_argument("makeTreeGrass: needs a positive cell, a radius beyond it and blades");
    }
    const float mpu = scene.config.scale.metersPerUnit();
    TreeGrassParams g{};
    g.grid = glm::vec4(config.cellMeters / mpu, config.radiusMeters / mpu, config.bladesNear, mpu);
    g.blade = glm::vec4(config.bladeMeters, config.widthMeters, 0.0f, 0.0f);
    g.counts = glm::uvec4(2u * uint32_t(std::ceil(config.radiusMeters / config.cellMeters)) + 2u, config.clumps,
                          config.seed, 0u);
    return g;
}

glm::ivec2 treeGrassGridOrigin(const TreeGrassParams& grass, glm::vec3 eye) {
    const int half = int(grass.counts.x / 2u);
    return {int(std::floor(eye.x / grass.grid.x)) - half + 1, int(std::floor(eye.z / grass.grid.x)) - half + 1};
}

std::vector<TreeGrassClump> treeGrassAround(const TreeTerrainParams& terrain, const TreeGrassParams& grass,
                                            glm::vec3 eye, float season) {
    const glm::ivec2 origin = treeGrassGridOrigin(grass, eye);
    std::vector<TreeGrassClump> clumps;
    for (uint32_t j = 0; j < grass.counts.x; ++j) {
        for (uint32_t i = 0; i < grass.counts.x; ++i) {
            const TreeGrassClump c =
                tree::treeGrassCell(terrain, grass, origin.x + int(i), origin.y + int(j), eye, season);
            if (c.kind.x > 0.0f) {
                clumps.push_back(c);
            }
        }
    }
    return clumps;
}

TreeGrass::TreeGrass(VulkanContext& vulkanContext, const TreeTerrainParams& terrain, const TreeGrassParams& grass,
                     std::shared_ptr<CameraUboType> camera, std::shared_ptr<HostValues<TreeRenderSettings>> settings,
                     VkExtent2D extent, TreeRenderInputs inputs)
    : params(grass),
      terrain(terrain) {
    const uint32_t capacity = grass.counts.y;
    auto setup = std::make_shared<BufferElementSinglePath<VulkanBuffer<TreeGrassSetup>>>(vulkanContext, 1u);
    setup->setName("TreeGrassSetup");
    setup->getBuffer().memcopyFrom(std::vector<TreeGrassSetup>{TreeGrassSetup{terrain, grass}});
    visible = std::make_shared<BufferElement<VulkanBuffer<TreeGrassVisible>>>(vulkanContext, capacity);
    visible->setName("TreeGrassVisible");
    counter = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, 4);
    counter->setName("TreeGrassCounter");
    counter->setRecordToZero(true);

    cull = std::make_shared<GeneralComputation<uint32_t>>(vulkanContext, "shaders/tree/tree_grass.comp.spv");
    cull->setName("TreeGrass");
    cull->setInput(setup, 0);
    cull->setInput(camera, 1);
    cull->setInput(settings, 2);
    cull->setInput(visible, 3);
    cull->setInput(counter, 4);
    frame = std::make_shared<HostValues<glm::vec4>>(
        vulkanContext, std::vector<glm::vec4>{glm::vec4(0.0f, 0.0f, 1.0f, 0.0f), glm::vec4(1.0f, 0.0f, 0.0f, 0.0f)});
    frame->setName("TreeGrassFrame");
    cull->setInput(frame, 5);
    const uint32_t groups = (grass.counts.x + 15u) / 16u;
    cull->setGroupCountX(groups);
    cull->setGroupCountY(groups);
    cull->setPushConstants({capacity});

    auto drawArgs = std::make_shared<DrawIndirectCommandBufferElement>(
        vulkanContext, 2u,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    drawArgs->setName("TreeGrassDrawArgs");
    dispatchArgs = std::make_shared<DispatchIndirectCommandBufferElement>(
        vulkanContext, 1u,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    dispatchArgs->setName("TreeGrassDispatchArgs");
    args = std::make_shared<GeneralComputation<glm::uvec2>>(vulkanContext, "shaders/tree/tree_place_args.comp.spv");
    args->setName("TreeGrassArgs");
    args->setInput(cull, 0, 4);
    args->setInput(drawArgs, 1);
    args->setInput(dispatchArgs, 2);
    args->setGroupCountX(1);
    args->setPushConstants({glm::uvec2(capacity, uint32_t(grass.grid.z) * kVerticesPerBlade)});

    StorageDrawConfig drawConfig;
    drawConfig.vertexShader = "shaders/tree/tree_grass.vert.spv";
    drawConfig.fragmentShader = "shaders/tree/tree_grass.frag.spv";
    drawConfig.extent = extent;
    draw = std::make_shared<StorageDrawComponent>(
        drawConfig, std::vector<BufferRef>{{cull, 3},
                                           {settings, -1},
                                           bufferOrPlaceholder(vulkanContext, inputs.sceneDepth),
                                           bufferOrPlaceholder(vulkanContext, inputs.light),
                                           {cull, 0},
                                           {cull, 5}});
    StorageDrawComponent::Draw d;
    d.indirect = {args, 1};
    draw->addDraw(std::move(d));
}

void TreeGrass::setWind(float seconds, float strength, glm::vec2 direction) {
    const glm::vec2 d = glm::length(direction) > 0.0f ? glm::normalize(direction) : glm::vec2(1.0f, 0.0f);
    frame->set(0, glm::vec4(seconds, strength, d.x, d.y));
}

void TreeGrass::setView(glm::vec3 eye) {
    const float above = eye.y - tree::treeTerrainHeight(terrain, eye.x, eye.z);
    frame->set(1, glm::vec4(above <= params.grid.y ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f));
}

std::vector<uint32_t> TreeGrass::readCounter(uint32_t pathId) {
    std::vector<uint32_t> values(4, 0u);
    counter->getBuffer(pathId).memcopyTo(values);
    return values;
}

std::vector<TreeGrassVisible> TreeGrass::readVisible(uint32_t pathId) {
    const uint32_t count = std::min(readCounter(pathId)[0], params.counts.y);
    std::vector<TreeGrassVisible> values(params.counts.y);
    visible->getBuffer(pathId).memcopyTo(values);
    values.resize(count);
    return values;
}

} // namespace klartraum
