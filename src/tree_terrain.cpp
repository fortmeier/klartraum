// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/tree/tree_terrain.hpp"

#include <algorithm>
#include <thread>

namespace klartraum {

TreeTerrainLod makeTreeTerrainLod(const TreeTerrainParams& terrain, const TreeTerrainLodConfig& config) {
    TreeTerrainLod lod{};
    const float size = 2.0f * std::max(terrain.outline.x, terrain.outline.y);
    lod.root = glm::vec4(-0.5f * size, -0.5f * size, size, 0.0f);
    lod.ranges = glm::vec4(config.rangeFactor, config.morphStart, float(config.quads), 0.0f);
    // Children enter a queue four at a time, so a queue holds a multiple of four.
    lod.levels = glm::uvec4(config.levels, std::min(config.boundsLevels, config.levels), config.nodeCapacity,
                            config.queueCapacity / 4u * 4u);
    return lod;
}

std::vector<glm::vec2> buildTerrainBounds(const TreeTerrainConfig& config) {
    const TreeTerrainLod& lod = config.lod;
    const uint32_t finest = lod.levels.y - 1u;
    const uint32_t cells = 1u << finest;
    const uint32_t samples = 2u * cells + 1u;
    const float step = lod.root.z / float(samples - 1u);
    std::vector<float> heights(size_t(samples) * samples);
    const uint32_t threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> workers;
    for (uint32_t t = 0; t < threads; ++t) {
        workers.emplace_back([&, t] {
            for (uint32_t j = t; j < samples; j += threads) {
                for (uint32_t i = 0; i < samples; ++i) {
                    heights[size_t(j) * samples + i] = tree::treeTerrainHeight(
                        config.terrain, lod.root.x + float(i) * step, lod.root.y + float(j) * step);
                }
            }
        });
    }
    for (std::thread& worker : workers) {
        worker.join();
    }

    const uint32_t total = ((1u << (2u * lod.levels.y)) - 1u) / 3u;
    std::vector<glm::vec2> bounds(total);
    const uint32_t finestOffset = ((1u << (2u * finest)) - 1u) / 3u;
    for (uint32_t cz = 0; cz < cells; ++cz) {
        for (uint32_t cx = 0; cx < cells; ++cx) {
            float lo = 1e30f, hi = -1e30f;
            for (uint32_t j = 0; j <= 2; ++j) {
                for (uint32_t i = 0; i <= 2; ++i) {
                    const float h = heights[size_t(2u * cz + j) * samples + 2u * cx + i];
                    lo = std::min(lo, h);
                    hi = std::max(hi, h);
                }
            }
            const float pad = 0.25f * (hi - lo) + 0.002f;
            bounds[finestOffset + cz * cells + cx] = glm::vec2(lo - pad, hi + pad);
        }
    }
    for (int p = int(finest) - 1; p >= 0; --p) {
        const uint32_t n = 1u << p;
        const uint32_t offset = ((1u << (2u * p)) - 1u) / 3u;
        const uint32_t childOffset = ((1u << (2u * (p + 1))) - 1u) / 3u;
        for (uint32_t cz = 0; cz < n; ++cz) {
            for (uint32_t cx = 0; cx < n; ++cx) {
                glm::vec2 b(1e30f, -1e30f);
                for (uint32_t k = 0; k < 4; ++k) {
                    const glm::vec2 c = bounds[childOffset + (2u * cz + k / 2u) * (2u * n) + 2u * cx + k % 2u];
                    b = glm::vec2(std::min(b.x, c.x), std::max(b.y, c.y));
                }
                bounds[offset + cz * n + cx] = b;
            }
        }
    }
    return bounds;
}

std::vector<TreeTerrainNode> selectTerrainNodesReference(const TreeTerrainConfig& config,
                                                         const std::vector<glm::vec2>& bounds,
                                                         const glm::mat4& viewProj, glm::vec3 eye) {
    std::vector<TreeTerrainNode> selected;
    std::vector<glm::uvec2> queue = {glm::uvec2(0u)};
    for (int level = int(config.lod.levels.x) - 1; level >= 0 && !queue.empty(); --level) {
        std::vector<glm::uvec2> next;
        const float size = tree::treeTerrainNodeSize(config.lod, uint32_t(level));
        for (const glm::uvec2& node : queue) {
            const glm::vec2 heights = bounds[tree::treeTerrainBoundsIndex(config.lod, node.x, node.y, uint32_t(level))];
            const uint32_t action =
                tree::treeTerrainNodeAction(config, viewProj, eye, node.x, node.y, uint32_t(level), heights);
            if (action == TREE_TERRAIN_DRAW) {
                selected.emplace_back(config.lod.root.x + float(node.x) * size,
                                      config.lod.root.y + float(node.y) * size, size, float(level));
            } else if (action == TREE_TERRAIN_SPLIT) {
                for (uint32_t k = 0; k < 4; ++k) {
                    next.emplace_back(2u * node.x + k % 2u, 2u * node.y + k / 2u);
                }
            }
        }
        queue = std::move(next);
    }
    return selected;
}

TreeTerrain::TreeTerrain(VulkanContext& vulkanContext, const TreeTerrainParams& terrain,
                         std::shared_ptr<CameraUboType> camera, const TreeTerrainLodConfig& lodConfig) {
    config.terrain = terrain;
    config.lod = makeTreeTerrainLod(terrain, lodConfig);
    bounds = buildTerrainBounds(config);

    auto configBuffer = std::make_shared<BufferElementSinglePath<VulkanBuffer<TreeTerrainConfig>>>(vulkanContext, 1u);
    configBuffer->setName("TreeTerrainConfig");
    configBuffer->getBuffer().memcopyFrom(std::vector<TreeTerrainConfig>{config});
    auto boundsBuffer =
        std::make_shared<BufferElementSinglePath<VulkanBuffer<glm::vec2>>>(vulkanContext, uint32_t(bounds.size()));
    boundsBuffer->setName("TreeTerrainBounds");
    boundsBuffer->getBuffer().memcopyFrom(bounds);
    nodes = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(vulkanContext, lodConfig.nodeCapacity);
    nodes->setName("TreeTerrainNodes");
    auto drawArgs = std::make_shared<DrawIndirectCommandBufferElement>(
        vulkanContext, 1u,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    drawArgs->setName("TreeTerrainDrawArgs");
    auto queues = std::make_shared<BufferElement<VulkanBuffer<glm::uvec2>>>(vulkanContext, 2u * lodConfig.queueCapacity,
                                                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                                            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    queues->setName("TreeTerrainQueues");
    stats = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, 4);
    stats->setName("TreeTerrainStats");
    stats->setRecordToZero(true);

    select = std::make_shared<GeneralComputation<uint32_t>>(vulkanContext, "shaders/tree/tree_terrain_select.comp.spv");
    select->setName("TreeTerrainSelect");
    select->setInput(nodes, 0);
    select->setInput(drawArgs, 1);
    select->setInput(queues, 2);
    select->setInput(configBuffer, 3);
    select->setInput(boundsBuffer, 4);
    select->setInput(camera, 5);
    select->setInput(stats, 6);
    select->setGroupCountX(1);
    select->setPushConstants({0u});
}

std::vector<TreeTerrainNode> TreeTerrain::readNodes(uint32_t pathId) {
    const uint32_t count = std::min(readStats(pathId)[kTreeTerrainStatSelected], config.lod.levels.z);
    std::vector<glm::vec4> values(config.lod.levels.z);
    nodes->getBuffer(pathId).memcopyTo(values);
    values.resize(count);
    return values;
}

std::vector<uint32_t> TreeTerrain::readStats(uint32_t pathId) {
    std::vector<uint32_t> values(4, 0u);
    stats->getBuffer(pathId).memcopyTo(values);
    return values;
}

} // namespace klartraum
