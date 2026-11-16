// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/tree/tree_tiles.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace klartraum {

TreeTileCache::TreeTileCache(const TreeTerrainParams& terrain, const TreePlacementParams& placement,
                             TreeTileLayout layout)
    : terrain(terrain),
      placement(placement),
      layout(layout),
      slots(layout.slots, glm::ivec4(0)),
      lastUsed(layout.slots, 0) {}

std::vector<glm::ivec2> TreeTileCache::tilesInRange(glm::vec3 eye) const {
    const float size = tileSize();
    const int x0 = int(std::floor((std::max(eye.x - layout.range, -terrain.outline.x)) / size));
    const int x1 = int(std::floor((std::min(eye.x + layout.range, terrain.outline.x)) / size));
    const int z0 = int(std::floor((std::max(eye.z - layout.range, -terrain.outline.y)) / size));
    const int z1 = int(std::floor((std::min(eye.z + layout.range, terrain.outline.y)) / size));
    std::vector<glm::ivec2> tiles;
    for (int tz = z0; tz <= z1; ++tz) {
        for (int tx = x0; tx <= x1; ++tx) {
            const glm::vec2 lo(tx * size, tz * size), hi = lo + glm::vec2(size);
            const glm::vec2 nearest = glm::clamp(glm::vec2(eye.x, eye.z), lo, hi);
            if (glm::length(nearest - glm::vec2(eye.x, eye.z)) > layout.range) {
                continue;
            }
            if (!tree::treeTerrainRectInSoil(terrain, lo, hi)) {
                continue;
            }
            tiles.emplace_back(tx, tz);
        }
    }
    // Nearest first, so that the closest tiles are generated first.
    std::sort(tiles.begin(), tiles.end(), [&](glm::ivec2 a, glm::ivec2 b) {
        const glm::vec2 e(eye.x / size - 0.5f, eye.z / size - 0.5f);
        return glm::length(glm::vec2(a) - e) < glm::length(glm::vec2(b) - e);
    });
    return tiles;
}

int TreeTileCache::slotOf(glm::ivec2 index) const {
    const auto it = resident.find({index.x, index.y});
    return it == resident.end() ? -1 : int(it->second);
}

std::vector<TreeTileCache::Tile> TreeTileCache::update(glm::vec3 eye) {
    ++frame;
    const std::vector<glm::ivec2> wanted = tilesInRange(eye);
    for (const glm::ivec2& t : wanted) {
        const int slot = slotOf(t);
        if (slot >= 0) {
            lastUsed[slot] = frame;
        }
    }
    std::vector<Tile> generate;
    missing = 0;
    for (const glm::ivec2& t : wanted) {
        if (slotOf(t) >= 0) {
            continue;
        }
        if (generate.size() >= layout.perFrame) {
            ++missing;
            continue;
        }
        // A free slot, else the one used longest ago by a tile out of range.
        uint32_t best = layout.slots;
        uint64_t oldest = std::numeric_limits<uint64_t>::max();
        for (uint32_t s = 0; s < layout.slots; ++s) {
            if (slots[s].w == 0) {
                best = s;
                break;
            }
            if (lastUsed[s] < frame && lastUsed[s] < oldest) {
                oldest = lastUsed[s];
                best = s;
            }
        }
        if (best == layout.slots) {
            ++missing;
            continue;
        }
        if (slots[best].w != 0) {
            resident.erase({slots[best].x, slots[best].y});
        }
        slots[best] = glm::ivec4(t.x, t.y, 0, 1);
        resident[{t.x, t.y}] = best;
        lastUsed[best] = frame;
        generate.push_back({t, best});
    }
    return generate;
}

TreePlacementTiles::TreePlacementTiles(VulkanContext& vulkanContext, const TreeTerrainParams& terrain,
                                       const TreePlacementParams& placement, TreeTileLayout layout)
    : cache(terrain, placement, layout) {
    config.terrain = terrain;
    config.placement = placement;
    config.tiles = glm::uvec4(layout.cells, layout.slots, layout.perFrame, 0u);

    auto configBuffer = std::make_shared<BufferElementSinglePath<VulkanBuffer<TreeTileConfig>>>(vulkanContext, 1u);
    configBuffer->setName("TreeTileConfig");
    configBuffer->getBuffer().memcopyFrom(std::vector<TreeTileConfig>{config});
    generateList =
        std::make_shared<HostValues<glm::ivec4>>(vulkanContext, std::vector<glm::ivec4>(layout.perFrame + 1u));
    generateList->setName("TreeTileGenerate");
    slotTable = std::make_shared<HostValues<glm::ivec4>>(vulkanContext, std::vector<glm::ivec4>(layout.slots));
    slotTable->setName("TreeTileSlots");
    // The pool persists from frame to frame, shared by all paths.
    sites = std::make_shared<BufferElementSinglePath<VulkanBuffer<TreeSite>>>(
        vulkanContext, layout.slots * layout.cells * layout.cells);
    sites->setName("TreeTileSites");
    bounds = std::make_shared<BufferElementSinglePath<VulkanBuffer<glm::vec4>>>(vulkanContext, layout.slots);
    bounds->setName("TreeTileBounds");

    generate = std::make_shared<GeneralComputation<uint32_t>>(vulkanContext, "shaders/tree/tree_place_tiles.comp.spv");
    generate->setName("TreePlaceTiles");
    generate->setInput(configBuffer, 0);
    generate->setInput(generateList, 1);
    generate->setInput(sites, 2);
    generate->setInput(bounds, 3);
    generate->setGroupCountX(layout.perFrame);
    generate->setPushConstants({0u});
}

void TreePlacementTiles::update(glm::vec3 eye) {
    const std::vector<TreeTileCache::Tile> tiles = cache.update(eye);
    std::vector<glm::ivec4> list(config.tiles.z + 1u, glm::ivec4(0));
    list[0].x = int(tiles.size());
    for (size_t i = 0; i < tiles.size(); ++i) {
        list[i + 1] = glm::ivec4(int(tiles[i].slot), tiles[i].index.x, tiles[i].index.y, 0);
    }
    generateList->set(list);
    slotTable->set(cache.getSlots());
}

std::vector<TreeSite> TreePlacementTiles::readSites(uint32_t slot, uint32_t pathId) {
    const uint32_t cells = config.tiles.x * config.tiles.x;
    std::vector<TreeSite> all(sites->getBuffer(pathId).getSize());
    sites->getBuffer(pathId).memcopyTo(all);
    return std::vector<TreeSite>(all.begin() + size_t(slot) * cells, all.begin() + size_t(slot + 1) * cells);
}

std::vector<glm::vec4> TreePlacementTiles::readBounds(uint32_t pathId) {
    std::vector<glm::vec4> values(bounds->getBuffer(pathId).getSize());
    bounds->getBuffer(pathId).memcopyTo(values);
    return values;
}

} // namespace klartraum
