// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_TREE_TREE_TILES_HPP
#define KLARTRAUM_TREE_TREE_TILES_HPP

#include <map>
#include <memory>
#include <vector>

#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/hostvalues.hpp"
#include "klartraum/tree/tree_placement.hpp"

namespace klartraum {

/// The tiles of the tree placement: squares of `cells` x `cells` cells, tile (tx, tz) holding the cells from
/// (tx * cells, tz * cells).
struct TreeTileLayout {
    uint32_t cells = 64;
    uint32_t slots = 256;   ///< tiles the pool holds
    uint32_t perFrame = 64; ///< tiles generated per frame at most
    float range = 0.5f;     ///< tiles closer to the camera than this (units) are kept in the pool
};

/**
 * @brief Which tiles of the placement are in the pool, decided on the CPU each frame.
 *
 * update() makes every tile that reaches into the soil and lies within
 * `range` of the camera resident: tiles already in the pool stay in their
 * slot; new ones take a free slot or the one of the tile used longest ago
 * that is out of range, at most `perFrame` per frame (the rest follow in
 * the next frames). Tiles in range that find no slot are counted.
 */
class TreeTileCache {
public:
    TreeTileCache(const TreeTerrainParams& terrain, const TreePlacementParams& placement, TreeTileLayout layout);

    struct Tile {
        glm::ivec2 index;
        uint32_t slot;
    };

    /// Updates the pool for the camera at `eye`; returns the tiles to generate this frame.
    std::vector<Tile> update(glm::vec3 eye);

    /// The slot of tile `index`, or -1 when it is not in the pool.
    int slotOf(glm::ivec2 index) const;
    /// The tiles within `range` of `eye` that reach into the soil.
    std::vector<glm::ivec2> tilesInRange(glm::vec3 eye) const;
    /// Per slot the tile it holds (index; w 1 when it holds one), as the cull pass reads them.
    const std::vector<glm::ivec4>& getSlots() const { return slots; }
    /// Tiles in range without a slot or still to be generated, in the last update.
    uint32_t getMissing() const { return missing; }
    float tileSize() const { return float(layout.cells) * placement.grid.x; }

private:
    TreeTerrainParams terrain;
    TreePlacementParams placement;
    TreeTileLayout layout;
    std::map<std::pair<int, int>, uint32_t> resident;
    std::vector<glm::ivec4> slots;
    std::vector<uint64_t> lastUsed;
    uint64_t frame = 0;
    uint32_t missing = 0;
};

/**
 * @brief The pool of placement tiles on the GPU: each frame `tree_place_tiles.comp` generates the tiles a
 *        TreeTileCache asks for into their slots (per slot cells^2 TreeSite, and bounds: lowest foot, highest top,
 *        number of sites).
 *
 * Call update() before each submission; it writes the tiles to generate and
 * the slot table (getSlotTable()) as HostValues.
 */
class TreePlacementTiles {
public:
    TreePlacementTiles(VulkanContext& vulkanContext, const TreeTerrainParams& terrain,
                       const TreePlacementParams& placement, TreeTileLayout layout = {});

    void update(glm::vec3 eye);

    ComputeGraphElementPtr getElement() const { return generate; }
    BufferRef getSites() const { return {generate, 2}; }
    BufferRef getBounds() const { return {generate, 3}; }
    BufferRef getConfig() const { return {generate, 0}; }
    std::shared_ptr<HostValues<glm::ivec4>> getSlotTable() const { return slotTable; }
    TreeTileCache& getCache() { return cache; }
    const TreeTileCache& getCache() const { return cache; }
    const TreeTileConfig& getTileConfig() const { return config; }

    /// After a submission of `pathId`: the sites of `slot`.
    std::vector<TreeSite> readSites(uint32_t slot, uint32_t pathId = 0);
    /// After a submission of `pathId`: the bounds of every slot.
    std::vector<glm::vec4> readBounds(uint32_t pathId = 0);

private:
    TreeTileConfig config{};
    TreeTileCache cache;
    std::shared_ptr<HostValues<glm::ivec4>> generateList;
    std::shared_ptr<HostValues<glm::ivec4>> slotTable;
    std::shared_ptr<BufferElementSinglePath<VulkanBuffer<TreeSite>>> sites;
    std::shared_ptr<BufferElementSinglePath<VulkanBuffer<glm::vec4>>> bounds;
    std::shared_ptr<GeneralComputation<uint32_t>> generate;
};

} // namespace klartraum

#endif // KLARTRAUM_TREE_TREE_TILES_HPP
