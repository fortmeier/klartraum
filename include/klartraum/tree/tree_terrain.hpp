// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_TREE_TREE_TERRAIN_HPP
#define KLARTRAUM_TREE_TREE_TERRAIN_HPP

#include <memory>
#include <vector>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/tree/tree_model.hpp"
#include "klartraum/vulkan_gaussian_splatting_types.hpp"

namespace klartraum {

/// Statistics of the terrain's node selection (TreeTerrain::readStats()).
enum TreeTerrainStat : uint32_t {
    kTreeTerrainStatSelected = 0,      ///< nodes selected for drawing (also those beyond the capacity)
    kTreeTerrainStatCulled = 1,        ///< nodes outside the soil or the view
    kTreeTerrainStatQueueOverflow = 2, ///< nodes drawn coarser than wanted because a queue was full
    kTreeTerrainStatNodeOverflow = 3,  ///< selected nodes left out because the node buffer was full
};

/// How the terrain's quadtree is laid out; see TreeTerrainLod.
struct TreeTerrainLodConfig {
    uint32_t levels = 12;         ///< of the quadtree: finest nodes 1/2048 of the root
    uint32_t boundsLevels = 9;    ///< of the height bounds pyramid: 256 x 256 cells at its finest
    uint32_t quads = 32;          ///< per node side
    float rangeFactor = 6.0f;     ///< a node of size s is drawn up to rangeFactor * s from the camera
    float morphStart = 0.66f;     ///< within the band between the finer level's range and the node's own
    uint32_t nodeCapacity = 1024; ///< each holds (quads + 1)^2 vertices of 16 bytes
    uint32_t queueCapacity = 4096;
};

/// The quadtree for `terrain`: its root square covers the soil's bounding box.
TreeTerrainLod makeTreeTerrainLod(const TreeTerrainParams& terrain, const TreeTerrainLodConfig& config = {});

/**
 * @brief The heights' bounds (min, max) per cell of the pyramid that `lod` describes (treeTerrainBoundsIndex()).
 *
 * Each finest cell is sampled on a 3 x 3 grid and padded, for the
 * variation between the samples, by a quarter of its range and 0.002 units;
 * coarser cells hold the bounds of their four children.
 */
std::vector<glm::vec2> buildTerrainBounds(const TreeTerrainConfig& config);

/// A selected node: xy its lower corner (x, z), z its size, w its level.
using TreeTerrainNode = glm::vec4;

/**
 * @brief The selection of `tree_terrain_select.comp` on the CPU: from the root down, every node is culled, drawn
 *        or split by treeTerrainNodeAction(), with the camera at `eye`, the view-projection `viewProj` and
 *        distances times `zoom` (treeTerrainZoom() of the projection).
 *
 * The nodes come in the order of the levels, coarsest first; within a level the GPU's order may differ.
 */
std::vector<TreeTerrainNode> selectTerrainNodesReference(const TreeTerrainConfig& config,
                                                         const std::vector<glm::vec2>& bounds,
                                                         const glm::mat4& viewProj, glm::vec3 eye, float zoom = 1.0f);

/**
 * @brief The terrain in the pot with a quadtree level of detail (CDLOD, Strugar 2009), selected on the GPU every
 *        frame.
 *
 * `tree_terrain_select.comp` walks the quadtree from the root, level by
 * level in one workgroup, and writes the nodes to draw (TreeTerrainNode) and
 * an indirect draw of `6 * quads^2` vertices per node. Nodes farther from the
 * camera are coarser; each level's range is twice the finer level's, so
 * neighbouring nodes differ by at most one level. `tree_terrain_vertices.comp`
 * then computes the heights and normals of the selected nodes' vertices,
 * morphed towards the coarser level near each level's range so that levels
 * meet without cracks or popping (getVertices(), TreeTerrainRenderer draws
 * them).
 */
class TreeTerrain {
public:
    TreeTerrain(VulkanContext& vulkanContext, const TreeTerrainParams& terrain, std::shared_ptr<CameraUboType> camera,
                const TreeTerrainLodConfig& lodConfig = {});

    const TreeTerrainConfig& getConfig() const { return config; }
    const std::vector<glm::vec2>& getBounds() const { return bounds; }
    ComputeGraphElementPtr getSelect() const { return select; }
    /// The vertex pass, which runs after the selection; depend on this one.
    ComputeGraphElementPtr getElement() const { return vertices; }
    /// Per selected node (quads + 1)^2 vertices, rows along x: xyz the normal, w the height.
    BufferRef getVertices() const { return {vertices, 3}; }
    /// The same vertices' land cover (TreeLandCover::vegetation; rock is the rest to 1).
    BufferRef getCovers() const { return {vertices, 4}; }
    /// The selected nodes (TreeTerrainNode), the indirect draw arguments and the configuration buffer.
    BufferRef getNodes() const { return {vertices, 0}; }
    BufferRef getDrawArgs() const { return {select, 1}; }
    BufferRef getConfigBuffer() const { return {vertices, 1}; }
    /// The indirect dispatch of one workgroup per selected node, for passes over the nodes after getElement().
    std::shared_ptr<DispatchIndirectCommandBufferElement> getDispatchArgs() const { return dispatchArgs; }

    /// After a submission of `pathId`: the selected nodes as the GPU wrote them.
    std::vector<TreeTerrainNode> readNodes(uint32_t pathId = 0);
    /// After a submission of `pathId`: the statistics (TreeTerrainStat).
    std::vector<uint32_t> readStats(uint32_t pathId = 0);
    /// After a submission of `pathId`: the vertices of the selected nodes (see getVertices()).
    std::vector<glm::vec4> readVertices(uint32_t pathId = 0);
    /// After a submission of `pathId`: the land cover of the selected nodes' vertices (see getCovers()).
    std::vector<glm::vec4> readCovers(uint32_t pathId = 0);

private:
    TreeTerrainConfig config{};
    std::vector<glm::vec2> bounds;
    std::shared_ptr<GeneralComputation<uint32_t>> select;
    std::shared_ptr<GeneralComputation<uint32_t>> vertices;
    std::shared_ptr<BufferElement<VulkanBuffer<glm::vec4>>> nodes;
    std::shared_ptr<BufferElement<VulkanBuffer<glm::vec4>>> vertexData;
    std::shared_ptr<BufferElement<VulkanBuffer<glm::vec4>>> coverData;
    std::shared_ptr<DispatchIndirectCommandBufferElement> dispatchArgs;
    std::shared_ptr<BufferElement<VulkanBuffer<uint32_t>>> stats;
};

} // namespace klartraum

#endif // KLARTRAUM_TREE_TREE_TERRAIN_HPP
