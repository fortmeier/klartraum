// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_TREE_TREE_GRASS_HPP
#define KLARTRAUM_TREE_TREE_GRASS_HPP

#include <memory>
#include <vector>

#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/hostvalues.hpp"
#include "klartraum/storage_draw_component.hpp"
#include "klartraum/tree/bonsai_scene.hpp"
#include "klartraum/tree/tree_render_inputs.hpp"

namespace klartraum {

/// The grass near the camera, in metres of the landscape.
struct TreeGrassConfig {
    float cellMeters = 0.25f;   ///< a clump of blades per cell
    float radiusMeters = 40.0f; ///< no grass beyond
    float bladesNear = 12.0f;   ///< blades of a clump near the camera, fewer from a third of the radius on
    float bladeMeters = 0.35f;  ///< mean height of a blade
    float widthMeters = 0.012f; ///< of a blade at its foot
    uint32_t clumps = 262144;   ///< at most
    uint32_t seed = 3;
};

/// The parameters of treeGrassCell() for `scene` at its scale.
TreeGrassParams makeTreeGrass(const BonsaiScene& scene, TreeGrassConfig config = {});

/// The clumps of the cells around `eye` in season S (the grid TreeGrass evaluates: counts.x cells per side), in
/// the order of the cells (rows along x).
std::vector<TreeGrassClump> treeGrassAround(const TreeTerrainParams& terrain, const TreeGrassParams& grass,
                                            glm::vec3 eye, float season);

/// The first cell of the grid around `eye` along x and z.
glm::ivec2 treeGrassGridOrigin(const TreeGrassParams& grass, glm::vec3 eye);

/**
 * @brief The grass near the camera, drawn as blades.
 *
 * `tree_grass.comp` evaluates the grid of cells around the camera
 * (treeGrassCell()) every frame and appends the clumps in the view;
 * `tree_place_args.comp` writes the indirect draw of their blades
 * (getDrawComponent(): kVerticesPerBlade vertices per blade, grid.z blades
 * per clump). Add getElement() as a compute dependency.
 */
class TreeGrass {
public:
    TreeGrass(VulkanContext& vulkanContext, const TreeTerrainParams& terrain, const TreeGrassParams& grass,
              std::shared_ptr<CameraUboType> camera, std::shared_ptr<HostValues<TreeRenderSettings>> settings,
              VkExtent2D extent, TreeRenderInputs inputs = {});

    static constexpr uint32_t kVerticesPerBlade = 15;

    ComputeGraphElementPtr getElement() const { return args; }
    std::shared_ptr<StorageDrawComponent> getDrawComponent() const { return draw; }
    BufferRef getVisible() const { return {cull, 3}; }
    BufferRef getCounter() const { return {cull, 4}; }
    /// Clumps whose dispatch (64 per workgroup) follows the visible ones.
    std::shared_ptr<DispatchIndirectCommandBufferElement> getDispatchArgs() const { return dispatchArgs; }
    const TreeGrassParams& getParams() const { return params; }
    /// The wind at `seconds`: `strength` in [0, 1] towards `direction` (x, z).
    void setWind(float seconds, float strength, glm::vec2 direction = {1.0f, 0.0f});
    /// The camera of the next frames is at `eye`: no grass is evaluated while it is farther above the ground than
    /// the grass's radius.
    void setView(glm::vec3 eye);

    /// After a submission of `pathId`: [0] clumps visible, [1] clumps left out for lack of room.
    std::vector<uint32_t> readCounter(uint32_t pathId = 0);
    /// After a submission of `pathId`: the visible clumps.
    std::vector<TreeGrassVisible> readVisible(uint32_t pathId = 0);

private:
    TreeGrassParams params;
    TreeTerrainParams terrain;
    std::shared_ptr<GeneralComputation<uint32_t>> cull;
    std::shared_ptr<GeneralComputation<glm::uvec2>> args;
    std::shared_ptr<BufferElement<VulkanBuffer<TreeGrassVisible>>> visible;
    std::shared_ptr<BufferElement<VulkanBuffer<uint32_t>>> counter;
    std::shared_ptr<DispatchIndirectCommandBufferElement> dispatchArgs;
    std::shared_ptr<HostValues<glm::vec4>> frame; ///< [0] the wind, [1] x 1 while grass is in reach
    std::shared_ptr<StorageDrawComponent> draw;
};

} // namespace klartraum

#endif // KLARTRAUM_TREE_TREE_GRASS_HPP
