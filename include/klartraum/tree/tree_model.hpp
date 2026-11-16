// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_TREE_TREE_MODEL_HPP
#define KLARTRAUM_TREE_TREE_MODEL_HPP

#include <cstdint>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

namespace klartraum {

/**
 * @brief The procedural tree model of Kuth et al., "Real-Time GPU Tree Generation" (HPG 2025).
 *
 * The structs and functions in this namespace are compiled from
 * `tree_shared.hpp`, which the tree shaders include as GLSL, so the CPU and the
 * GPU share one definition of the model.
 */
namespace tree {

using glm::uvec4;
using glm::vec2;
using glm::vec3;
using glm::vec4;
using uint = uint32_t;
using namespace glm;

#define TREE_FN inline
#include "klartraum/tree/tree_shared.hpp"
#undef TREE_FN

} // namespace tree

using tree::LeafRecord;
using tree::SegmentRecord;
using tree::StemRecord;
using tree::TreeFloorSetup;
using tree::TreeFrame;
using tree::TreeGrassClump;
using tree::TreeGrassParams;
using tree::TreeGrassSetup;
using tree::TreeGrassVisible;
using tree::TreeImpostorParams;
using tree::TreeInstance;
using tree::TreeLandCover;
using tree::TreeLodInstance;
using tree::TreeLodPatch;
using tree::TreeLodSplat;
using tree::TreePlacementParams;
using tree::TreeRenderSettings;
using tree::TreeSceneVertex;
using tree::TreeSite;
using tree::TreeTerrainConfig;
using tree::TreeTerrainLod;
using tree::TreeTerrainParams;
using tree::TreeTileConfig;
using tree::TreeTypeParams;

static_assert(sizeof(TreeTypeParams) == 25 * 16, "TreeTypeParams must match its std430 layout");
static_assert(sizeof(TreeInstance) == 48, "TreeInstance must match its std430 layout");
static_assert(sizeof(StemRecord) == 64, "StemRecord must match its std430 layout");
static_assert(sizeof(SegmentRecord) == 96, "SegmentRecord must match its std430 layout");
static_assert(sizeof(LeafRecord) == 48, "LeafRecord must match its std430 layout");
static_assert(sizeof(TreeFrame) == 64, "TreeFrame must match its std430 layout");
static_assert(sizeof(TreeRenderSettings) == 288, "TreeRenderSettings must match its std430 layout");
static_assert(sizeof(TreeTerrainParams) == 80, "TreeTerrainParams must match its std430 layout");
static_assert(sizeof(TreeTerrainLod) == 64, "TreeTerrainLod must match its std430 layout");
static_assert(sizeof(TreePlacementParams) == 64, "TreePlacementParams must match its std430 layout");
static_assert(sizeof(TreeSite) == 32, "TreeSite must match its std430 layout");
static_assert(sizeof(TreeTileConfig) == 160, "TreeTileConfig must match its std430 layout");
static_assert(sizeof(TreeImpostorParams) == 80, "TreeImpostorParams must match its std430 layout");
static_assert(sizeof(TreeGrassParams) == 48, "TreeGrassParams must match its std430 layout");
static_assert(sizeof(TreeGrassClump) == 32, "TreeGrassClump must match its std430 layout");
static_assert(sizeof(TreeGrassVisible) == 64, "TreeGrassVisible must match its std430 layout");
static_assert(sizeof(TreeGrassSetup) == 128, "TreeGrassSetup must match its std430 layout");
static_assert(sizeof(TreeFloorSetup) == 112, "TreeFloorSetup must match its std430 layout");
static_assert(sizeof(TreeTerrainConfig) == 224, "TreeTerrainConfig must match its std430 layout");

/// The records the generation writes for a set of trees.
struct TreeRecords {
    std::vector<StemRecord> stems[TREE_MAX_LEVELS]; ///< per stem level
    std::vector<SegmentRecord> segments;
    std::vector<LeafRecord> leaves;
};

/**
 * @brief Generates the trees on the CPU with the same code the GPU runs.
 *
 * The reference for testing the GPU generation; the order of the records
 * within each list differs from the GPU's, where threads append concurrently.
 */
TreeRecords generateTreesReference(const std::vector<TreeTypeParams>& types, const std::vector<TreeInstance>& instances,
                                   const TreeFrame& frame);

/// A TreeFrame with +y up, no wind, summer.
TreeFrame defaultTreeFrame();

/// Render settings for a target of `extent` pixels: 4 pixels per triangle edge, stems thinner than half a pixel
/// culled, leaves simplified from 24 down to 8 pixels and thinned out below 4 pixels, a sun from the upper left, a
/// bluish sky and a brownish ground.
TreeRenderSettings defaultTreeRenderSettings(VkExtent2D extent);

/**
 * @brief A tree type for trees farther away (the level of detail L1): one stem level less, its leaves on the stems
 *        of the new last level instead.
 *
 * Each of those stems carries the leaves its dropped children carried,
 * `leafShare` of their number and accordingly larger (the leaves' area stays),
 * so the tree's foliage keeps its extent and density with far fewer records.
 */
TreeTypeParams treeReducedType(const TreeTypeParams& type, float leafShare = 0.25f);

/// Rotation (x, y, z, w) that turns the tree's +z growth direction to `up`.
glm::vec4 treeUpRotation(const glm::vec3& up);

/// Sets how far `inst` has grown, from 0 (a seed) to 1 (fully grown, as an instance without an age).
void setTreeAge(TreeInstance& inst, float age);
/// How far `inst` has grown (setTreeAge()); 1 for an instance without an age.
float treeAge(const TreeInstance& inst);

/// Tree types with the parameters of Weber and Penn (1995), and leaf outlines after Kuth et al.
namespace tree_presets {
TreeTypeParams quakingAspen();
TreeTypeParams blackTupelo();
/// Three-lobed leaves, as of the sassafras in Kuth et al.
TreeTypeParams sassafras();
/// A conifer with needle-like leaves.
TreeTypeParams pine();
} // namespace tree_presets

} // namespace klartraum

#endif // KLARTRAUM_TREE_TREE_MODEL_HPP
