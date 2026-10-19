// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_TREE_BONSAI_SCENE_HPP
#define KLARTRAUM_TREE_BONSAI_SCENE_HPP

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "klartraum/tree/tree_model.hpp"
#include "klartraum/vulkan_gaussian_splatting_types.hpp"

namespace klartraum {

/// A shallow oval pot: a superellipse in plan, standing on y = 0. Units as the lantern capture (about 0.28 tall).
struct BonsaiPot {
    float halfLength = 1.5f; ///< along x
    float halfWidth = 0.9f;  ///< along z
    float exponent = 4.0f;   ///< of the superellipse |x/a|^e + |z/b|^e = 1: 2 an ellipse, larger more rectangular
    float height = 0.3f;     ///< of the rim
    float wall = 0.05f;      ///< thickness at the rim
    float soilHeight = 0.26f;
    float soilMound = 0.07f; ///< the soil rises this much towards the middle
    glm::vec3 glaze{0.17f, 0.21f, 0.26f};
};

struct BonsaiSceneConfig {
    uint32_t treeCount = 13; ///< 8 to 20
    uint32_t seed = 7;
    BonsaiPot pot;
    glm::vec3 lanternPosition{0.72f, 0.0f, 0.36f}; ///< y is taken from the soil
    float lanternYaw = 0.6f;                       ///< radians about +y
    float lanternScale = 1.0f;
};

/**
 * @brief A bonsai forest planting (yose-ue): 8 to 20 trees in an oval pot, and the lantern on the moss.
 *
 * The trees are 4 to 9 times as tall as the lantern, so it stands among them
 * like a real stone lantern in the woods. Everything is described on the CPU
 * once; the trees are generated and drawn on the GPU every frame.
 */
struct BonsaiScene {
    BonsaiSceneConfig config;
    std::vector<TreeTypeParams> types;
    std::vector<TreeInstance> trees;
    std::vector<TreeSceneVertex> mesh; ///< pot and soil, a triangle list
    /// Puts the lantern capture (Y up, its ground at y = 0) on the soil.
    glm::quat lanternRotation{1.0f, 0.0f, 0.0f, 0.0f};
    glm::vec3 lanternTranslation{0.0f};
    glm::vec3 focus{0.0f}; ///< where a camera looks

    /// Height of the soil surface at (x, z) inside the pot.
    float soilHeight(float x, float z) const;
    /// The superellipse measure of (x, z): below 1 inside the pot's outline scaled by `scale`.
    float potMeasure(float x, float z, float scale = 1.0f) const;
};

BonsaiScene makeBonsaiScene(const BonsaiSceneConfig& config = BonsaiSceneConfig{});

/**
 * @brief Samples a triangle list at random points about `spacing` apart, for TreeGaussians' surfels.
 *
 * Deterministic for a `seed`. Each sample has the interpolated normal, the
 * triangle's colour and material, and `spacing` in pos.w.
 */
std::vector<TreeSceneVertex> sampleMeshSurfels(const std::vector<TreeSceneVertex>& mesh, float spacing,
                                               uint32_t seed = 1);

/// Broadleaf and pine types for a bonsai forest: fine twigs, small leaves.
namespace tree_presets {
TreeTypeParams bonsaiBroadleaf();
TreeTypeParams bonsaiPine();
} // namespace tree_presets

} // namespace klartraum

#endif // KLARTRAUM_TREE_BONSAI_SCENE_HPP
