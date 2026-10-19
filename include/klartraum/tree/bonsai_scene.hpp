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
    glm::vec3 focus{0.0f};        ///< where a camera looks
    glm::vec3 lanternFlame{0.0f}; ///< where the lantern's flame burns, in its fire box

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

/// A piece of captured ground: Gaussians around the origin, their ground at y = 0.
using MossPatch = std::vector<Gaussian3D>;

/**
 * @brief Cuts the mossy ground around the lantern of a capture into `sectors` patches.
 *
 * The capture stands on y = 0 with the lantern at the origin (loadGaussiansSpz
 * with flipY). The ground in the ring from `inner` to `outer` around it, below
 * `maxHeight`, is cut into sectors, and the sectors where at least
 * `minGreen` of the Gaussians are green (moss rather than the dry grass on
 * the capture's east side) become patches. Each patch is moved so that its
 * centre is at the origin and its ground at y = 0, and its Gaussians fade out
 * (opacity) towards the cuts, so patches blend into the soil.
 */
std::vector<MossPatch> extractMossPatches(const std::vector<Gaussian3D>& capture, uint32_t sectors = 12,
                                          float inner = 0.14f, float outer = 0.23f, float maxHeight = 0.05f,
                                          float minGreen = 0.6f);

/**
 * @brief Places `count` copies of the `patchCount` patches on the soil of `scene`, turned at random, away from the
 *        lantern.
 *
 * Each copy is enlarged by `scale` and lies on the plane that touches the
 * soil at its centre (TreeLodInstance). Deterministic for a `seed`.
 */
std::vector<TreeLodInstance> placeMossPatches(const BonsaiScene& scene, uint32_t patchCount, uint32_t count,
                                              uint32_t seed = 1, float scale = 1.0f);

/**
 * @brief The placed copies of the patches at full detail: what `tree_lod_write.comp` writes at level 0, without the
 *        changes of colour.
 */
std::vector<Gaussian3D> scatterMossPatches(const std::vector<MossPatch>& patches,
                                           const std::vector<TreeLodInstance>& instances);

/// Broadleaf and pine types for a bonsai forest: fine twigs, small leaves.
namespace tree_presets {
TreeTypeParams bonsaiBroadleaf();
TreeTypeParams bonsaiPine();
} // namespace tree_presets

} // namespace klartraum

#endif // KLARTRAUM_TREE_BONSAI_SCENE_HPP
