// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_TREE_TREE_PLACEMENT_HPP
#define KLARTRAUM_TREE_TREE_PLACEMENT_HPP

#include <vector>

#include "klartraum/tree/bonsai_scene.hpp"

namespace klartraum {

/**
 * @brief Where the trees of `scene` grow (treeSiteAt()): a cell per tree at the forest density of the scale
 *        (BonsaiScale::kTreesPerHectare), jittered by `jitter` cells, broadleaves in the valleys and conifers from
 *        `coniferMeters` above them, the scale's mature tree height.
 */
TreePlacementParams makeTreePlacement(const BonsaiScene& scene, uint32_t seed = 1, float jitter = 0.6f,
                                      float coniferMeters = 650.0f);

/**
 * @brief The `count` sites nearest to `eye` (by the distance to their foot), nearest first.
 *
 * Searches rings of cells around `eye` until the nearest `count` are
 * certain or `maxCells` cells away; fewer are returned over open land.
 */
std::vector<TreeSite> nearestTreeSites(const TreeTerrainParams& terrain, const TreePlacementParams& placement,
                                       glm::vec3 eye, uint32_t count, int maxCells = 300);

/// The instance of a TreeGenerator for `site`: type 0 for broadleaves and shrubs, 1 for conifers, scaled to the
/// site's height (the types are `shapeBase.z` metres tall at scale 1), turned by its yaw, seeded by
/// treeSiteSeed() of `seed` and `variants`.
TreeInstance treeInstanceFor(const TreeSite& site, const std::vector<TreeTypeParams>& types, uint32_t seed,
                             uint32_t variants = 0);

/// The sites with a tree among the cells [x0, x1) x [z0, z1), in the order of the cells (rows along x).
std::vector<TreeSite> treeSitesIn(const TreeTerrainParams& terrain, const TreePlacementParams& placement, int x0,
                                  int z0, int x1, int z1);

} // namespace klartraum

#endif // KLARTRAUM_TREE_TREE_PLACEMENT_HPP
