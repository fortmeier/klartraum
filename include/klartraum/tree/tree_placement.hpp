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

/// The sites with a tree among the cells [x0, x1) x [z0, z1), in the order of the cells (rows along x).
std::vector<TreeSite> treeSitesIn(const TreeTerrainParams& terrain, const TreePlacementParams& placement, int x0,
                                  int z0, int x1, int z1);

} // namespace klartraum

#endif // KLARTRAUM_TREE_TREE_PLACEMENT_HPP
