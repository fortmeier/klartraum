// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/tree/tree_placement.hpp"

#include <cmath>

namespace klartraum {

TreePlacementParams makeTreePlacement(const BonsaiScene& scene, uint32_t seed, float jitter, float coniferMeters) {
    const BonsaiScale& scale = scene.config.scale;
    TreePlacementParams p{};
    const float cellMeters = std::sqrt(1e4f / BonsaiScale::kTreesPerHectare);
    p.grid = glm::vec4(cellMeters / scale.metersPerUnit(), jitter, 0.35f, scale.metersPerUnit());
    p.species = glm::vec4(coniferMeters, 200.0f, 0.25f, 0.0f);
    p.size = glm::vec4(scale.treeHeightMeters(), 0.25f, 3.0f, 0.0f);
    p.seed = glm::uvec4(seed, 0u, 0u, 0u);
    return p;
}

std::vector<TreeSite> treeSitesIn(const TreeTerrainParams& terrain, const TreePlacementParams& placement, int x0,
                                  int z0, int x1, int z1) {
    std::vector<TreeSite> sites;
    for (int cz = z0; cz < z1; ++cz) {
        for (int cx = x0; cx < x1; ++cx) {
            const TreeSite site = tree::treeSiteAt(terrain, placement, cx, cz);
            if (site.posHeight.w > 0.0f) {
                sites.push_back(site);
            }
        }
    }
    return sites;
}

} // namespace klartraum
