// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/tree/tree_placement.hpp"

#include <algorithm>
#include <cmath>

#include <glm/gtc/quaternion.hpp>

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

std::vector<TreeSite> nearestTreeSites(const TreeTerrainParams& terrain, const TreePlacementParams& placement,
                                       glm::vec3 eye, uint32_t count, int maxCells) {
    const float cell = placement.grid.x;
    const int cx = int(std::floor(eye.x / cell)), cz = int(std::floor(eye.z / cell));
    std::vector<std::pair<float, TreeSite>> found;
    auto distance = [&](const TreeSite& s) { return glm::length(glm::vec3(s.posHeight) - eye); };
    for (int ring = 0; ring <= maxCells; ++ring) {
        for (int dz = -ring; dz <= ring; ++dz) {
            for (int dx = -ring; dx <= ring; ++dx) {
                if (std::max(std::abs(dx), std::abs(dz)) != ring) {
                    continue;
                }
                const TreeSite site = tree::treeSiteAt(terrain, placement, cx + dx, cz + dz);
                if (site.posHeight.w > 0.0f) {
                    found.emplace_back(distance(site), site);
                }
            }
        }
        // Every cell beyond this ring is farther than ring cells away horizontally.
        if (found.size() >= count) {
            std::partial_sort(found.begin(), found.begin() + count, found.end(),
                              [](const auto& a, const auto& b) { return a.first < b.first; });
            if (found[count - 1].first <= float(ring) * cell) {
                break;
            }
        }
    }
    std::sort(found.begin(), found.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::vector<TreeSite> sites;
    for (size_t i = 0; i < found.size() && i < count; ++i) {
        sites.push_back(found[i].second);
    }
    return sites;
}

TreeInstance treeInstanceFor(const TreeSite& site, const std::vector<TreeTypeParams>& types, uint32_t seed,
                             uint32_t variants) {
    const uint32_t type = site.kind.x == TREE_SITE_CONIFER && types.size() > 1 ? 1u : 0u;
    const glm::vec4 up = treeUpRotation(glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::quat rotation =
        glm::angleAxis(site.kind.y, glm::vec3(0.0f, 1.0f, 0.0f)) * glm::quat(up.w, up.x, up.y, up.z);
    TreeInstance inst{};
    inst.posScale = glm::vec4(glm::vec3(site.posHeight), site.posHeight.w / types[type].shapeBase.z);
    inst.rot = glm::vec4(rotation.x, rotation.y, rotation.z, rotation.w);
    inst.ids = glm::uvec4(type, tree::treeSiteSeed(seed, site.kind.y, variants), 0u, 0u);
    return inst;
}

} // namespace klartraum
