// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - placementFollowsTheScale: cells are 3.9 m wide at every scale (650 trees per hectare) and trees have the
 *   scale's height
 * - sitesKeepTheirDistance: every tree lies in its cell, on the ground, and trees of neighbouring cells stand at
 *   least (1 - jitter) cells apart
 * - treesGrowOnlyInTheForest: trees stand where the forest weight is positive, shrubs only where there is scrub,
 *   none above the tree line or on rock; a map of the sites is written to build/TestingOutput/tree_terrain/
 * - densityFollowsTheForest: in pure forest there are 650 trees per hectare (+- 10 %), and over the region the
 *   trees number the sum of the forest weights
 * - coniferShareRisesWithAltitude: broadleaves dominate the valleys, conifers the slopes high above them, and
 *   trees get smaller towards the tree line
 * - sitesAreTheSameInAnyOrder: a cell's site does not depend on when or in which order cells are evaluated, and
 *   another seed gives other sites
 * - aMillionTreesAtTheLargestScale: the region at scale 1 holds a tree per 3.9 m square of its forest (half of
 *   the soil): about 1.1 million trees
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>

#include "klartraum/tree/tree_placement.hpp"

using namespace klartraum;

namespace {

BonsaiScene sceneAt(float s, uint32_t seed = 7) {
    BonsaiSceneConfig config;
    config.scale.s = s;
    config.seed = seed;
    config.treeCount = s > 0.3f ? 12u : 0u;
    return makeBonsaiScene(config);
}

// The cells of a square of `n` x `n` cells around (x, z) in units.
struct Block {
    int x0, z0, x1, z1;
};

Block blockAround(const TreePlacementParams& p, float x, float z, int n) {
    const int cx = int(std::floor(x / p.grid.x)), cz = int(std::floor(z / p.grid.x));
    return {cx - n / 2, cz - n / 2, cx + n / 2, cz + n / 2};
}

// A map of the sites in [x0, x0 + width) x [z0, z0 + width * 0.6) units from above: the land cover's colours
// (forest dark green, open land light green, rock grey) with the trees as dots, broadleaves light, conifers dark,
// shrubs olive.
void writePlacementMap(const BonsaiScene& scene, const TreePlacementParams& p, float x0, float z0, float width,
                       const std::string& name) {
    const int w = 900, h = 540;
    const float step = width / w;
    std::vector<glm::vec3> image(size_t(w) * h);
    for (int j = 0; j < h; ++j) {
        for (int i = 0; i < w; ++i) {
            const TreeLandCover c = tree::treeLandCover(scene.terrain, x0 + (i + 0.5f) * step, z0 + (j + 0.5f) * step);
            image[size_t(j) * w + i] =
                c.vegetation.x * glm::vec3(0.55f, 0.7f, 0.5f) +
                (c.vegetation.y + c.vegetation.z + c.vegetation.w) * glm::vec3(0.8f, 0.9f, 0.6f) +
                c.bare.x * glm::vec3(0.75f);
        }
    }
    const int cx0 = int(std::floor(x0 / p.grid.x)), cz0 = int(std::floor(z0 / p.grid.x));
    const int cx1 = int(std::ceil((x0 + width) / p.grid.x)), cz1 = int(std::ceil((z0 + h * step) / p.grid.x));
    for (const TreeSite& site : treeSitesIn(scene.terrain, p, cx0, cz0, cx1, cz1)) {
        const glm::vec3 color = site.kind.x == TREE_SITE_SHRUB     ? glm::vec3(0.5f, 0.5f, 0.15f)
                                : site.kind.x == TREE_SITE_CONIFER ? glm::vec3(0.05f, 0.25f, 0.1f)
                                                                   : glm::vec3(0.3f, 0.55f, 0.15f);
        const float radius =
            std::max(1.0f, 0.35f * p.grid.x / step * std::min(site.posHeight.w * p.grid.w / 25.0f, 1.2f));
        const float px = (site.posHeight.x - x0) / step, pz = (site.posHeight.z - z0) / step;
        for (int j = int(pz - radius); j <= int(pz + radius); ++j) {
            for (int i = int(px - radius); i <= int(px + radius); ++i) {
                if (i >= 0 && j >= 0 && i < w && j < h && glm::length(glm::vec2(i - px, j - pz)) <= radius) {
                    image[size_t(j) * w + i] = color;
                }
            }
        }
    }
    std::filesystem::create_directories("build/TestingOutput/tree_terrain");
    std::ofstream file("build/TestingOutput/tree_terrain/" + name + ".ppm", std::ios::binary);
    file << "P6\n" << w << " " << h << "\n255\n";
    for (const glm::vec3& v : image) {
        const char rgb[3] = {char(std::lround(255.0f * v.x)), char(std::lround(255.0f * v.y)),
                             char(std::lround(255.0f * v.z))};
        file.write(rgb, 3);
    }
}

} // namespace

TEST(TreePlacement, placementFollowsTheScale) {
    for (float s : {0.3f, 0.5f, 1.0f}) {
        const BonsaiScene scene = sceneAt(s);
        const TreePlacementParams p = makeTreePlacement(scene);
        EXPECT_NEAR(p.grid.x * scene.config.scale.metersPerUnit(), std::sqrt(1e4f / 650.0f), 1e-3f) << s;
        EXPECT_NEAR(p.size.x, scene.config.scale.treeHeightMeters(), 1e-3f) << s;
    }
}

TEST(TreePlacement, sitesKeepTheirDistance) {
    const BonsaiScene scene = sceneAt(1.0f);
    const TreePlacementParams p = makeTreePlacement(scene);
    const Block b = blockAround(p, -0.3f, 0.1f, 120);
    int trees = 0;
    for (int cz = b.z0; cz < b.z1; ++cz) {
        for (int cx = b.x0; cx < b.x1; ++cx) {
            const TreeSite a = tree::treeSiteAt(scene.terrain, p, cx, cz);
            if (a.posHeight.w <= 0.0f) {
                continue;
            }
            ++trees;
            EXPECT_GE(a.posHeight.x, cx * p.grid.x);
            EXPECT_LT(a.posHeight.x, (cx + 1) * p.grid.x);
            EXPECT_GE(a.posHeight.z, cz * p.grid.x);
            EXPECT_LT(a.posHeight.z, (cz + 1) * p.grid.x);
            EXPECT_FLOAT_EQ(a.posHeight.y, scene.soilHeight(a.posHeight.x, a.posHeight.z));
            for (int d = 0; d < 4; ++d) {
                const int nx = cx + (d == 0 ? 1 : d == 1 ? -1 : 0), nz = cz + (d == 2 ? 1 : d == 3 ? -1 : 0);
                const TreeSite n = tree::treeSiteAt(scene.terrain, p, nx, nz);
                if (n.posHeight.w > 0.0f) {
                    const float distance =
                        glm::length(glm::vec2(a.posHeight.x - n.posHeight.x, a.posHeight.z - n.posHeight.z));
                    EXPECT_GE(distance, (1.0f - p.grid.y) * p.grid.x * 0.999f);
                }
            }
        }
    }
    EXPECT_GT(trees, 1000);
}

TEST(TreePlacement, treesGrowOnlyInTheForest) {
    const BonsaiScene scene = sceneAt(1.0f);
    const TreePlacementParams p = makeTreePlacement(scene);
    int trees = 0, shrubs = 0;
    // Blocks across valleys, slopes and the high ridge.
    for (glm::vec2 at :
         {glm::vec2(-0.8f, 0.3f), glm::vec2(0.0f, 0.0f), glm::vec2(0.9f, -0.1f), glm::vec2(0.4f, 0.5f)}) {
        const Block b = blockAround(p, at.x, at.y, 160);
        for (const TreeSite& site : treeSitesIn(scene.terrain, p, b.x0, b.z0, b.x1, b.z1)) {
            const glm::vec4 land = tree::treeLandSite(scene.terrain, site.posHeight.x, site.posHeight.z);
            const TreeLandCover c = tree::treeLandCoverFrom(scene.terrain, land);
            if (site.kind.x == TREE_SITE_SHRUB) {
                EXPECT_GT(c.vegetation.w, 0.0f);
                ++shrubs;
            } else {
                EXPECT_GT(c.vegetation.x, 0.0f);
                EXPECT_LT(land.y, BonsaiScale::kTreeLineMeters);
                EXPECT_LT(c.bare.x, 1.0f);
                ++trees;
            }
        }
    }
    EXPECT_GT(trees, 5000);
    EXPECT_GT(shrubs, 50);
    writePlacementMap(scene, p, 0.78f, -0.2f, 0.1f, "placement_s100_edge");
    writePlacementMap(scene, p, 0.55f, -0.35f, 0.1f, "placement_s100_high");
}

TEST(TreePlacement, densityFollowsTheForest) {
    const BonsaiScene scene = sceneAt(1.0f);
    const TreePlacementParams p = makeTreePlacement(scene);
    const float hectare = 1e4f / (scene.config.scale.metersPerUnit() * scene.config.scale.metersPerUnit());
    // Pure forest: count per area of cells whose forest weight is nearly 1.
    int pureCells = 0, pureTrees = 0;
    double weights = 0.0;
    int all = 0;
    const Block b = blockAround(p, 0.0f, 0.0f, 1400);
    for (int cz = b.z0; cz < b.z1; cz += 3) {
        for (int cx = b.x0; cx < b.x1; cx += 3) {
            const float x = (cx + 0.5f) * p.grid.x, z = (cz + 0.5f) * p.grid.x;
            if (tree::treeTerrainRadius(scene.terrain, x, z) >= 0.9f) {
                continue;
            }
            const TreeSite site = tree::treeSiteAt(scene.terrain, p, cx, cz);
            const bool isTree = site.posHeight.w > 0.0f && site.kind.x != TREE_SITE_SHRUB;
            all += isTree ? 1 : 0;
            weights += tree::treeLandCover(scene.terrain, x, z).vegetation.x;
            if (tree::treeLandCover(scene.terrain, x, z).vegetation.x > 0.999f) {
                ++pureCells;
                pureTrees += isTree ? 1 : 0;
            }
        }
    }
    ASSERT_GT(pureCells, 2000);
    const float perHectare = float(pureTrees) / (float(pureCells) * p.grid.x * p.grid.x) * hectare;
    std::printf("pure forest: %.0f trees per hectare; trees %d for forest weights %.0f\n", perHectare, all, weights);
    EXPECT_NEAR(perHectare, 650.0f, 65.0f);
    EXPECT_NEAR(float(all), float(weights), 0.05f * float(weights));
}

TEST(TreePlacement, coniferShareRisesWithAltitude) {
    const BonsaiScene scene = sceneAt(1.0f);
    const TreePlacementParams p = makeTreePlacement(scene);
    // Per band of 250 m altitude: conifers, trees and mean height.
    double conifers[6] = {}, trees[6] = {}, height[6] = {};
    const Block b = blockAround(p, 0.0f, 0.0f, 1800);
    for (int cz = b.z0; cz < b.z1; cz += 2) {
        for (int cx = b.x0; cx < b.x1; cx += 2) {
            const TreeSite site = tree::treeSiteAt(scene.terrain, p, cx, cz);
            if (site.posHeight.w <= 0.0f || site.kind.x == TREE_SITE_SHRUB) {
                continue;
            }
            const float altitude = tree::treeTerrainAltitude(scene.terrain, glm::vec3(site.posHeight));
            const int band = std::clamp(int(altitude / 250.0f), 0, 5);
            conifers[band] += site.kind.x == TREE_SITE_CONIFER ? 1.0 : 0.0;
            trees[band] += 1.0;
            height[band] += site.posHeight.w * p.grid.w;
        }
    }
    ASSERT_GT(trees[0], 200);
    ASSERT_GT(trees[3], 50);
    for (int k = 0; k < 6; ++k) {
        if (trees[k] > 0) {
            std::printf("%4d m: %6.0f trees, conifers %.2f, mean height %.1f m\n", 250 * k, trees[k],
                        conifers[k] / trees[k], height[k] / trees[k]);
        }
    }
    EXPECT_LT(conifers[0] / trees[0], 0.4);
    EXPECT_GT(conifers[3] / trees[3], 0.8);
    // Smaller at the tree line than in the valleys.
    const int high = trees[4] > 20 ? 4 : 3;
    EXPECT_LT(height[high] / trees[high], 0.9 * height[0] / trees[0]);
}

TEST(TreePlacement, sitesAreTheSameInAnyOrder) {
    const BonsaiScene scene = sceneAt(1.0f);
    const TreePlacementParams p = makeTreePlacement(scene);
    const Block b = blockAround(p, -0.4f, 0.2f, 60);
    const std::vector<TreeSite> rows = treeSitesIn(scene.terrain, p, b.x0, b.z0, b.x1, b.z1);
    std::vector<glm::ivec2> cells;
    for (int cz = b.z0; cz < b.z1; ++cz) {
        for (int cx = b.x0; cx < b.x1; ++cx) {
            cells.emplace_back(cx, cz);
        }
    }
    std::shuffle(cells.begin(), cells.end(), std::mt19937(3));
    size_t found = 0;
    for (const glm::ivec2& c : cells) {
        const TreeSite site = tree::treeSiteAt(scene.terrain, p, c.x, c.y);
        if (site.posHeight.w <= 0.0f) {
            continue;
        }
        ++found;
        const auto same = std::find_if(rows.begin(), rows.end(), [&](const TreeSite& r) {
            return r.posHeight == site.posHeight && r.kind == site.kind;
        });
        EXPECT_NE(same, rows.end());
    }
    EXPECT_EQ(found, rows.size());
    // Another seed: other jitter.
    const TreePlacementParams other = makeTreePlacement(scene, 2);
    const std::vector<TreeSite> again = treeSitesIn(scene.terrain, other, b.x0, b.z0, b.x1, b.z1);
    ASSERT_FALSE(again.empty());
    EXPECT_NE(again.front().posHeight, rows.front().posHeight);
}

TEST(TreePlacement, aMillionTreesAtTheLargestScale) {
    const BonsaiScene scene = sceneAt(1.0f);
    const TreePlacementParams p = makeTreePlacement(scene);
    // Every 7th cell along both axes, scaled up.
    const int n = int(2.0f * scene.terrain.outline.x / p.grid.x) + 2;
    double count = 0.0;
    for (int cz = -n / 2; cz < n / 2; cz += 7) {
        for (int cx = -n / 2; cx < n / 2; cx += 7) {
            const TreeSite site = tree::treeSiteAt(scene.terrain, p, cx, cz);
            count += (site.posHeight.w > 0.0f && site.kind.x != TREE_SITE_SHRUB) ? 49.0 : 0.0;
        }
    }
    const double squareMeters = double(scene.soilArea) * std::pow(double(scene.config.scale.metersPerUnit()), 2.0);
    const double expected = scene.config.scale.forestShare() * squareMeters * 1e-4 * 650.0;
    std::printf("about %.2f million trees, expected %.2f million\n", count * 1e-6, expected * 1e-6);
    EXPECT_GT(count, 1e6);
    EXPECT_NEAR(count, expected, 0.1 * expected);
}
