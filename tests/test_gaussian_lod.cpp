// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - smallSetsStayTheSame: sets within the budget are returned unchanged
 * - reductionKeepsSourceGaussians: a reduced set has the budget's size and consists of input Gaussians, with their
 *   positions, colours, opacities and orientations
 * - everyClusterKeepsAGaussian: reducing ten small, separate clusters to twenty keeps at least one in each
 * - coverageIsPreserved: the visible area (optical depth times broad-axis area) of a uniform field reduced to a
 *   quarter stays, also over two such steps, and the thin axis does not grow; one step to a sixteenth keeps 9/16,
 *   as the broad axes grow at most 3 times
 * - levelsShrinkByQuarters: buildGaussianLods gives the full set and 1/4, 1/16, 1/64 of it, at least the minimum
 * - mossPatchLevels: the captured moss patches reduce to their levels with their coverage
 * - pixelDiameterOfASphere: treePixelDiameter is twice the focal length times the radius over the distance
 * - levelsBlendAroundBoundaries: treeLodBlend gives level 0 above the first boundary's band, the coarser levels
 *   below each boundary, fades linearly within the bands and is continuous in the diameter
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <set>

#include "klartraum/gaussian_data_standard.hpp"
#include "klartraum/gaussian_lod.hpp"
#include "klartraum/tree/bonsai_scene.hpp"
#include "klartraum/tree/tree_model.hpp"

using namespace klartraum;

namespace {

Gaussian3D gaussianAt(float x, float y, float z, float alpha = 0.6f) {
    Gaussian3D g{};
    g.position = {x, y, z};
    g.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
    g.scale = {0.01f, 0.001f, 0.01f};
    g.color = {x, y, z};
    g.alpha = alpha;
    return g;
}

// A uniform layer of Gaussians on y = 0.
std::vector<Gaussian3D> field(int n) {
    std::vector<Gaussian3D> gaussians;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            gaussians.push_back(gaussianAt(float(i) * 0.01f, 0.0f, float(j) * 0.01f));
        }
    }
    return gaussians;
}

double visibleArea(const std::vector<Gaussian3D>& gaussians) {
    double area = 0.0;
    for (const Gaussian3D& g : gaussians) {
        std::array<float, 3> s = g.scale;
        std::sort(s.begin(), s.end());
        area += -std::log1p(-std::min(g.alpha, 0.999f)) * s[1] * s[2];
    }
    return area;
}

} // namespace

TEST(GaussianLod, smallSetsStayTheSame) {
    const std::vector<Gaussian3D> gaussians = field(4);
    const std::vector<Gaussian3D> reduced = gaussianSpatialReduce(gaussians, 100);
    ASSERT_EQ(reduced.size(), gaussians.size());
    for (size_t i = 0; i < reduced.size(); ++i) {
        EXPECT_EQ(reduced[i].position, gaussians[i].position);
        EXPECT_EQ(reduced[i].scale, gaussians[i].scale);
    }
}

TEST(GaussianLod, reductionKeepsSourceGaussians) {
    const std::vector<Gaussian3D> gaussians = field(40);
    const std::vector<Gaussian3D> reduced = gaussianSpatialReduce(gaussians, 300);
    ASSERT_EQ(reduced.size(), 300u);
    std::set<std::array<float, 3>> sources;
    for (const Gaussian3D& g : gaussians) {
        sources.insert(g.position);
    }
    std::set<std::array<float, 3>> seen;
    for (const Gaussian3D& g : reduced) {
        EXPECT_TRUE(sources.count(g.position)) << "not a source Gaussian";
        EXPECT_TRUE(seen.insert(g.position).second) << "taken twice";
        EXPECT_EQ(g.color, g.position); // the colour stays the Gaussian's own
        EXPECT_EQ(g.alpha, 0.6f);
        EXPECT_EQ(g.rotation, (std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}));
    }
}

TEST(GaussianLod, everyClusterKeepsAGaussian) {
    std::vector<Gaussian3D> gaussians;
    for (int c = 0; c < 10; ++c) {
        for (int k = 0; k < 50; ++k) {
            // Away from the cell boundaries of a grid over the clusters.
            gaussians.push_back(
                gaussianAt(float(c) * 1.0f + 0.4f + 0.001f * float(k % 7), 0.0f, 0.001f * float(k / 7)));
        }
    }
    // With room for twice as many: the grid that fits separates the clusters. (With exactly ten, a grid of
    // ten cells across can split a cluster at a cell edge and merge two others.)
    const std::vector<Gaussian3D> reduced = gaussianSpatialReduce(gaussians, 20);
    ASSERT_EQ(reduced.size(), 20u);
    std::set<int> clusters;
    for (const Gaussian3D& g : reduced) {
        clusters.insert(int(std::floor(g.position[0])));
    }
    EXPECT_EQ(clusters.size(), 10u);
}

TEST(GaussianLod, coverageIsPreserved) {
    const std::vector<Gaussian3D> gaussians = field(64);
    // A quarter: the broad axes grow by 2.
    const std::vector<Gaussian3D> quarter = gaussianSpatialReduce(gaussians, 1024);
    EXPECT_NEAR(visibleArea(quarter) / visibleArea(gaussians), 1.0, 0.1);
    for (const Gaussian3D& g : quarter) {
        EXPECT_EQ(g.scale[1], 0.001f); // the thin axis
    }
    // A sixteenth in one step would need 4: the growth stops at 3, 9/16 of the area.
    const std::vector<Gaussian3D> sixteenth = gaussianSpatialReduce(gaussians, 256);
    EXPECT_NEAR(visibleArea(sixteenth) / visibleArea(gaussians), 9.0 / 16.0, 0.05);
    // In two quarter steps, as buildGaussianLods() reduces, it keeps its coverage.
    const std::vector<Gaussian3D> twoSteps = gaussianSpatialReduce(quarter, 256);
    EXPECT_NEAR(visibleArea(twoSteps) / visibleArea(gaussians), 1.0, 0.1);
}

TEST(GaussianLod, levelsShrinkByQuarters) {
    const GaussianLods lods = buildGaussianLods(field(64), 16);
    EXPECT_EQ(lods.levels[0].size(), 4096u);
    EXPECT_EQ(lods.levels[1].size(), 1024u);
    EXPECT_EQ(lods.levels[2].size(), 256u);
    EXPECT_EQ(lods.levels[3].size(), 64u);
    const GaussianLods small = buildGaussianLods(field(16), 16);
    EXPECT_EQ(small.levels[3].size(), 16u);
}

TEST(GaussianLod, mossPatchLevels) {
    const std::vector<MossPatch> patches = extractMossPatches(loadGaussiansSpz("data/lantern.spz", true));
    ASSERT_FALSE(patches.empty());
    for (const MossPatch& patch : patches) {
        const GaussianLods lods = buildGaussianLods(patch);
        std::cout << "moss patch levels " << lods.levels[0].size() << " " << lods.levels[1].size() << " "
                  << lods.levels[2].size() << " " << lods.levels[3].size() << std::endl;
        for (int level = 1; level < 4; ++level) {
            EXPECT_LT(lods.levels[level].size(), lods.levels[level - 1].size());
            // The coverage changes little; the clamp of the growth to 3x lets the coarsest level lose some.
            EXPECT_GT(visibleArea(lods.levels[level]) / visibleArea(lods.levels[0]), 0.7) << level;
        }
    }
}

TEST(GaussianLod, pixelDiameterOfASphere) {
    const float pixels = tree::treePixelDiameter(glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, -10.0f), 1.0f, 1000.0f);
    EXPECT_NEAR(pixels, 2.0f * 1000.0f * 1.0f / std::sqrt(99.0f), 1e-3f);
    // Inside the sphere: large, not infinite.
    EXPECT_GT(tree::treePixelDiameter(glm::vec3(0.0f), glm::vec3(0.0f), 1.0f, 1000.0f), 1e5f);
}

TEST(GaussianLod, levelsBlendAroundBoundaries) {
    const glm::vec3 b(160.0f, 48.0f, 12.0f);
    const float band = 0.15f;
    EXPECT_EQ(tree::treeLodBlend(400.0f, b, band), glm::vec2(0.0f, 0.0f));
    EXPECT_EQ(tree::treeLodBlend(100.0f, b, band), glm::vec2(1.0f, 0.0f));
    EXPECT_EQ(tree::treeLodBlend(20.0f, b, band), glm::vec2(2.0f, 0.0f));
    EXPECT_EQ(tree::treeLodBlend(5.0f, b, band), glm::vec2(3.0f, 0.0f));
    EXPECT_NEAR(tree::treeLodBlend(160.0f, b, band).y, 0.5f, 1e-6f); // half way at the boundary
    // Continuous: the effective level (level + fade) changes smoothly with the diameter.
    float previous = 0.0f;
    for (float pixels = 400.0f; pixels > 1.0f; pixels -= 0.05f) {
        const glm::vec2 lod = tree::treeLodBlend(pixels, b, band);
        const float effective = lod.x + lod.y;
        EXPECT_GE(effective, previous - 1e-5f) << pixels;
        EXPECT_LT(effective - previous, 0.02f) << pixels;
        previous = effective;
    }
}
