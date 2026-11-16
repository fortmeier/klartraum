// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - reliefGrowsWithTheScale: BonsaiScale::reliefUnits is 0 up to kReliefStart, grows steadily to kReliefMeters
 *   in true proportion at scale 1, and scales with reliefExaggeration (1.5 by default)
 * - terrainEqualsTheSoilWithoutRelief: up to kReliefStart the ground is the mounded soil with its bumps as
 *   before the landscape existed
 * - terrainLiesInThePot: with relief, the ground stays between the soil level at the rim and the soil plus the
 *   relief, and the landscape's shape stays in [0, 1]
 * - terrainReliefRangeGrowsWithTheScale: the measured height range of the landscape grows with the scale and
 *   spans most of reliefUnits at scale 1; height maps are written to build/TestingOutput/tree_terrain/
 * - terrainIsContinuousAcrossTheRimBand: along rays from the middle to the rim the ground has no jumps, the
 *   relief falls to nothing at the rim, and beyond the outline the ground is the soil at the rim
 * - terrainIsDeterministicPerSeed: the same seed gives the same ground, another seed another landscape
 * - landCoverWeightsSumToOne: the weights of the classes are in [0, 1] and sum to 1 everywhere, water is 0
 * - landCoverIsAllForestAtSmallScales: up to scale 0.3 forest covers all of the soil
 * - forestCoversHalfTheRegion: at scale 1 forest covers 50 +- 5 % of the soil for several seeds, measured on a
 *   grid other than the calibration's; between, the share follows BonsaiScale::forestShare; land-cover maps are
 *   written to build/TestingOutput/tree_terrain/
 * - landCoverFormsConnectedAreas: forest and open land form large connected areas with few tiny islands
 * - landCoverFollowsTheTerrain: fields lie mostly on slopes below 10 degrees and low ground; rock on steep
 *   ground or above the tree line; no forest above the tree line
 * - landCoverIsDeterministicPerSeed: the same seed gives the same land cover, another seed another one
 * - terrainShadowsItself: without relief nothing is in shadow, under an overhead sun nothing either; under a low
 *   sun ground facing away from it is in shadow, ground facing it mostly lit, with soft edges between
 * - hazeGrowsWithDistance: the miniature world's haze is off without an atmosphere, grows monotonically with the
 *   distance travelled inside the world's box, counts only that part of a view from outside, never covers a view
 *   that misses the box, and thickens towards the valley floor with valley fog
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "klartraum/tree/bonsai_scene.hpp"

using namespace klartraum;

namespace {

BonsaiScene sceneAt(float s, uint32_t seed = 7, float exaggeration = BonsaiScale{}.reliefExaggeration) {
    BonsaiSceneConfig config;
    config.scale.s = s;
    config.scale.reliefExaggeration = exaggeration;
    config.seed = seed;
    // A few trees: the scene's tree placement is not what these tests are about, and it is limited to plantings.
    config.treeCount = s > 0.3f ? 12u : 0u;
    return makeBonsaiScene(config);
}

// The ground as the soil only: mound and bumps, without the landscape.
float soilOnly(const BonsaiScene& scene, float x, float z) {
    const BonsaiPot& pot = scene.config.pot;
    const float inner = 1.0f - pot.wall / pot.halfWidth;
    const float m = std::pow(std::abs(x / (pot.halfLength * inner)), pot.exponent) +
                    std::pow(std::abs(z / (pot.halfWidth * inner)), pot.exponent);
    const float r = std::min(std::pow(m, 1.0f / pot.exponent), 1.0f);
    const float bumps = 0.012f * (tree::treeFbm(glm::vec3(x * 3.0f, 0.0f, z * 3.0f)) - 0.5f);
    return pot.soilHeight + pot.soilMound * (1.0f - r * r) + bumps * (1.0f - r);
}

// Calls f(x, z, r) on a grid over the soil's bounding box for the points inside its outline.
template <typename F>
void forSoilGrid(const BonsaiScene& scene, int n, F f) {
    const TreeTerrainParams& t = scene.terrain;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            const float x = t.outline.x * (2.0f * (i + 0.5f) / n - 1.0f);
            const float z = t.outline.y * (2.0f * (j + 0.5f) / n - 1.0f);
            const float r = tree::treeTerrainRadius(t, x, z);
            if (r < 1.0f) {
                f(x, z, r);
            }
        }
    }
}

// A map of the soil from above, coloured by `color(x, z, height)` and shaded by the slopes (black outside the
// soil's outline), to build/TestingOutput/tree_terrain/<name>.ppm.
void writeShadedMap(const BonsaiScene& scene, const std::string& name, int width,
                    const std::function<glm::vec3(float, float, float)>& color) {
    const TreeTerrainParams& t = scene.terrain;
    const int height = int(width * t.outline.y / t.outline.x);
    std::filesystem::create_directories("build/TestingOutput/tree_terrain");
    std::ofstream file("build/TestingOutput/tree_terrain/" + name + ".ppm", std::ios::binary);
    file << "P6\n" << width << " " << height << "\n255\n";
    const float step = 2.0f * t.outline.x / width;
    const glm::vec3 light = glm::normalize(glm::vec3(-1.0f, 1.2f, -1.0f));
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            const float x = -t.outline.x + (i + 0.5f) * step;
            const float z = -t.outline.y + (j + 0.5f) * step;
            glm::vec3 rgb(0.0f);
            if (tree::treeTerrainRadius(t, x, z) < 1.0f) {
                const float h = scene.soilHeight(x, z);
                const glm::vec3 n =
                    glm::normalize(glm::vec3(scene.soilHeight(x - step, z) - scene.soilHeight(x + step, z), 2.0f * step,
                                             scene.soilHeight(x, z - step) - scene.soilHeight(x, z + step)));
                rgb = color(x, z, h) * (0.25f + 0.75f * std::max(glm::dot(n, light), 0.0f));
            }
            const char c[3] = {char(std::lround(255.0f * std::clamp(rgb.x, 0.0f, 1.0f))),
                               char(std::lround(255.0f * std::clamp(rgb.y, 0.0f, 1.0f))),
                               char(std::lround(255.0f * std::clamp(rgb.z, 0.0f, 1.0f)))};
            file.write(c, 3);
        }
    }
}

// The soil's heights, from green valleys to white peaks.
void writeHeightMap(const BonsaiScene& scene, const std::string& name, int width) {
    const TreeTerrainParams& t = scene.terrain;
    const float low = t.soil.x - t.soil.z;
    const float high = t.soil.x + t.soil.y + std::max(scene.config.scale.reliefUnits(), 0.05f);
    writeShadedMap(scene, name, width, [&](float, float, float h) {
        const float a = std::clamp((h - low) / (high - low), 0.0f, 1.0f);
        return glm::mix(glm::vec3(0.25f, 0.45f, 0.2f), glm::vec3(0.95f, 0.92f, 0.85f), a);
    });
}

// The land cover: forest dark green, meadow light green, field yellow, scrub olive, rock grey.
void writeLandCoverMap(const BonsaiScene& scene, const std::string& name, int width) {
    writeShadedMap(scene, name, width, [&](float x, float z, float) {
        const TreeLandCover c = tree::treeLandCover(scene.terrain, x, z);
        return c.vegetation.x * glm::vec3(0.08f, 0.3f, 0.1f) + c.vegetation.y * glm::vec3(0.45f, 0.75f, 0.3f) +
               c.vegetation.z * glm::vec3(0.9f, 0.8f, 0.35f) + c.vegetation.w * glm::vec3(0.45f, 0.5f, 0.2f) +
               c.bare.x * glm::vec3(0.6f, 0.58f, 0.55f);
    });
}

// The mean land cover over a grid on the soil, and each grid point's weights.
struct CoverGrid {
    int nx = 0, nz = 0;
    std::vector<TreeLandCover> cells; // nx * nz, rows along x; bare.w = 1 inside the soil's outline
    std::vector<glm::vec4> sites;
    TreeLandCover mean{};
};

CoverGrid coverGrid(const BonsaiScene& scene, int nx, float offset = 0.5f) {
    const TreeTerrainParams& t = scene.terrain;
    CoverGrid g;
    g.nx = nx;
    g.nz = int(nx * t.outline.y / t.outline.x);
    int inside = 0;
    for (int j = 0; j < g.nz; ++j) {
        for (int i = 0; i < g.nx; ++i) {
            const float x = t.outline.x * (2.0f * (i + offset) / g.nx - 1.0f);
            const float z = t.outline.y * (2.0f * (j + offset) / g.nz - 1.0f);
            TreeLandCover c{};
            glm::vec4 site(0.0f);
            if (tree::treeTerrainRadius(t, x, z) < 1.0f) {
                site = tree::treeLandSite(t, x, z);
                c = tree::treeLandCoverFrom(t, site);
                g.mean.vegetation += c.vegetation;
                g.mean.bare += c.bare;
                c.bare.w = 1.0f;
                ++inside;
            }
            g.cells.push_back(c);
            g.sites.push_back(site);
        }
    }
    g.mean.vegetation /= float(inside);
    g.mean.bare /= float(inside);
    return g;
}

// Sizes of the 4-connected components of the cells where `in` holds.
std::vector<int> componentSizes(const CoverGrid& g, const std::function<bool(const TreeLandCover&)>& in) {
    std::vector<int> label(g.cells.size(), -1), sizes;
    for (size_t start = 0; start < g.cells.size(); ++start) {
        if (label[start] >= 0 || !in(g.cells[start])) {
            continue;
        }
        std::vector<size_t> stack = {start};
        label[start] = int(sizes.size());
        int size = 0;
        while (!stack.empty()) {
            const size_t k = stack.back();
            stack.pop_back();
            ++size;
            const int i = int(k % g.nx), j = int(k / g.nx);
            const int di[4] = {1, -1, 0, 0}, dj[4] = {0, 0, 1, -1};
            for (int d = 0; d < 4; ++d) {
                const int a = i + di[d], b = j + dj[d];
                if (a < 0 || b < 0 || a >= g.nx || b >= g.nz) {
                    continue;
                }
                const size_t n = size_t(b) * g.nx + a;
                if (label[n] < 0 && in(g.cells[n])) {
                    label[n] = label[start];
                    stack.push_back(n);
                }
            }
        }
        sizes.push_back(size);
    }
    return sizes;
}

} // namespace

TEST(TreeTerrain, reliefGrowsWithTheScale) {
    EXPECT_EQ(BonsaiScale{0.0f}.reliefUnits(), 0.0f);
    EXPECT_EQ(BonsaiScale{BonsaiScale::kForest}.reliefUnits(), 0.0f);
    EXPECT_EQ(BonsaiScale{BonsaiScale::kReliefStart}.reliefUnits(), 0.0f);
    float previous = 0.0f;
    for (float s = 0.3f; s <= 1.0001f; s += 0.05f) {
        const float relief = BonsaiScale{s}.reliefUnits();
        EXPECT_GT(relief, previous) << s;
        previous = relief;
    }
    // True proportions at scale 1: kReliefMeters over the 8 km the soil stands for.
    const BonsaiScale largest{1.0f, 1.0f};
    EXPECT_NEAR(largest.reliefUnits() * largest.metersPerUnit(), BonsaiScale::kReliefMeters, 1.0f);
    const BonsaiScale exaggerated{1.0f, 2.5f};
    EXPECT_NEAR(exaggerated.reliefUnits(), 2.5f * largest.reliefUnits(), 1e-5f);
    EXPECT_NEAR(BonsaiScale{1.0f}.reliefUnits(), 1.5f * largest.reliefUnits(), 1e-5f);
    // Ridges come with the mountains.
    EXPECT_EQ(BonsaiScale{0.3f}.reliefRidges(), 0.0f);
    EXPECT_EQ(BonsaiScale{1.0f}.reliefRidges(), 1.0f);
}

TEST(TreeTerrain, terrainEqualsTheSoilWithoutRelief) {
    for (float s : {0.0f, 0.1f, BonsaiScale::kForest, BonsaiScale::kReliefStart}) {
        const BonsaiScene scene = sceneAt(s);
        EXPECT_EQ(scene.terrain.relief.x, 0.0f) << s;
        int count = 0;
        forSoilGrid(scene, 61, [&](float x, float z, float) {
            EXPECT_NEAR(scene.soilHeight(x, z), soilOnly(scene, x, z), 1e-6f) << s << " " << x << " " << z;
            ++count;
        });
        EXPECT_GT(count, 2000) << s;
    }
}

TEST(TreeTerrain, terrainLiesInThePot) {
    for (float exaggeration : {1.0f, 3.0f}) {
        for (float s : {0.3f, 0.5f, 0.75f, 1.0f}) {
            const BonsaiScene scene = sceneAt(s, 7, exaggeration);
            const TreeTerrainParams& t = scene.terrain;
            ASSERT_GT(t.relief.x, 0.0f);
            forSoilGrid(scene, 81, [&](float x, float z, float) {
                const float shape = tree::treeTerrainShape(t, x, z);
                EXPECT_GE(shape, 0.0f);
                EXPECT_LE(shape, 1.0f);
                const float h = scene.soilHeight(x, z);
                const float soil = soilOnly(scene, x, z);
                EXPECT_GE(h, soil - 1e-6f);
                EXPECT_LE(h, soil + t.relief.x + 1e-5f);
                // Above the pot's bottom, and below the rim plus the relief.
                EXPECT_GT(h, 0.5f * scene.config.pot.soilHeight);
                EXPECT_LT(h, scene.config.pot.height + t.soil.y + t.relief.x);
            });
        }
    }
}

TEST(TreeTerrain, terrainReliefRangeGrowsWithTheScale) {
    float previous = 0.0f;
    for (float s : {0.3f, 0.5f, 0.75f, 1.0f}) {
        const BonsaiScene scene = sceneAt(s);
        float low = 1e9f, high = -1e9f;
        // The landscape alone, inside the band at the rim.
        forSoilGrid(scene, 121, [&](float x, float z, float r) {
            if (r < 1.0f - scene.terrain.relief.w) {
                const float relief = scene.soilHeight(x, z) - soilOnly(scene, x, z);
                low = std::min(low, relief);
                high = std::max(high, relief);
            }
        });
        const float range = high - low;
        EXPECT_GT(range, previous) << s;
        // Valleys reach down near the soil, peaks up to most of the relief (measured 0.6 to 0.9 of it).
        EXPECT_LT(low, 0.1f * scene.config.scale.reliefUnits()) << s;
        EXPECT_GT(range, 0.5f * scene.config.scale.reliefUnits()) << s;
        previous = range;
        writeHeightMap(scene, "height_s" + std::to_string(int(std::lround(s * 100.0f))), 720);
    }
    writeHeightMap(sceneAt(1.0f, 7, 2.5f), "height_s100_exaggerated", 720);
}

TEST(TreeTerrain, terrainIsContinuousAcrossTheRimBand) {
    const BonsaiScene scene = sceneAt(1.0f, 7, 3.0f);
    const TreeTerrainParams& t = scene.terrain;
    const float rim = t.soil.x;
    for (int k = 0; k < 64; ++k) {
        const float theta = 2.0f * 3.14159265f * (k + 0.37f) / 64.0f;
        // Along the ray through the superellipse's outline at theta, from the middle to beyond the rim.
        const float e = 2.0f / t.outline.z;
        const glm::vec2 edge(t.outline.x * std::copysign(std::pow(std::abs(std::cos(theta)), e), std::cos(theta)),
                             t.outline.y * std::copysign(std::pow(std::abs(std::sin(theta)), e), std::sin(theta)));
        float previous = scene.soilHeight(0.0f, 0.0f);
        float maxStep = 0.0f;
        for (int i = 1; i <= 1200; ++i) {
            const glm::vec2 p = edge * (1.2f * i / 1200.0f);
            const float h = scene.soilHeight(p.x, p.y);
            maxStep = std::max(maxStep, std::abs(h - previous));
            previous = h;
            const float r = tree::treeTerrainRadius(t, p.x, p.y);
            if (r >= 1.0f) {
                EXPECT_FLOAT_EQ(h, rim) << k << " " << i;
            } else if (r > 0.99f) {
                // The relief has (almost) faded out.
                EXPECT_LT(h - soilOnly(scene, p.x, p.y), 0.01f * t.relief.x) << k << " " << i;
            }
        }
        // Steps of 1/1000 of the ray: the steepest flanks (fine ridges at 3x relief) rise at most about 10 units
        // per unit, measured below 0.012 per step.
        EXPECT_LT(maxStep, 0.03f) << k;
    }
}

TEST(TreeTerrain, terrainIsDeterministicPerSeed) {
    const BonsaiScene a = sceneAt(1.0f, 7);
    const BonsaiScene b = sceneAt(1.0f, 7);
    const BonsaiScene c = sceneAt(1.0f, 8);
    double difference = 0.0;
    int count = 0;
    forSoilGrid(a, 61, [&](float x, float z, float r) {
        EXPECT_EQ(a.soilHeight(x, z), b.soilHeight(x, z));
        if (r < 0.75f) {
            difference += std::abs(a.soilHeight(x, z) - c.soilHeight(x, z));
            ++count;
        }
    });
    // Another seed is another landscape: on average its heights differ by a good part of the relief.
    EXPECT_GT(difference / count, 0.1 * a.terrain.relief.x);
}

TEST(TreeTerrain, landCoverWeightsSumToOne) {
    for (float s : {0.0f, 0.3f, 0.5f, 1.0f}) {
        const BonsaiScene scene = sceneAt(s);
        forSoilGrid(scene, 61, [&](float x, float z, float) {
            const TreeLandCover c = tree::treeLandCover(scene.terrain, x, z);
            for (float w : {c.vegetation.x, c.vegetation.y, c.vegetation.z, c.vegetation.w, c.bare.x, c.bare.y}) {
                EXPECT_GE(w, 0.0f);
                EXPECT_LE(w, 1.0f);
            }
            EXPECT_EQ(c.bare.y, 0.0f);
            EXPECT_NEAR(c.vegetation.x + c.vegetation.y + c.vegetation.z + c.vegetation.w + c.bare.x + c.bare.y, 1.0f,
                        1e-5f);
        });
    }
}

TEST(TreeTerrain, landCoverIsAllForestAtSmallScales) {
    for (float s : {0.0f, BonsaiScale::kForest, 0.3f}) {
        EXPECT_EQ(BonsaiScale{s}.forestShare(), 1.0f) << s;
        const BonsaiScene scene = sceneAt(s);
        forSoilGrid(scene, 41, [&](float x, float z, float) {
            EXPECT_FLOAT_EQ(tree::treeLandCover(scene.terrain, x, z).vegetation.x, 1.0f) << s << " " << x << " " << z;
        });
    }
}

TEST(TreeTerrain, forestCoversHalfTheRegion) {
    EXPECT_NEAR(BonsaiScale{1.0f}.forestShare(), 0.5f, 1e-6f);
    for (uint32_t seed : {7u, 8u, 9u, 21u}) {
        const BonsaiScene scene = sceneAt(1.0f, seed);
        // A finer grid, offset from the calibration's.
        const CoverGrid g = coverGrid(scene, 233, 0.31f);
        EXPECT_NEAR(g.mean.vegetation.x, 0.5f, 0.05f) << seed;
        // Every class of open land occurs.
        EXPECT_GT(g.mean.vegetation.y, 0.05f) << seed;
        EXPECT_GT(g.mean.vegetation.z, 0.02f) << seed;
        EXPECT_GT(g.mean.vegetation.w, 0.02f) << seed;
        EXPECT_GT(g.mean.bare.x, 0.01f) << seed;
        std::printf("seed %u: forest %.3f meadow %.3f field %.3f scrub %.3f rock %.3f\n", seed, g.mean.vegetation.x,
                    g.mean.vegetation.y, g.mean.vegetation.z, g.mean.vegetation.w, g.mean.bare.x);
        writeLandCoverMap(scene, "cover_s100_seed" + std::to_string(seed), 720);
    }
    for (float s : {0.45f, 0.6f, 0.75f}) {
        const BonsaiScene scene = sceneAt(s);
        EXPECT_NEAR(coverGrid(scene, 233, 0.31f).mean.vegetation.x, scene.config.scale.forestShare(), 0.05f) << s;
        writeLandCoverMap(scene, "cover_s" + std::to_string(int(std::lround(s * 100.0f))), 720);
    }
}

TEST(TreeTerrain, landCoverFormsConnectedAreas) {
    for (uint32_t seed : {7u, 8u, 9u}) {
        const BonsaiScene scene = sceneAt(1.0f, seed);
        const CoverGrid g = coverGrid(scene, 240);
        auto check = [&](const char* what, const std::function<bool(const TreeLandCover&)>& in) {
            const std::vector<int> sizes = componentSizes(g, in);
            int total = 0, tiny = 0, largest = 0;
            for (int size : sizes) {
                total += size;
                largest = std::max(largest, size);
                // Below 0.1 % of the soil: some 50 cells, at scale 1 about 18 ha.
                tiny += size < int(0.001f * g.cells.size()) ? size : 0;
            }
            // Tiny islands hold a small part of the area, and one area holds a good part of it.
            EXPECT_LT(tiny, 0.05f * total) << what << " " << seed;
            EXPECT_GT(largest, 0.2f * total) << what << " " << seed;
        };
        check("forest", [](const TreeLandCover& c) { return c.bare.w > 0.0f && c.vegetation.x > 0.5f; });
        check("open land", [](const TreeLandCover& c) { return c.bare.w > 0.0f && c.vegetation.x <= 0.5f; });
    }
}

TEST(TreeTerrain, landCoverFollowsTheTerrain) {
    const float gentle = std::tan(glm::radians(10.0f));
    const float steep = std::tan(glm::radians(30.0f));
    for (uint32_t seed : {7u, 8u, 9u}) {
        const BonsaiScene scene = sceneAt(1.0f, seed);
        const CoverGrid g = coverGrid(scene, 200);
        double field = 0.0, fieldGentle = 0.0, rock = 0.0, rockSteepOrHigh = 0.0;
        for (size_t k = 0; k < g.cells.size(); ++k) {
            const TreeLandCover& c = g.cells[k];
            const glm::vec4& site = g.sites[k];
            field += c.vegetation.z;
            fieldGentle += site.x < gentle ? c.vegetation.z : 0.0f;
            rock += c.bare.x;
            rockSteepOrHigh += (site.x > steep || site.y > BonsaiScale::kTreeLineMeters) ? c.bare.x : 0.0f;
            if (site.y > BonsaiScale::kTreeLineMeters) {
                EXPECT_EQ(c.vegetation.x, 0.0f);
            }
        }
        EXPECT_GT(fieldGentle, 0.8 * field) << seed;
        EXPECT_GT(rockSteepOrHigh, 0.8 * rock) << seed;
    }
}

TEST(TreeTerrain, landCoverIsDeterministicPerSeed) {
    const BonsaiScene a = sceneAt(1.0f, 7);
    const BonsaiScene b = sceneAt(1.0f, 7);
    const BonsaiScene c = sceneAt(1.0f, 8);
    EXPECT_EQ(a.terrain.cover.x, b.terrain.cover.x);
    int differ = 0, count = 0;
    forSoilGrid(a, 61, [&](float x, float z, float) {
        const float fa = tree::treeLandCover(a.terrain, x, z).vegetation.x;
        EXPECT_EQ(fa, tree::treeLandCover(b.terrain, x, z).vegetation.x);
        differ += std::abs(fa - tree::treeLandCover(c.terrain, x, z).vegetation.x) > 0.5f ? 1 : 0;
        ++count;
    });
    // Two independent half-forested maps disagree on about half of the soil.
    EXPECT_GT(differ, count / 4);
}

TEST(TreeTerrain, terrainShadowsItself) {
    const glm::vec3 low = glm::normalize(glm::vec3(-1.0f, 0.25f, 0.3f));
    const BonsaiScene flat = sceneAt(0.21f);
    forSoilGrid(flat, 21, [&](float x, float z, float) {
        EXPECT_EQ(tree::treeTerrainSunVisibility(flat.terrain, glm::vec3(x, flat.soilHeight(x, z), z), low), 1.0f);
    });
    const BonsaiScene scene = sceneAt(1.0f);
    int away = 0, awayShadowed = 0, facing = 0, facingLit = 0, soft = 0;
    forSoilGrid(scene, 81, [&](float x, float z, float r) {
        if (r > 0.7f) {
            return;
        }
        const glm::vec3 p(x, scene.soilHeight(x, z), z);
        EXPECT_GT(tree::treeTerrainSunVisibility(scene.terrain, p, glm::vec3(0.0f, 1.0f, 0.0f)), 0.999f);
        const float e = 0.004f;
        const glm::vec3 n = glm::normalize(glm::vec3(scene.soilHeight(x - e, z) - scene.soilHeight(x + e, z), 2.0f * e,
                                                     scene.soilHeight(x, z - e) - scene.soilHeight(x, z + e)));
        const float v = tree::treeTerrainSunVisibility(scene.terrain, p, low);
        EXPECT_GE(v, 0.0f);
        EXPECT_LE(v, 1.0f);
        soft += (v > 0.05f && v < 0.95f) ? 1 : 0;
        if (glm::dot(n, low) < -0.1f) {
            ++away;
            awayShadowed += v < 0.5f ? 1 : 0;
        } else if (glm::dot(n, low) > 0.3f) {
            ++facing;
            facingLit += v > 0.5f ? 1 : 0;
        }
    });
    std::printf("facing away: %d of %d in shadow; facing: %d of %d lit; %d soft\n", awayShadowed, away, facingLit,
                facing, soft);
    ASSERT_GT(away, 100);
    ASSERT_GT(facing, 100);
    EXPECT_GT(awayShadowed, away * 9 / 10);
    EXPECT_GT(facingLit, facing * 7 / 10);
    EXPECT_GT(soft, 20);
}

TEST(TreeTerrain, hazeGrowsWithDistance) {
    TreeRenderSettings s = defaultTreeRenderSettings({1920, 1080});
    const glm::vec3 eye(0.0f, 0.5f, 0.0f);
    EXPECT_EQ(tree::treeHaze(eye + glm::vec3(1.0f, 0.0f, 0.0f), eye, s).w, 0.0f);
    // A world of 8 km in 2.88 units, 15 km visibility, from y 0.26 to 1.5.
    s.atmosphere = glm::vec4(8000.0f / 2.88f, 15000.0f, 0.0f, 0.0f);
    s.atmosphereBox = glm::vec4(1.4f, 0.85f, 0.26f, 1.5f);
    float previous = 0.0f;
    for (float d = 0.05f; d < 1.35f; d += 0.05f) {
        const float haze = tree::treeHaze(eye + glm::vec3(d, 0.0f, 0.0f), eye, s).w;
        EXPECT_GT(haze, previous) << d;
        EXPECT_NEAR(haze, 1.0f - std::exp(-d * s.atmosphere.x / s.atmosphere.y), 1e-4f) << d;
        previous = haze;
    }
    // From outside the box only the part inside counts: 0.5 units of the 3.5 to the point.
    const glm::vec3 outside(-3.0f, 0.5f, 0.0f);
    EXPECT_NEAR(tree::treeHaze(glm::vec3(-0.9f, 0.5f, 0.0f), outside, s).w,
                1.0f - std::exp(-0.5f * s.atmosphere.x / s.atmosphere.y), 1e-4f);
    // Past the box: no haze.
    EXPECT_EQ(tree::treeHaze(glm::vec3(-3.0f, 3.0f, 2.0f), outside, s).w, 0.0f);
    // Valley fog: thicker near the floor than high up, at the same distance.
    s.atmosphere.z = 4.0f;
    const float low = tree::treeHaze(glm::vec3(0.5f, 0.27f, 0.0f), glm::vec3(0.0f, 0.27f, 0.0f), s).w;
    const float high = tree::treeHaze(glm::vec3(0.5f, 1.2f, 0.0f), glm::vec3(0.0f, 1.2f, 0.0f), s).w;
    EXPECT_GT(low, 1.5f * high);
}
