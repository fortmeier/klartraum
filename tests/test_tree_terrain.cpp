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
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>

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

// A shaded height map of the soil (black outside its outline) to build/TestingOutput/tree_terrain/<name>.ppm.
void writeHeightMap(const BonsaiScene& scene, const std::string& name, int width) {
    const TreeTerrainParams& t = scene.terrain;
    const int height = int(width * t.outline.y / t.outline.x);
    const float low = t.soil.x - t.soil.z;
    const float high = t.soil.x + t.soil.y + std::max(scene.config.scale.reliefUnits(), 0.05f);
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
                const float a = std::clamp((h - low) / (high - low), 0.0f, 1.0f);
                const glm::vec3 color = glm::mix(glm::vec3(0.25f, 0.45f, 0.2f), glm::vec3(0.95f, 0.92f, 0.85f), a);
                rgb = color * (0.25f + 0.75f * std::max(glm::dot(n, light), 0.0f));
            }
            const char c[3] = {char(std::lround(255.0f * std::clamp(rgb.x, 0.0f, 1.0f))),
                               char(std::lround(255.0f * std::clamp(rgb.y, 0.0f, 1.0f))),
                               char(std::lround(255.0f * std::clamp(rgb.z, 0.0f, 1.0f)))};
            file.write(c, 3);
        }
    }
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
