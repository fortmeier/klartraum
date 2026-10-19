// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - deciduousLeavesFollowTheYear: no leaves in winter, budding and fresh in spring, full in summer, coloured and
 *   then dropped in fall; evergreen leaves stay
 * - leafStateIsContinuous: every leaf's state changes only a little between nearby seasons
 * - seasonPacksIntoTheRecord: freshness and fall colouring survive packing into a leaf record
 * - snowOnlyInWinter: snow cover in deep winter only, lying on what faces upwards
 * - toneCurveKeepsBrightLight: the display colour rises with the light without clipping and stays in [0, 1]
 * - generationFollowsTheSeason: the reference generation has no deciduous leaves in winter, fewer in spring than
 *   in summer, and the evergreen ones all year
 * - gpuMatchesReferenceInFall: in fall, when some leaves have dropped, the GPU writes the reference's leaves
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/tree/bonsai_scene.hpp"
#include "klartraum/tree/tree_generator.hpp"

#include "tree_test_helpers.hpp"

using namespace klartraum;
using namespace klartraum::tree;
using namespace klartraum::tree_test;

namespace {

TreeFrame seasonFrame(float season) {
    TreeFrame frame = defaultTreeFrame();
    frame.timeSeason.y = season;
    return frame;
}

} // namespace

TEST(TreeSeasons, deciduousLeavesFollowTheYear) {
    for (float r : {0.0f, 0.5f, 0.99f}) {
        EXPECT_EQ(treeLeafSeason(0.3f, r, 0.0f, true).x, 0.0f) << r;                      // winter
        EXPECT_LT(treeLeafSeason(1.0f, r, 0.0f, true).x, 1.0f) << r;                      // spring: budding
        EXPECT_GT(treeLeafSeason(1.0f, r, 0.0f, true).y, 0.9f) << r;                      // fresh
        EXPECT_EQ(treeLeafSeason(2.0f, r, 0.0f, true), glm::vec3(1.0f, 0.0f, 0.0f)) << r; // summer
        EXPECT_GT(treeLeafSeason(3.6f, r, 0.0f, true).z, 0.99f) << r;                     // coloured
        EXPECT_EQ(treeLeafSeason(3.95f, r, 0.0f, true).x, 0.0f) << r;                     // dropped
    }
    EXPECT_LT(treeLeafSeason(1.0f, 0.3f, 0.0f, true).x, treeLeafSeason(1.3f, 0.3f, 0.0f, true).x);
    // Each leaf drops at its own time.
    EXPECT_GT(treeLeafSeason(3.5f, 0.9f, 0.0f, true).x, 0.5f);
    EXPECT_EQ(treeLeafSeason(3.5f, 0.1f, 0.0f, true).x, 0.0f);
    // The type's offset shifts its seasons.
    EXPECT_EQ(treeLeafSeason(2.3f, 0.5f, 0.3f, true), treeLeafSeason(2.0f, 0.5f, 0.0f, true));
    for (float s : {0.0f, 1.0f, 2.0f, 3.0f, 3.9f}) {
        EXPECT_EQ(treeLeafSeason(s, 0.5f, 0.0f, false), glm::vec3(1.0f, 0.0f, 0.0f)) << s;
    }
}

TEST(TreeSeasons, leafStateIsContinuous) {
    for (float r : {0.1f, 0.6f}) {
        glm::vec3 previous = treeLeafSeason(0.0f, r, 0.0f, true);
        for (float s = 0.001f; s < 4.0f; s += 0.001f) {
            const glm::vec3 state = treeLeafSeason(s, r, 0.0f, true);
            EXPECT_LT(glm::length(state - previous), 0.05f) << s;
            previous = state;
        }
    }
}

TEST(TreeSeasons, seasonPacksIntoTheRecord) {
    const glm::vec3 state(0.7f, 0.25f, 0.8f);
    const glm::vec2 unpacked = treeLeafSeasonUnpack(treeLeafSeasonPack(state));
    EXPECT_NEAR(unpacked.x, 0.25f, 1.0f / 255.0f);
    EXPECT_NEAR(unpacked.y, 0.8f, 1.0f / 255.0f);
}

TEST(TreeSeasons, snowOnlyInWinter) {
    EXPECT_EQ(treeSnowCover(0.0f), 1.0f);
    EXPECT_EQ(treeSnowCover(2.0f), 0.0f);
    EXPECT_EQ(treeSnowCover(1.0f), 0.0f);
    EXPECT_GT(treeSnowCover(3.9f), 0.5f);
    EXPECT_EQ(treeSnowCover(4.0f), 1.0f);
    const glm::vec3 p(0.3f, 0.2f, 0.1f);
    EXPECT_NEAR(treeSnowOn(glm::vec3(0.0f, 1.0f, 0.0f), p, 1.0f), 1.0f, 1e-6f);
    EXPECT_EQ(treeSnowOn(glm::vec3(1.0f, 0.0f, 0.0f), p, 1.0f), 0.0f);
    EXPECT_EQ(treeSnowOn(glm::vec3(0.0f, 1.0f, 0.0f), p, 0.0f), 0.0f);
}

TEST(TreeSeasons, toneCurveKeepsBrightLight) {
    float previous = -1.0f;
    for (float light = 0.0f; light < 8.0f; light += 0.25f) {
        const float value = treeDisplayColor(glm::vec3(light), 1.4f).x;
        EXPECT_GE(value, 0.0f);
        EXPECT_LT(value, 1.0f);
        EXPECT_GT(value, previous);
        previous = value;
    }
    // Sunlit snow (0.8 albedo in about 2.0 light) and white differ.
    EXPECT_LT(treeDisplayColor(glm::vec3(1.6f), 1.4f).x, 0.99f);
}

TEST(TreeSeasons, generationFollowsTheSeason) {
    const BonsaiScene scene = makeBonsaiScene();
    // Number and total length of the leaves of a type.
    auto leavesOf = [&](float season, uint32_t type) {
        const TreeRecords r = generateTreesReference(scene.types, scene.trees, seasonFrame(season));
        std::pair<size_t, double> result{0, 0.0};
        for (const LeafRecord& l : r.leaves) {
            if (l.ids.x == type) {
                result.first++;
                result.second += l.posScale.w;
            }
        }
        return result;
    };
    // Type 0 is deciduous, type 1 (the pines) evergreen.
    EXPECT_EQ(leavesOf(0.2f, 0).first, 0u);
    EXPECT_LT(leavesOf(0.75f, 0).first, leavesOf(2.0f, 0).first);  // budding
    EXPECT_LT(leavesOf(1.0f, 0).second, leavesOf(2.0f, 0).second); // growing
    EXPECT_LT(leavesOf(3.6f, 0).first, leavesOf(2.0f, 0).first);   // dropping
    EXPECT_EQ(leavesOf(0.2f, 1).first, leavesOf(2.0f, 1).first);
}

TEST(TreeSeasons, gpuMatchesReferenceInFall) {
    HeadlessFrontend frontend;
    VulkanContext& vc = frontend.getKlartraumEngine().getVulkanContext();
    BonsaiSceneConfig config;
    config.treeCount = 8;
    const BonsaiScene scene = makeBonsaiScene(config);
    // A season where no leaf's drop threshold 3.3 + 0.6 r meets it exactly for a representable r, where a
    // rounding difference between CPU and GPU would decide whether the leaf is still there.
    const TreeFrame frame = seasonFrame(3.4537f);
    auto generator = std::make_shared<TreeGenerator>(
        vc, scene.types, scene.trees, TreeCapacity::fromReference(scene.types, scene.trees, frame), frame);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(generator);
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    TreeRecords gpu = generator->readRecords();
    TreeRecords cpu = generateTreesReference(scene.types, scene.trees, frame);
    ASSERT_EQ(gpu.leaves.size(), cpu.leaves.size());
    auto key = [](const LeafRecord& l) { return std::make_pair(l.ids.z, l.ids.y); };
    std::sort(gpu.leaves.begin(), gpu.leaves.end(), [&](auto& a, auto& b) { return key(a) < key(b); });
    std::sort(cpu.leaves.begin(), cpu.leaves.end(), [&](auto& a, auto& b) { return key(a) < key(b); });
    for (size_t i = 0; i < gpu.leaves.size(); ++i) {
        ASSERT_EQ(gpu.leaves[i].ids, cpu.leaves[i].ids);
        ASSERT_NEAR(gpu.leaves[i].posScale.w, cpu.leaves[i].posScale.w, 1e-5f);
    }
}
