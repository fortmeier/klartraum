// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - tilesInRangeReachTheSoil: the tiles in range lie within the range of the camera and reach into the soil,
 *   nearest first
 * - residentTilesKeepTheirSlots: while the camera moves, every tile in range is in the pool, each in its own
 *   slot, tiles that stay in range keep their slot, and tiles far behind are replaced
 * - newTilesComeAFewPerFrame: at most perFrame tiles are generated per frame; the others follow in the next frames
 * - aSmallPoolCountsMissingTiles: tiles in range that find no slot are counted
 * - gpuTilesMatchTheCpu: the sites tree_place_tiles.comp writes are those of treeSiteAt() (all but a few decisions
 *   at a threshold), and the bounds contain the feet and tops of the sites
 * - noHolesWhileTheCameraMoves: flying across the region at scale 1, after each frame every tile in range is in
 *   the pool and generated; the GPU time per generated tile is printed
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <set>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/tree/tree_tiles.hpp"

using namespace klartraum;

namespace {

BonsaiScene sceneAt(float s) {
    BonsaiSceneConfig config;
    config.scale.s = s;
    config.treeCount = s > 0.3f ? 12u : 0u;
    return makeBonsaiScene(config);
}

// A flight along the pot at about 300 m above the valleys of scale 1.
glm::vec3 flight(const BonsaiScene& scene, float t) {
    const float x = -1.2f + 2.4f * t, z = 0.3f * std::sin(3.0f * t);
    return glm::vec3(x, scene.soilHeight(x, z) + 0.1f, z);
}

} // namespace

TEST(TreeTiles, tilesInRangeReachTheSoil) {
    const BonsaiScene scene = sceneAt(1.0f);
    TreeTileLayout layout;
    layout.range = 0.4f;
    TreeTileCache cache(scene.terrain, makeTreePlacement(scene), layout);
    const glm::vec3 eye(1.2f, 0.5f, 0.6f);
    const std::vector<glm::ivec2> tiles = cache.tilesInRange(eye);
    ASSERT_GT(tiles.size(), 10u);
    const float size = cache.tileSize();
    float previous = 0.0f;
    for (const glm::ivec2& t : tiles) {
        const glm::vec2 lo(t.x * size, t.y * size), hi = lo + glm::vec2(size);
        const glm::vec2 nearest = glm::clamp(glm::vec2(eye.x, eye.z), lo, hi);
        EXPECT_LE(glm::length(nearest - glm::vec2(eye.x, eye.z)), layout.range);
        EXPECT_TRUE(tree::treeTerrainRectInSoil(scene.terrain, lo, hi));
        const float d = glm::length((lo + hi) * 0.5f - glm::vec2(eye.x, eye.z));
        EXPECT_GE(d, previous - 1e-5f);
        previous = d;
    }
}

TEST(TreeTiles, residentTilesKeepTheirSlots) {
    const BonsaiScene scene = sceneAt(1.0f);
    TreeTileLayout layout;
    layout.range = 0.3f;
    layout.perFrame = 1000;
    // A pool a little larger than the tiles in range at any point of the flight, so that it has to replace tiles.
    size_t most = 0;
    for (int frame = 0; frame <= 40; ++frame) {
        most = std::max(most, TreeTileCache(scene.terrain, makeTreePlacement(scene), layout)
                                  .tilesInRange(flight(scene, frame / 40.0f))
                                  .size());
    }
    layout.slots = uint32_t(most) + 8u;
    TreeTileCache cache(scene.terrain, makeTreePlacement(scene), layout);
    std::map<std::pair<int, int>, int> before;
    for (int frame = 0; frame <= 40; ++frame) {
        const glm::vec3 eye = flight(scene, frame / 40.0f);
        cache.update(eye);
        EXPECT_EQ(cache.getMissing(), 0u);
        std::set<int> used;
        std::map<std::pair<int, int>, int> now;
        for (const glm::ivec2& t : cache.tilesInRange(eye)) {
            const int slot = cache.slotOf(t);
            ASSERT_GE(slot, 0);
            EXPECT_TRUE(used.insert(slot).second);
            now[{t.x, t.y}] = slot;
            const auto it = before.find({t.x, t.y});
            if (it != before.end()) {
                EXPECT_EQ(it->second, slot);
            }
        }
        before = now;
    }
    // The flight's start is far behind: its tiles were replaced.
    EXPECT_LT(cache.slotOf(cache.tilesInRange(flight(scene, 0.0f)).front()), 0);
}

TEST(TreeTiles, newTilesComeAFewPerFrame) {
    const BonsaiScene scene = sceneAt(1.0f);
    TreeTileLayout layout;
    layout.range = 0.3f;
    layout.perFrame = 8;
    TreeTileCache cache(scene.terrain, makeTreePlacement(scene), layout);
    const glm::vec3 eye = flight(scene, 0.5f);
    const size_t wanted = cache.tilesInRange(eye).size();
    ASSERT_GT(wanted, 16u);
    size_t generated = 0;
    int frames = 0;
    while (generated < wanted && frames < 100) {
        const size_t n = cache.update(eye).size();
        EXPECT_LE(n, 8u);
        generated += n;
        ++frames;
    }
    EXPECT_EQ(generated, wanted);
    EXPECT_EQ(cache.getMissing(), 0u);
    EXPECT_EQ(frames, int((wanted + 7) / 8));
}

TEST(TreeTiles, aSmallPoolCountsMissingTiles) {
    const BonsaiScene scene = sceneAt(1.0f);
    TreeTileLayout layout;
    layout.range = 0.3f;
    layout.slots = 4;
    TreeTileCache cache(scene.terrain, makeTreePlacement(scene), layout);
    const glm::vec3 eye = flight(scene, 0.5f);
    const size_t wanted = cache.tilesInRange(eye).size();
    cache.update(eye);
    EXPECT_EQ(cache.getMissing(), wanted - 4);
}

class TreeTilesGpuTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
    }
    void TearDown() override { frontend.reset(); }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
};

TEST_F(TreeTilesGpuTest, gpuTilesMatchTheCpu) {
    const BonsaiScene scene = sceneAt(1.0f);
    const TreePlacementParams placement = makeTreePlacement(scene);
    TreeTileLayout layout;
    layout.range = 0.15f;
    TreePlacementTiles tiles(*vc, scene.terrain, placement, layout);
    ComputeGraph graph(*vc, 1);
    graph.compileFrom(tiles.getElement());
    const glm::vec3 eye = flight(scene, 0.3f);
    tiles.update(eye);
    graph.submitAndWait(vc->getGraphicsQueue(), 0);
    const std::vector<glm::vec4> bounds = tiles.readBounds();
    size_t cells = 0, differ = 0, trees = 0;
    for (const glm::ivec2& t : tiles.getCache().tilesInRange(eye)) {
        const int slot = tiles.getCache().slotOf(t);
        ASSERT_GE(slot, 0);
        const std::vector<TreeSite> gpu = tiles.readSites(uint32_t(slot));
        const int n = int(layout.cells);
        float lowest = 1e30f, highest = 0.0f;
        int count = 0;
        for (int c = 0; c < n * n; ++c) {
            const TreeSite cpu = tree::treeSiteAt(scene.terrain, placement, t.x * n + c % n, t.y * n + c / n);
            const TreeSite& g = gpu[c];
            ++cells;
            if ((cpu.posHeight.w > 0.0f) != (g.posHeight.w > 0.0f) || cpu.kind.x != g.kind.x) {
                ++differ;
            } else if (cpu.posHeight.w > 0.0f) {
                ++trees;
                EXPECT_NEAR(g.posHeight.x, cpu.posHeight.x, 1e-6f);
                EXPECT_NEAR(g.posHeight.z, cpu.posHeight.z, 1e-6f);
                // The GPU's sines and cosines differ in the last bits: heights within 1e-5 units (3 cm).
                EXPECT_NEAR(g.posHeight.y, cpu.posHeight.y, 1e-5f);
                EXPECT_NEAR(g.posHeight.w, cpu.posHeight.w, 1e-6f);
            }
            if (g.posHeight.w > 0.0f) {
                lowest = std::min(lowest, g.posHeight.y);
                highest = std::max(highest, g.posHeight.y + g.posHeight.w);
                ++count;
            }
        }
        EXPECT_LE(bounds[slot].x, lowest);
        EXPECT_GE(bounds[slot].y, highest);
        EXPECT_EQ(int(bounds[slot].z), count);
    }
    std::printf("%zu cells, %zu trees, %zu decisions differ\n", cells, trees, differ);
    EXPECT_GT(trees, 1000u);
    EXPECT_LT(differ, cells / 500);
}

TEST_F(TreeTilesGpuTest, noHolesWhileTheCameraMoves) {
    const BonsaiScene scene = sceneAt(1.0f);
    TreeTileLayout layout;
    layout.range = 0.25f;
    TreePlacementTiles tiles(*vc, scene.terrain, makeTreePlacement(scene), layout);
    ComputeGraph graph(*vc, 1);
    graph.enableProfiling();
    graph.compileFrom(tiles.getElement());
    for (int frame = 0; frame <= 60; ++frame) {
        const glm::vec3 eye = flight(scene, frame / 60.0f);
        tiles.update(eye);
        graph.resetProfiling();
        graph.submitAndWait(vc->getGraphicsQueue(), 0);
        if (frame == 0) {
            const size_t count = tiles.getCache().tilesInRange(eye).size();
            std::printf("first frame: %zu tiles in %.2f ms of GPU time\n", count, graph.getGpuSpanMs());
        }
        EXPECT_EQ(tiles.getCache().getMissing(), 0u) << frame;
        for (const glm::ivec2& t : tiles.getCache().tilesInRange(eye)) {
            EXPECT_GE(tiles.getCache().slotOf(t), 0) << frame;
        }
    }
    // The tiles at the end hold their sites: the counts of the bounds are those of the CPU.
    const glm::vec3 eye = flight(scene, 1.0f);
    const std::vector<glm::vec4> bounds = tiles.readBounds();
    const TreePlacementParams placement = makeTreePlacement(scene);
    const int n = int(layout.cells);
    for (const glm::ivec2& t : tiles.getCache().tilesInRange(eye)) {
        const int count =
            int(treeSitesIn(scene.terrain, placement, t.x * n, t.y * n, (t.x + 1) * n, (t.y + 1) * n).size());
        EXPECT_NEAR(bounds[tiles.getCache().slotOf(t)].z, float(count), 0.01f * count + 2.0f);
    }
}
