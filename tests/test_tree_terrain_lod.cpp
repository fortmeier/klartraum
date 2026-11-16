// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - nodeSizesAndRangesDouble: each level's nodes and range are twice the finer level's; the root covers the soil;
 *   the morph rises from 0 to 1 within the band below a level's range
 * - boundsContainTheTerrain: the pyramid's finest cells contain the terrain's heights at random points, and each
 *   coarser cell contains its children
 * - selectionCoversTheSoilWithoutOverlap: seen from above, every point of the soil lies in exactly one selected
 *   node
 * - neighbouringNodesDifferByAtMostOneLevel: nodes that share an edge differ by at most one level, also with the
 *   camera at ground level in the mountains
 * - levelsGrowWithDistance: no node is finer than its distance needs, and the mean level grows with the distance;
 *   a map of the selected nodes is written to build/TestingOutput/tree_terrain/
 * - nodesOutsideTheViewAreCulled: no selected node lies outside the view; looking along the ground, nodes behind
 *   the camera are not selected and fewer nodes are selected than from above
 * - gpuSelectionMatchesTheCpu: the nodes tree_terrain_select.comp selects are those of
 *   selectTerrainNodesReference() for views from outside the pot and at ground level, with indirect draw arguments
 *   for all of them
 * - narrowViewsSelectFinerNodes: with a narrower field of view the same place gets finer nodes
 * - nodeCountIsBounded: at ground level at scale 1 the selection needs a few hundred nodes, without overflows
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/tree/bonsai_scene.hpp"
#include "klartraum/tree/tree_terrain.hpp"

#include "tree_test_helpers.hpp"

using namespace klartraum;
using namespace klartraum::tree_test;

namespace {

BonsaiScene sceneAt(float s, uint32_t seed = 7) {
    BonsaiSceneConfig config;
    config.scale.s = s;
    config.seed = seed;
    config.treeCount = s > 0.3f ? 12u : 0u;
    return makeBonsaiScene(config);
}

TreeTerrainConfig configFor(const BonsaiScene& scene, const TreeTerrainLodConfig& lod = {}) {
    TreeTerrainConfig c{};
    c.terrain = scene.terrain;
    c.lod = makeTreeTerrainLod(scene.terrain, lod);
    return c;
}

struct View {
    glm::vec3 eye;
    glm::vec3 target;
    glm::mat4 viewProj;
};

View viewFrom(glm::vec3 eye, glm::vec3 target, float aspect = 16.0f / 9.0f) {
    CameraMVP ubo{};
    setCamera(ubo, eye, target, 40.0f, aspect, 1e-5f, 100.0f);
    return {eye, target, ubo.proj * ubo.view};
}

// The example's default view of the pot, from outside.
View outsideView() {
    const glm::vec3 focus(0.0f, 0.8f, 0.0f);
    const float d = 4.2f, elevation = 0.32f, azimuth = 0.35f;
    return viewFrom(focus + d * glm::vec3(std::cos(elevation) * std::sin(azimuth), std::sin(elevation),
                                          std::cos(elevation) * std::cos(azimuth)),
                    focus);
}

// Standing 2 m above the ground at scale 1 (0.0007 units), looking along the ground.
View groundView(const BonsaiScene& scene, glm::vec2 at = glm::vec2(0.35f, 0.15f),
                glm::vec2 towards = glm::vec2(-0.9f, -0.3f)) {
    const float above = 2.0f / scene.config.scale.metersPerUnit();
    const glm::vec3 eye(at.x, scene.soilHeight(at.x, at.y) + above, at.y);
    return viewFrom(eye, glm::vec3(towards.x, eye.y, towards.y));
}

// From high above the middle, looking straight down at the whole soil.
View topView() { return viewFrom(glm::vec3(0.0f, 6.0f, 0.001f), glm::vec3(0.0f)); }

glm::vec3 boxLo(const TreeTerrainConfig& c, const std::vector<glm::vec2>& bounds, const TreeTerrainNode& n) {
    const uint32_t level = uint32_t(n.w);
    const uint32_t ix = uint32_t(std::lround((n.x - c.lod.root.x) / n.z));
    const uint32_t iz = uint32_t(std::lround((n.y - c.lod.root.y) / n.z));
    return glm::vec3(n.x, bounds[tree::treeTerrainBoundsIndex(c.lod, ix, iz, level)].x, n.y);
}

glm::vec3 boxHi(const TreeTerrainConfig& c, const std::vector<glm::vec2>& bounds, const TreeTerrainNode& n) {
    const uint32_t level = uint32_t(n.w);
    const uint32_t ix = uint32_t(std::lround((n.x - c.lod.root.x) / n.z));
    const uint32_t iz = uint32_t(std::lround((n.y - c.lod.root.y) / n.z));
    return glm::vec3(n.x + n.z, bounds[tree::treeTerrainBoundsIndex(c.lod, ix, iz, level)].y, n.y + n.z);
}

bool sameNodes(std::vector<TreeTerrainNode> a, std::vector<TreeTerrainNode> b) {
    auto less = [](const glm::vec4& p, const glm::vec4& q) {
        return std::tie(p.w, p.x, p.y, p.z) < std::tie(q.w, q.x, q.y, q.z);
    };
    std::sort(a.begin(), a.end(), less);
    std::sort(b.begin(), b.end(), less);
    return a == b;
}

// The selected nodes from above, coloured by level (fine red to coarse blue) with dark edges, the camera white.
void writeNodeMap(const TreeTerrainConfig& c, const std::vector<TreeTerrainNode>& nodes, glm::vec3 eye,
                  const std::string& name, int width) {
    const int height = int(width * c.terrain.outline.y / c.terrain.outline.x);
    std::vector<glm::vec3> image(size_t(width) * height, glm::vec3(0.0f));
    const float step = 2.0f * c.terrain.outline.x / width;
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            const float x = -c.terrain.outline.x + (i + 0.5f) * step;
            const float z = -c.terrain.outline.y + (j + 0.5f) * step;
            for (const TreeTerrainNode& n : nodes) {
                if (x >= n.x && x < n.x + n.z && z >= n.y && z < n.y + n.z) {
                    const float t = n.w / float(c.lod.levels.x - 1u);
                    glm::vec3 color = glm::mix(glm::vec3(0.95f, 0.3f, 0.2f), glm::vec3(0.2f, 0.4f, 0.95f), t);
                    const float edge = std::min(std::min(x - n.x, n.x + n.z - x), std::min(z - n.y, n.y + n.z - z));
                    color *=
                        edge < 0.6f * step ? 0.3f : (tree::treeTerrainRadius(c.terrain, x, z) < 1.0f ? 1.0f : 0.5f);
                    image[size_t(j) * width + i] = color;
                }
            }
            if (glm::length(glm::vec2(x - eye.x, z - eye.z)) < 3.0f * step) {
                image[size_t(j) * width + i] = glm::vec3(1.0f);
            }
        }
    }
    std::filesystem::create_directories("build/TestingOutput/tree_terrain");
    std::ofstream file("build/TestingOutput/tree_terrain/" + name + ".ppm", std::ios::binary);
    file << "P6\n" << width << " " << height << "\n255\n";
    for (const glm::vec3& v : image) {
        const char rgb[3] = {char(std::lround(255.0f * v.x)), char(std::lround(255.0f * v.y)),
                             char(std::lround(255.0f * v.z))};
        file.write(rgb, 3);
    }
}

} // namespace

TEST(TreeTerrainLod, nodeSizesAndRangesDouble) {
    const BonsaiScene scene = sceneAt(1.0f);
    const TreeTerrainConfig c = configFor(scene);
    EXPECT_EQ(c.lod.levels.x, 12u);
    // The root covers the soil's bounding box.
    EXPECT_LE(c.lod.root.x, -scene.terrain.outline.x);
    EXPECT_LE(c.lod.root.y, -scene.terrain.outline.y);
    EXPECT_GE(c.lod.root.x + c.lod.root.z, scene.terrain.outline.x);
    EXPECT_GE(c.lod.root.y + c.lod.root.z, scene.terrain.outline.y);
    EXPECT_FLOAT_EQ(tree::treeTerrainNodeSize(c.lod, 11u), c.lod.root.z);
    for (uint32_t level = 1; level < c.lod.levels.x; ++level) {
        EXPECT_FLOAT_EQ(tree::treeTerrainNodeSize(c.lod, level), 2.0f * tree::treeTerrainNodeSize(c.lod, level - 1));
        EXPECT_FLOAT_EQ(tree::treeTerrainRange(c.lod, level), 2.0f * tree::treeTerrainRange(c.lod, level - 1));
        const float range = tree::treeTerrainRange(c.lod, level);
        EXPECT_EQ(tree::treeTerrainMorph(c.lod, level, 0.5f * range), 0.0f);
        EXPECT_EQ(tree::treeTerrainMorph(c.lod, level, range), 1.0f);
        float previous = 0.0f;
        for (float d = 0.5f; d <= 1.0f; d += 0.01f) {
            const float k = tree::treeTerrainMorph(c.lod, level, d * range);
            EXPECT_GE(k, previous);
            previous = k;
        }
    }
}

TEST(TreeTerrainLod, boundsContainTheTerrain) {
    const BonsaiScene scene = sceneAt(1.0f);
    const TreeTerrainConfig c = configFor(scene);
    const std::vector<glm::vec2> bounds = buildTerrainBounds(c);
    const uint32_t finest = c.lod.levels.y - 1u;
    const uint32_t cells = 1u << finest;
    const float cell = c.lod.root.z / float(cells);
    for (uint32_t k = 0; k < 20000; ++k) {
        const float x = c.lod.root.x + c.lod.root.z * tree::treeRand(3u, 2u * k);
        const float z = c.lod.root.y + c.lod.root.z * tree::treeRand(3u, 2u * k + 1u);
        const uint32_t cx = std::min(uint32_t((x - c.lod.root.x) / cell), cells - 1u);
        const uint32_t cz = std::min(uint32_t((z - c.lod.root.y) / cell), cells - 1u);
        const glm::vec2 b = bounds[((1u << (2u * finest)) - 1u) / 3u + cz * cells + cx];
        const float h = tree::treeTerrainHeight(c.terrain, x, z);
        EXPECT_GE(h, b.x) << x << " " << z;
        EXPECT_LE(h, b.y) << x << " " << z;
    }
    for (uint32_t p = 0; p < finest; ++p) {
        const uint32_t n = 1u << p;
        for (uint32_t cz = 0; cz < n; ++cz) {
            for (uint32_t cx = 0; cx < n; ++cx) {
                const glm::vec2 parent = bounds[((1u << (2u * p)) - 1u) / 3u + cz * n + cx];
                for (uint32_t k = 0; k < 4; ++k) {
                    const glm::vec2 child =
                        bounds[((1u << (2u * (p + 1))) - 1u) / 3u + (2u * cz + k / 2u) * 2u * n + 2u * cx + k % 2u];
                    EXPECT_LE(parent.x, child.x);
                    EXPECT_GE(parent.y, child.y);
                }
            }
        }
    }
}

TEST(TreeTerrainLod, selectionCoversTheSoilWithoutOverlap) {
    for (float s : {0.0f, 1.0f}) {
        const BonsaiScene scene = sceneAt(s);
        const TreeTerrainConfig c = configFor(scene);
        const std::vector<glm::vec2> bounds = buildTerrainBounds(c);
        for (const View& view : {topView(), outsideView()}) {
            const std::vector<TreeTerrainNode> nodes = selectTerrainNodesReference(c, bounds, view.viewProj, view.eye);
            ASSERT_FALSE(nodes.empty());
            int checked = 0;
            for (int j = 0; j < 97; ++j) {
                for (int i = 0; i < 163; ++i) {
                    // Points off the nodes' edges, so each lies inside exactly one node of a tiling.
                    const float x = c.terrain.outline.x * (2.0f * (i + 0.4321f) / 163.0f - 1.0f);
                    const float z = c.terrain.outline.y * (2.0f * (j + 0.3719f) / 97.0f - 1.0f);
                    if (tree::treeTerrainRadius(c.terrain, x, z) >= 1.0f) {
                        continue;
                    }
                    // Only points the view sees count.
                    const glm::vec4 clip = view.viewProj * glm::vec4(x, scene.soilHeight(x, z), z, 1.0f);
                    if (clip.w <= 0.0f || std::abs(clip.x) > 0.95f * clip.w || std::abs(clip.y) > 0.95f * clip.w) {
                        continue;
                    }
                    int containing = 0;
                    for (const TreeTerrainNode& n : nodes) {
                        containing += (x >= n.x && x < n.x + n.z && z >= n.y && z < n.y + n.z) ? 1 : 0;
                    }
                    EXPECT_EQ(containing, 1) << s << " " << x << " " << z;
                    ++checked;
                }
            }
            EXPECT_GT(checked, 1000) << s;
        }
    }
}

TEST(TreeTerrainLod, neighbouringNodesDifferByAtMostOneLevel) {
    const BonsaiScene scene = sceneAt(1.0f);
    const TreeTerrainConfig c = configFor(scene);
    const std::vector<glm::vec2> bounds = buildTerrainBounds(c);
    for (const View& view : {outsideView(), groundView(scene), groundView(scene, {-0.6f, 0.3f}, {0.8f, -0.2f})}) {
        const std::vector<TreeTerrainNode> nodes = selectTerrainNodesReference(c, bounds, view.viewProj, view.eye);
        int pairs = 0;
        for (size_t a = 0; a < nodes.size(); ++a) {
            for (size_t b = a + 1; b < nodes.size(); ++b) {
                const TreeTerrainNode& p = nodes[a];
                const TreeTerrainNode& q = nodes[b];
                const float eps = 1e-6f;
                // Overlap of their extents along x and z; neighbours touch along one and overlap along the other.
                const float ox = std::min(p.x + p.z, q.x + q.z) - std::max(p.x, q.x);
                const float oz = std::min(p.y + p.z, q.y + q.z) - std::max(p.y, q.y);
                const bool touching = (std::abs(ox) < eps && oz > eps) || (std::abs(oz) < eps && ox > eps);
                if (touching) {
                    EXPECT_LE(std::abs(p.w - q.w), 1.0f) << p.w << " " << q.w;
                    ++pairs;
                }
            }
        }
        EXPECT_GT(pairs, 10);
    }
}

TEST(TreeTerrainLod, levelsGrowWithDistance) {
    const BonsaiScene scene = sceneAt(1.0f);
    const TreeTerrainConfig c = configFor(scene);
    const std::vector<glm::vec2> bounds = buildTerrainBounds(c);
    const View view = groundView(scene);
    const std::vector<TreeTerrainNode> nodes = selectTerrainNodesReference(c, bounds, view.viewProj, view.eye);
    std::map<int, std::pair<double, int>> byLevel; // level -> sum of distances, count
    for (const TreeTerrainNode& n : nodes) {
        const uint32_t level = uint32_t(n.w);
        const float d = tree::treeBoxDistance(view.eye, boxLo(c, bounds, n), boxHi(c, bounds, n));
        if (level > 0) {
            // Not finer than needed: a finer level would have been drawn closer.
            EXPECT_GT(d, tree::treeTerrainRange(c.lod, level - 1));
        }
        byLevel[int(level)].first += d;
        byLevel[int(level)].second += 1;
    }
    // The camera stands on the ground: the finest level is there, and the mean distance grows with the level.
    EXPECT_GT(byLevel.count(0), 0u);
    double previous = -1.0;
    for (const auto& [level, sum] : byLevel) {
        const double mean = sum.first / sum.second;
        EXPECT_GT(mean, previous) << level;
        previous = mean;
    }
    EXPECT_GE(byLevel.size(), 8u);
    writeNodeMap(c, nodes, view.eye, "nodes_ground_s100", 900);
}

TEST(TreeTerrainLod, nodesOutsideTheViewAreCulled) {
    const BonsaiScene scene = sceneAt(1.0f);
    const TreeTerrainConfig c = configFor(scene);
    const std::vector<glm::vec2> bounds = buildTerrainBounds(c);
    const View view = groundView(scene);
    const std::vector<TreeTerrainNode> nodes = selectTerrainNodesReference(c, bounds, view.viewProj, view.eye);
    const glm::vec3 forward = glm::normalize(view.target - view.eye);
    for (const TreeTerrainNode& n : nodes) {
        const glm::vec3 lo = boxLo(c, bounds, n), hi = boxHi(c, bounds, n);
        EXPECT_FALSE(tree::treeBoxOutsideFrustum(view.viewProj, lo, hi));
        // Nothing entirely behind the camera.
        const glm::vec3 far(forward.x >= 0.0f ? hi.x : lo.x, 0.0f, forward.z >= 0.0f ? hi.z : lo.z);
        EXPECT_GT(glm::dot(glm::vec3(far.x - view.eye.x, 0.0f, far.z - view.eye.z), forward), 0.0f);
    }
    // The same place, looking all around from above: more nodes.
    const View above = viewFrom(view.eye + glm::vec3(0.0f, 0.01f, 0.0f), view.eye + glm::vec3(0.0f, 0.0f, 0.0001f));
    EXPECT_LT(nodes.size(), selectTerrainNodesReference(c, bounds, above.viewProj, above.eye).size());
}

class TreeTerrainGpuTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
    }
    void TearDown() override { frontend.reset(); }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
};

TEST_F(TreeTerrainGpuTest, gpuSelectionMatchesTheCpu) {
    for (float s : {0.0f, 1.0f}) {
        const BonsaiScene scene = sceneAt(s);
        auto camera = std::make_shared<CameraUboType>();
        TreeTerrain terrain(*vc, scene.terrain, camera);
        ComputeGraph graph(*vc, 1);
        graph.compileFrom(terrain.getSelect());
        std::vector<View> views = {outsideView(), topView()};
        if (s > 0.5f) {
            views.push_back(groundView(scene));
            views.push_back(groundView(scene, {-0.6f, 0.3f}, {0.8f, -0.2f}));
        }
        for (const View& view : views) {
            setCamera(camera->ubo, view.eye, view.target, 40.0f, 16.0f / 9.0f, 1e-5f, 100.0f);
            camera->update(0);
            graph.submitAndWait(vc->getGraphicsQueue(), 0);
            const std::vector<TreeTerrainNode> gpu = terrain.readNodes();
            const std::vector<TreeTerrainNode> cpu = selectTerrainNodesReference(
                terrain.getConfig(), terrain.getBounds(), camera->ubo.proj * camera->ubo.view, view.eye,
                tree::treeTerrainZoom(camera->ubo.proj));
            EXPECT_TRUE(sameNodes(gpu, cpu)) << s << ": " << gpu.size() << " vs " << cpu.size();
            const std::vector<uint32_t> stats = terrain.readStats();
            EXPECT_EQ(stats[kTreeTerrainStatSelected], cpu.size());
            EXPECT_EQ(stats[kTreeTerrainStatQueueOverflow], 0u);
            EXPECT_EQ(stats[kTreeTerrainStatNodeOverflow], 0u);
            auto args = std::dynamic_pointer_cast<DrawIndirectCommandBufferElement>(terrain.getDrawArgs().buffer());
            std::vector<VkDrawIndirectCommand> command(1);
            args->getBuffer(0).memcopyTo(command);
            EXPECT_EQ(command[0].vertexCount, 6u * 32u * 32u);
            EXPECT_EQ(command[0].instanceCount, cpu.size());
        }
    }
}

TEST_F(TreeTerrainGpuTest, nodeCountIsBounded) {
    const BonsaiScene scene = sceneAt(1.0f);
    auto camera = std::make_shared<CameraUboType>();
    TreeTerrain terrain(*vc, scene.terrain, camera);
    ComputeGraph graph(*vc, 1);
    graph.compileFrom(terrain.getSelect());
    for (const View& view : {groundView(scene), groundView(scene, {-0.6f, 0.3f}, {0.8f, -0.2f}), outsideView()}) {
        setCamera(camera->ubo, view.eye, view.target, 40.0f, 16.0f / 9.0f, 1e-5f, 100.0f);
        camera->update(0);
        graph.submitAndWait(vc->getGraphicsQueue(), 0);
        const std::vector<uint32_t> stats = terrain.readStats();
        std::printf("selected %u culled %u\n", stats[kTreeTerrainStatSelected], stats[kTreeTerrainStatCulled]);
        EXPECT_GT(stats[kTreeTerrainStatSelected], 10u);
        EXPECT_LT(stats[kTreeTerrainStatSelected], 1500u);
        EXPECT_EQ(stats[kTreeTerrainStatQueueOverflow], 0u);
        EXPECT_EQ(stats[kTreeTerrainStatNodeOverflow], 0u);
    }
}

TEST(TreeTerrainLod, narrowViewsSelectFinerNodes) {
    const BonsaiScene scene = sceneAt(1.0f);
    const TreeTerrainConfig c = configFor(scene);
    const std::vector<glm::vec2> bounds = buildTerrainBounds(c);
    const glm::vec3 eye(0.0f, 8.0f, 2.0f);
    float meanLevel[2];
    for (int narrow = 0; narrow < 2; ++narrow) {
        CameraMVP ubo{};
        setCamera(ubo, eye, glm::vec3(0.0f), narrow ? 10.0f : 40.0f, 16.0f / 9.0f, 1e-5f, 100.0f);
        const std::vector<TreeTerrainNode> nodes =
            selectTerrainNodesReference(c, bounds, ubo.proj * ubo.view, eye, tree::treeTerrainZoom(ubo.proj));
        double sum = 0.0;
        for (const TreeTerrainNode& n : nodes) {
            sum += n.w;
        }
        meanLevel[narrow] = float(sum / nodes.size());
    }
    // A quarter of the field of view: about two levels finer.
    EXPECT_NEAR(meanLevel[0] - meanLevel[1], 2.0f, 0.6f);
}
