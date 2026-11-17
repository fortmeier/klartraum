// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - floorFollowsTheScale: the copies are as large as the planting's (the lantern's moss enlarged by 1.3), cells are
 *   1.3 patch radii, the radius is 12 m at every scale, and the grid holds every cell within it
 * - copiesLieOnTheForestFloor: copies stand where the forest grows, on the ground and in the middle of their cell,
 *   with the ground's slope; none on open land
 * - copiesStayWhileTheCameraMoves: a cell's copy does not depend on where the camera is (only its fading does), and
 *   another seed gives other copies
 * - copiesFadeOutWithDistance: copies are opaque up to 70 % of the radius, fade out beyond and are gone past it
 * - densityFollowsTheForest: over a stretch of forest the share of cells with a copy is the forest weight (+- 5 %)
 * - placementMatchesTheCpu: in a forest at scale 1 the copies tree_floor_place.comp writes are the CPU's, with how
 *   much sun the terrain takes away in [0, 1]
 * - noFloorFromHighAbove: once the camera is told to be 100 m above the ground (setView()), no copy is placed
 * - theRenderersDrawTheFloor: standing in a forest at scale 1 at 1920 x 1080, both renderers place copies on the
 *   forest floor and fit them into their pool, none from 100 m above; a planting has no floor
 * - floorRendersMoss: drawn alone over black from 1.7 m above the forest floor at scale 1, the copies cover most of
 *   the ground near the camera in a mossy green, all within the pool; an image and GPU times are written
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>

#include "klartraum/rendering/gaussian_data_standard.hpp"
#include "klartraum/rendering/gaussian_splatting_factory.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/tree/bonsai_renderer.hpp"
#include "klartraum/tree/tree_floor.hpp"

#include "tree_test_helpers.hpp"

using namespace klartraum;
using namespace klartraum::tree_test;

namespace {

BonsaiScene sceneAt(float s, uint32_t seed = 7) {
    BonsaiSceneConfig config;
    config.scale.s = s;
    config.seed = seed;
    return makeBonsaiScene(config);
}

const std::vector<MossPatch>& patches() {
    static const std::vector<MossPatch> p = extractMossPatches(loadGaussiansSpz("data/lantern.spz", true));
    return p;
}

TreeFloorSetup floorOf(const BonsaiScene& scene, TreeFloorConfig config = {}) {
    return makeTreeFloor(scene, uint32_t(patches().size()), mossPatchRadius(patches()), 1.3f * scene.lanternScale,
                         config);
}

// A point in flat forest of `scene`: of a grid of samples, the one with the most forest where the ground rises less
// than 1 in 10 (in the pot).
glm::vec3 forestFloor(const BonsaiScene& scene) {
    glm::vec3 best(0.0f);
    float most = -1.0f;
    for (int j = -20; j <= 20; ++j) {
        for (int i = -30; i <= 30; ++i) {
            const float x = 0.04f * i + 0.013f, z = 0.025f * j + 0.007f;
            if (tree::treeTerrainRadius(scene.terrain, x, z) > 0.8f) {
                continue;
            }
            const float e = 0.002f, y = tree::treeTerrainHeight(scene.terrain, x, z);
            const float dx = tree::treeTerrainHeight(scene.terrain, x + e, z) - y;
            const float dz = tree::treeTerrainHeight(scene.terrain, x, z + e) - y;
            if (std::sqrt(dx * dx + dz * dz) > 0.1f * e) {
                continue;
            }
            const float w = tree::treeLandCover(scene.terrain, x, z).vegetation.x;
            if (w > most) {
                most = w;
                best = glm::vec3(x, y, z);
            }
        }
    }
    return best;
}

float fadeOf(const TreeLodInstance& inst) { return glm::uintBitsToFloat(inst.ids.y); }

} // namespace

TEST(TreeFloor, floorFollowsTheScale) {
    const float radius = mossPatchRadius(patches());
    EXPECT_GT(radius, 0.03f);
    EXPECT_LT(radius, 0.15f);
    for (float s : {0.5f, 0.75f, 1.0f}) {
        const BonsaiScene scene = sceneAt(s);
        const TreeFloorSetup f = floorOf(scene);
        const float mpu = scene.config.scale.metersPerUnit();
        EXPECT_FLOAT_EQ(f.grid.z, 1.3f * scene.lanternScale) << s;
        EXPECT_NEAR(f.grid.x, 1.3f * radius * f.grid.z, 1e-6f) << s;
        EXPECT_NEAR(f.grid.y * mpu, 12.0f, 1e-3f) << s;
        std::printf("s = %.2f: patches %.2f m across, cells %.2f m, %u x %u cells\n", s, 2.0f * radius * f.grid.z * mpu,
                    f.grid.x * mpu, f.counts.x, f.counts.x);
        EXPECT_EQ(f.counts.y, uint32_t(patches().size()));
        for (const glm::vec3 eye : {glm::vec3(0.0f), glm::vec3(0.1234f, 0.0f, -0.4321f)}) {
            const glm::ivec2 o = treeFloorGridOrigin(f, eye);
            const int n = int(f.counts.x);
            for (const glm::vec2 d : {glm::vec2(-1, 0), glm::vec2(1, 0), glm::vec2(0, -1), glm::vec2(0, 1)}) {
                const int cx = int(std::floor((eye.x + d.x * f.grid.y) / f.grid.x));
                const int cz = int(std::floor((eye.z + d.y * f.grid.y) / f.grid.x));
                EXPECT_GE(cx, o.x);
                EXPECT_LT(cx, o.x + n);
                EXPECT_GE(cz, o.y);
                EXPECT_LT(cz, o.y + n);
            }
        }
    }
}

TEST(TreeFloor, copiesLieOnTheForestFloor) {
    const BonsaiScene scene = sceneAt(1.0f);
    const TreeFloorSetup f = floorOf(scene);
    int copies = 0, open = 0;
    for (const glm::vec3 eye : {forestFloor(scene), glm::vec3(0.0f), glm::vec3(0.5f, 0.0f, 0.2f)}) {
        const glm::ivec2 o = treeFloorGridOrigin(f, eye);
        for (int j = 0; j < int(f.counts.x); ++j) {
            for (int i = 0; i < int(f.counts.x); ++i) {
                const int cx = o.x + i, cz = o.y + j;
                const TreeLodInstance c = tree::treeFloorCell(f, cx, cz, eye);
                if (c.plane.z <= 0.0f) {
                    continue;
                }
                ++copies;
                const float x = c.posYaw.x, z = c.posYaw.z;
                ASSERT_GE(x, (cx + 0.2f) * f.grid.x - 1e-6f);
                ASSERT_LE(x, (cx + 0.8f) * f.grid.x + 1e-6f);
                ASSERT_GE(z, (cz + 0.2f) * f.grid.x - 1e-6f);
                ASSERT_LE(z, (cz + 0.8f) * f.grid.x + 1e-6f);
                ASSERT_NEAR(c.posYaw.y, tree::treeTerrainHeight(scene.terrain, x, z), 1e-6f);
                ASSERT_FLOAT_EQ(c.plane.z, f.grid.z);
                ASSERT_LT(c.ids.x, f.counts.y);
                const float e = 0.5f * f.grid.x;
                ASSERT_NEAR(c.plane.x, (tree::treeTerrainHeight(scene.terrain, x + e, z) - c.posYaw.y) / e, 1e-4f);
                const TreeLandCover cover = tree::treeLandCover(scene.terrain, x, z);
                ASSERT_GT(cover.vegetation.x, 0.0f);
                open += cover.vegetation.x < 1e-3f;
            }
        }
    }
    EXPECT_GT(copies, 300);
    EXPECT_EQ(open, 0);
}

TEST(TreeFloor, copiesStayWhileTheCameraMoves) {
    const BonsaiScene scene = sceneAt(1.0f);
    const TreeFloorSetup f = floorOf(scene);
    TreeFloorSetup other = f;
    other.counts.z += 1u;
    const glm::vec3 eye = forestFloor(scene);
    const glm::vec3 moved = eye + glm::vec3(0.2f * f.grid.y, 0.0f, -0.15f * f.grid.y);
    const glm::ivec2 o = treeFloorGridOrigin(f, eye);
    int compared = 0, differ = 0;
    for (int j = 0; j < int(f.counts.x); ++j) {
        for (int i = 0; i < int(f.counts.x); ++i) {
            const TreeLodInstance a = tree::treeFloorCell(f, o.x + i, o.y + j, eye);
            const TreeLodInstance b = tree::treeFloorCell(f, o.x + i, o.y + j, moved);
            if (a.plane.z <= 0.0f || b.plane.z <= 0.0f) {
                continue;
            }
            ++compared;
            ASSERT_EQ(a.posYaw, b.posYaw);
            ASSERT_EQ(a.plane, b.plane);
            ASSERT_EQ(a.ids.x, b.ids.x);
            const TreeLodInstance c = tree::treeFloorCell(other, o.x + i, o.y + j, eye);
            differ += c.plane.z <= 0.0f || c.posYaw != a.posYaw;
        }
    }
    EXPECT_GT(compared, 100);
    EXPECT_GT(differ, compared * 9 / 10);
}

TEST(TreeFloor, copiesFadeOutWithDistance) {
    const BonsaiScene scene = sceneAt(1.0f);
    const TreeFloorSetup f = floorOf(scene);
    const glm::vec3 eye = forestFloor(scene);
    int fading = 0, copies = 0;
    for (const TreeLodInstance& c : treeFloorAround(f, eye)) {
        if (c.plane.z <= 0.0f) {
            continue;
        }
        ++copies;
        const float d = glm::length(glm::vec2(c.posYaw.x - eye.x, c.posYaw.z - eye.z));
        ASSERT_LE(d, f.grid.y);
        if (d <= 0.7f * f.grid.y) {
            ASSERT_EQ(fadeOf(c), 0.0f);
        } else {
            ASSERT_GT(fadeOf(c), 0.0f);
            ASSERT_LE(fadeOf(c), 1.0f);
            ++fading;
        }
        if (d > 0.97f * f.grid.y) {
            ASSERT_GT(fadeOf(c), 0.95f);
        }
    }
    EXPECT_GT(copies, 300);
    EXPECT_GT(fading, copies / 4);
}

TEST(TreeFloor, densityFollowsTheForest) {
    const BonsaiScene scene = sceneAt(1.0f);
    // Small cells, so that the stretch holds many.
    TreeFloorConfig config;
    config.spacing = 0.4f;
    const TreeFloorSetup f = floorOf(scene, config);
    const glm::vec3 eye = forestFloor(scene);
    const std::vector<TreeLodInstance> copies = treeFloorAround(f, eye);
    const glm::ivec2 o = treeFloorGridOrigin(f, eye);
    double expected = 0.0;
    int found = 0;
    const int n = int(f.counts.x);
    for (int j = n / 4; j < 3 * n / 4; ++j) {
        for (int i = n / 4; i < 3 * n / 4; ++i) {
            const float x = (o.x + i + 0.5f) * f.grid.x, z = (o.y + j + 0.5f) * f.grid.x;
            if (glm::length(glm::vec2(x - eye.x, z - eye.z)) > 0.9f * f.grid.y) {
                continue;
            }
            expected += tree::treeLandCover(scene.terrain, x, z).vegetation.x;
            found += copies[size_t(j) * n + i].plane.z > 0.0f;
        }
    }
    ASSERT_GT(expected, 1000.0);
    EXPECT_NEAR(found / expected, 1.0, 0.05);
}

namespace {

struct Eye {
    glm::vec3 eye;
    glm::vec3 target;
};

// 1.7 m above the forest floor of `scene`, looking along it (slightly down) towards +x.
Eye standingInTheForest(const BonsaiScene& scene) {
    const glm::vec3 ground = forestFloor(scene);
    const float mpu = scene.config.scale.metersPerUnit();
    const glm::vec3 eye = ground + glm::vec3(0.0f, 1.7f / mpu, 0.0f);
    return {eye, eye + glm::vec3(10.0f / mpu, -3.5f / mpu, 3.0f / mpu)};
}

// The forest floor of a scene as Gaussians, drawn alone into a black target.
struct FloorScene {
    FloorScene(VulkanContext& vc, const BonsaiScene& scene, VkExtent2D extent, uint32_t slots = 1500000)
        : vc(vc),
          extent(extent),
          slots(slots) {
        camera = std::make_shared<CameraUboType>();
        settings = std::make_shared<HostValues<TreeRenderSettings>>(
            vc, std::vector<TreeRenderSettings>{defaultTreeRenderSettings(extent)});
        TreeTypeParams p = tree_presets::quakingAspen();
        p.ratios.x = 1.0f;
        p.leafCount.x = 0.0f;
        const std::vector<TreeTypeParams> types = {p};
        const std::vector<TreeInstance> trees = {makeInstance(0, 1, glm::vec3(10.0f, 0.0f, 0.0f), 0.01f)};
        generator = std::make_shared<TreeGenerator>(vc, types, trees,
                                                    TreeCapacity::fromReference(types, trees, defaultTreeFrame()));
        TreeGaussianSources sources;
        sources.trees = false;
        sources.extraSlots = slots;
        gaussians = std::make_unique<TreeGaussians>(vc, *generator, camera, settings, std::move(sources));
        floor = std::make_unique<TreeForestFloor>(vc, floorOf(scene), gaussians->getGaussians(), patches(), camera,
                                                  settings, BufferRef{}, BonsaiRenderConfig{}.mossLod, slots,
                                                  gaussians->getExtraOffset());
        target = std::make_shared<OffscreenTarget>(vc, extent, 1);
        splatting = createGaussianSplatting(vc, GsplatBackend::Raster, target, camera, floor->getGaussians());
        graph = std::make_unique<ComputeGraph>(vc, 1);
        graph->enableProfiling();
        graph->compileFrom(splatting);
        gaussians->initializeBuffers(1);
    }

    std::vector<uint8_t> render(const Eye& e) {
        setCamera(camera->ubo, e.eye, e.target, 50.0f, float(extent.width) / float(extent.height), 1e-5f, 100.0f);
        camera->update(0);
        graph->submitAndWait(vc.getGraphicsQueue(), 0);
        return readImage(vc, target->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
    }

    VulkanContext& vc;
    VkExtent2D extent;
    uint32_t slots;
    std::shared_ptr<CameraUboType> camera;
    std::shared_ptr<HostValues<TreeRenderSettings>> settings;
    std::shared_ptr<TreeGenerator> generator;
    std::unique_ptr<TreeGaussians> gaussians;
    std::unique_ptr<TreeForestFloor> floor;
    std::shared_ptr<OffscreenTarget> target;
    std::shared_ptr<ComputeGraphElement> splatting;
    std::unique_ptr<ComputeGraph> graph;
};

} // namespace

class TreeFloorTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
    }
    void TearDown() override { frontend.reset(); }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
};

TEST_F(TreeFloorTest, placementMatchesTheCpu) {
    const BonsaiScene scene = sceneAt(1.0f);
    FloorScene f(*vc, scene, {320, 180});
    const Eye e = standingInTheForest(scene);
    f.render(e);
    const std::vector<TreeLodInstance> gpu = f.floor->readInstances();
    const std::vector<TreeLodInstance> cpu = treeFloorAround(f.floor->getSetup(), e.eye);
    ASSERT_EQ(gpu.size(), cpu.size());
    int copies = 0, mismatched = 0;
    for (size_t i = 0; i < cpu.size(); ++i) {
        copies += cpu[i].plane.z > 0.0f;
        const bool same = (gpu[i].plane.z > 0.0f) == (cpu[i].plane.z > 0.0f) &&
                          glm::length(glm::vec3(gpu[i].posYaw) - glm::vec3(cpu[i].posYaw)) < 1e-5f &&
                          gpu[i].ids.x == cpu[i].ids.x && std::abs(fadeOf(gpu[i]) - fadeOf(cpu[i])) < 1e-3f;
        mismatched += same ? 0 : 1;
        if (gpu[i].plane.z > 0.0f) {
            const float taken = glm::uintBitsToFloat(gpu[i].ids.z);
            ASSERT_GE(taken, 0.0f);
            ASSERT_LE(taken, 1.0f);
        }
    }
    std::printf("copies: %d, mismatched %d\n", copies, mismatched);
    EXPECT_GT(copies, 300);
    // Sines and cosines of the GPU can move a copy at the edge of the radius or of the forest.
    EXPECT_LE(mismatched, copies / 200 + 2);
}

TEST_F(TreeFloorTest, noFloorFromHighAbove) {
    const BonsaiScene scene = sceneAt(1.0f);
    FloorScene f(*vc, scene, {320, 180});
    Eye e = standingInTheForest(scene);
    const glm::vec3 up(0.0f, 100.0f / scene.config.scale.metersPerUnit(), 0.0f);
    e.eye += up;
    e.target += up;
    f.floor->setView(e.eye);
    f.render(e);
    for (const TreeLodInstance& inst : f.floor->readInstances()) {
        ASSERT_EQ(inst.plane.z, 0.0f);
    }
    EXPECT_EQ(f.floor->getLod().readCounter()[0], 0u);
}

TEST_F(TreeFloorTest, floorRendersMoss) {
    const BonsaiScene scene = sceneAt(1.0f);
    const VkExtent2D extent{960, 540};
    FloorScene f(*vc, scene, extent);
    const Eye e = standingInTheForest(scene);
    const std::vector<uint8_t> image = f.render(e);
    writePpm("floor_moss", image, extent);
    const std::vector<uint32_t> counter = f.floor->getLod().readCounter();
    std::printf("floor: %u Gaussians, %u copies left out\n", counter[0], counter[1]);
    for (const auto& [name, ms] : f.graph->getProfilingResults()) {
        if (ms > 0.2f) {
            std::printf("  %s: %.2f ms\n", name.c_str(), ms);
        }
    }
    EXPECT_GT(counter[0], 10000u);
    EXPECT_EQ(counter[1], 0u);
    uint32_t covered = 0;
    glm::vec3 mean(0.0f);
    for (uint32_t y = extent.height * 4 / 5; y < extent.height; ++y) {
        for (uint32_t x = 0; x < extent.width; ++x) {
            const uint8_t* p = &image[4 * (size_t(y) * extent.width + x)];
            if (p[0] + p[1] + p[2] > 30) {
                ++covered;
                mean += glm::vec3(p[2], p[1], p[0]) / 255.0f;
            }
        }
    }
    const float share = float(covered) / float(extent.width * (extent.height / 5));
    mean /= float(std::max(covered, 1u));
    std::printf("covered %.2f near, mean (%.2f, %.2f, %.2f)\n", share, mean.x, mean.y, mean.z);
    EXPECT_GT(share, 0.8f);
    EXPECT_GT(mean.y, mean.z);
}

TEST_F(TreeFloorTest, theRenderersDrawTheFloor) {
    const BonsaiScene scene = sceneAt(1.0f);
    const VkExtent2D extent{1920, 1080};
    const Eye low = standingInTheForest(scene);
    const glm::vec3 up(0.0f, 100.0f / scene.config.scale.metersPerUnit(), 0.0f);
    for (BonsaiRenderMode mode : {BonsaiRenderMode::Classical, BonsaiRenderMode::Gaussian}) {
        auto target = std::make_shared<OffscreenTarget>(*vc, extent, 1);
        BonsaiRenderConfig config;
        config.extent = extent;
        config.mode = mode;
        BonsaiRenderer renderer(*vc, scene, target, config);
        ComputeGraph graph(*vc, 1);
        graph.compileFrom(renderer.getOutput());
        renderer.initialize(1);
        ASSERT_NE(renderer.getForestFloor(), nullptr);
        for (const glm::vec3 lift : {glm::vec3(0.0f), up}) {
            const glm::vec3 eye = low.eye + lift;
            setCamera(renderer.getCamera()->ubo, eye, low.target + lift, 50.0f, 16.0f / 9.0f, 1e-5f, 100.0f);
            renderer.getCamera()->update(0);
            renderer.setView(eye);
            graph.submitAndWait(vc->getGraphicsQueue(), 0);
            const std::vector<uint32_t> counter = renderer.getForestFloor()->getLod().readCounter();
            if (lift == glm::vec3(0.0f)) {
                std::printf(
                    "%s: floor %u Gaussians of %u; grass %u Gaussians of %u\n",
                    mode == BonsaiRenderMode::Classical ? "classical" : "Gaussian", counter[0], config.floorPool,
                    renderer.getGrassGaussians() ? renderer.getGrassGaussians()->readTaken() : 0u, config.grassPool);
                EXPECT_GT(counter[0], 10000u);
                EXPECT_EQ(counter[1], 0u);
            } else {
                EXPECT_EQ(counter[0], 0u);
            }
        }
    }
    const BonsaiScene planting = sceneAt(0.21f);
    auto target = std::make_shared<OffscreenTarget>(*vc, extent, 1);
    BonsaiRenderConfig config;
    config.extent = extent;
    BonsaiRenderer renderer(*vc, planting, target, config);
    EXPECT_EQ(renderer.getForestFloor(), nullptr);
}
