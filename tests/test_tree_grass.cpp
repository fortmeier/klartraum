// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - grassFollowsTheScale: a cell is 0.25 m and the radius 40 m at every scale, and the grid around the camera
 *   holds every cell within the radius
 * - grassGrowsOnlyOnOpenLand: clumps stand where meadow, field or scrub grows, on the ground, in their cell, 10 cm
 *   to 1.2 m tall; none in pure forest, on rock or in water
 * - clumpsStayWhileTheCameraMoves: a cell's clump does not depend on where the camera is (only its blades do), and
 *   another seed gives other clumps
 * - fewerBladesWithDistance: clumps have bladesNear blades up to a third of the radius, fewer beyond, and none
 *   beyond the radius
 * - densityFollowsTheOpenLand: over a stretch of open land the share of cells with a clump is the weight of meadow,
 *   field (but ploughed parcels) and (60 % of) scrub (+- 5 %)
 * - cropsFollowTheSeason: in grain fields the clumps grow tall in summer and are stubble in winter, and ploughed
 *   parcels have none; meadows keep their grass
 * - visibleClumpsAreThoseInTheView: standing in a meadow at scale 0.75, the clumps tree_grass.comp finds are the
 *   clumps of the CPU's grid whose box is in the view, with the ground's slope and an albedo of the open land
 * - lookingUpShowsNoGrass: looking up into the sky, no clump is visible
 * - aFullListCountsTheRest: with a small capacity the list is full, the rest is counted, and the indirect draw
 *   holds the capacity
 * - noGrassFromHighAbove: once the camera is told to be 100 m above the ground (setView()), no clump is evaluated;
 *   back down at 1.7 m the grass returns
 * - grassRendersBothWays: from 1.7 m above open land in summer, the Gaussian grass puts as much light into blocks of
 *   the image below the horizon as the classical blades (mean of all pixels, green within 0.05 per block and the
 *   colour within 0.03 overall); images and GPU times are written
 * - theRenderersDrawGrass: standing in a meadow at scale 0.75 both renderers find grass in the view (the Gaussian
 *   renderer within its pool), none from 100 m above; a planting has no grass
 * - bladesCoverTheGround: drawn alone over black from 1.7 m above open land, the blades cover much of the ground
 *   in the view and none of the sky, green in summer and lighter under snow; images and the GPU times are written
 *   (build/TestingOutput/tree/)
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>

#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/tree/bonsai_renderer.hpp"
#include "klartraum/tree/tree_gaussians.hpp"
#include "klartraum/tree/tree_grass.hpp"

#include "tree_test_helpers.hpp"

using namespace klartraum;
using namespace klartraum::tree_test;

namespace {

const float kSummer = 1.2f;

BonsaiScene sceneAt(float s, uint32_t seed = 7) {
    BonsaiSceneConfig config;
    config.scale.s = s;
    config.seed = seed;
    return makeBonsaiScene(config);
}

float openWeight(const TreeLandCover& c) { return c.vegetation.y + c.vegetation.z + 0.6f * c.vegetation.w; }

// A point on flat open land of `scene`: of a grid of samples, the one with the most open weight where the ground
// rises less than 1 in 10 (in the pot).
glm::vec3 openLand(const BonsaiScene& scene) {
    glm::vec3 best(0.0f);
    float most = -1.0f;
    for (int j = -20; j <= 20; ++j) {
        for (int i = -30; i <= 30; ++i) {
            const float x = 0.04f * i, z = 0.025f * j;
            if (tree::treeTerrainRadius(scene.terrain, x, z) > 0.8f) {
                continue;
            }
            const float e = 0.002f, y = tree::treeTerrainHeight(scene.terrain, x, z);
            const float dx = tree::treeTerrainHeight(scene.terrain, x + e, z) - y;
            const float dz = tree::treeTerrainHeight(scene.terrain, x, z + e) - y;
            if (std::sqrt(dx * dx + dz * dz) > 0.1f * e) {
                continue;
            }
            const float w = openWeight(tree::treeLandCover(scene.terrain, x, z));
            if (w > most) {
                most = w;
                best = glm::vec3(x, tree::treeTerrainHeight(scene.terrain, x, z), z);
            }
        }
    }
    return best;
}

struct Eye {
    glm::vec3 eye;
    glm::vec3 target;
};

// 1.7 m above the open land of `scene`, looking along the ground (slightly down) towards +x.
Eye standingInTheMeadow(const BonsaiScene& scene) {
    const glm::vec3 ground = openLand(scene);
    const float mpu = scene.config.scale.metersPerUnit();
    const glm::vec3 eye = ground + glm::vec3(0.0f, 1.7f / mpu, 0.0f);
    return {eye, eye + glm::vec3(10.0f / mpu, -2.5f / mpu, 3.0f / mpu)};
}

void setSeason(HostValues<TreeRenderSettings>& settings, float season) {
    TreeRenderSettings values = settings.values().at(0);
    values.season = glm::vec4(season, tree::treeSnowCover(season), 0.0f, 0.0f);
    settings.set(0, values);
}

// The grass of a scene, drawn alone into a black target.
struct GrassScene {
    GrassScene(VulkanContext& vc, const BonsaiScene& scene, VkExtent2D extent, TreeGrassConfig config = {})
        : vc(vc),
          scene(scene),
          extent(extent) {
        camera = std::make_shared<CameraUboType>();
        settings = std::make_shared<HostValues<TreeRenderSettings>>(
            vc, std::vector<TreeRenderSettings>{defaultTreeRenderSettings(extent)});
        params = makeTreeGrass(scene, config);
        grass = std::make_unique<TreeGrass>(vc, scene.terrain, params, camera, settings, extent);
        target = std::make_shared<OffscreenTarget>(vc, extent, 1);
        pass = std::make_shared<RenderPass>(vc.getSwapChainImageFormat(), extent);
        pass->setInput(target, 0);
        pass->setInput(camera, 1);
        pass->setDepthAttachment(true);
        pass->addComputeDependency(grass->getElement());
        pass->addComputeDependency(settings);
        pass->addDrawComponent(grass->getDrawComponent());
        graph = std::make_unique<ComputeGraph>(vc, 1);
        graph->enableProfiling();
        graph->compileFrom(pass);
    }

    // The same grass as Gaussians, drawn alone into a second black target.
    void addGaussians(uint32_t slots = 262144) {
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
        grassGaussians = std::make_unique<TreeGrassGaussians>(vc, *grass, gaussians->getGaussians(),
                                                              gaussians->getExtraOffset(), slots, settings, camera);
        gaussianTarget = std::make_shared<OffscreenTarget>(vc, extent, 1);
        splatting =
            createGaussianSplatting(vc, GsplatBackend::Raster, gaussianTarget, camera, grassGaussians->getGaussians());
        gaussianGraph = std::make_unique<ComputeGraph>(vc, 1);
        gaussianGraph->enableProfiling();
        gaussianGraph->compileFrom(splatting);
        gaussians->initializeBuffers(1);
    }

    std::vector<uint8_t> renderGaussians(const Eye& e) {
        setCamera(camera->ubo, e.eye, e.target, 50.0f, float(extent.width) / float(extent.height), 1e-5f, 100.0f);
        camera->update(0);
        gaussianGraph->submitAndWait(vc.getGraphicsQueue(), 0);
        return readImage(vc, gaussianTarget->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
    }

    std::vector<uint8_t> render(const Eye& e) {
        setCamera(camera->ubo, e.eye, e.target, 50.0f, float(extent.width) / float(extent.height), 1e-5f, 100.0f);
        camera->update(0);
        graph->submitAndWait(vc.getGraphicsQueue(), 0);
        return readImage(vc, target->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
    }

    // The clumps of the CPU's grid whose box is in the view, as tree_grass.comp tests it.
    std::vector<TreeGrassClump> visibleOnTheCpu(const Eye& e) {
        const glm::mat4 viewProj = camera->ubo.proj * camera->ubo.view;
        std::vector<TreeGrassClump> result;
        for (const TreeGrassClump& c :
             treeGrassAround(scene.terrain, params, e.eye, settings->values().at(0).season.x)) {
            const glm::vec3 foot(c.posHeight);
            const float h = c.posHeight.w;
            const glm::vec3 r(0.75f * params.grid.x + 0.5f * h, 0.25f * params.grid.x,
                              0.75f * params.grid.x + 0.5f * h);
            if (!tree::treeBoxOutsideFrustum(viewProj, foot - r, foot + r + glm::vec3(0.0f, 1.3f * h, 0.0f))) {
                result.push_back(c);
            }
        }
        return result;
    }

    VulkanContext& vc;
    const BonsaiScene& scene;
    VkExtent2D extent;
    TreeGrassParams params{};
    std::shared_ptr<CameraUboType> camera;
    std::shared_ptr<HostValues<TreeRenderSettings>> settings;
    std::unique_ptr<TreeGrass> grass;
    std::shared_ptr<OffscreenTarget> target;
    std::shared_ptr<RenderPass> pass;
    std::unique_ptr<ComputeGraph> graph;
    std::shared_ptr<TreeGenerator> generator;
    std::unique_ptr<TreeGaussians> gaussians;
    std::unique_ptr<TreeGrassGaussians> grassGaussians;
    std::shared_ptr<OffscreenTarget> gaussianTarget;
    std::shared_ptr<ComputeGraphElement> splatting;
    std::unique_ptr<ComputeGraph> gaussianGraph;
};

// The share of non-black pixels in rows [y0, y1) and their mean colour.
struct Coverage {
    float share = 0.0f;
    glm::vec3 mean{0.0f};
};

Coverage coverageOf(const std::vector<uint8_t>& bgra, VkExtent2D extent, uint32_t y0, uint32_t y1) {
    Coverage c;
    uint32_t covered = 0;
    for (uint32_t y = y0; y < y1; ++y) {
        for (uint32_t x = 0; x < extent.width; ++x) {
            const uint8_t* p = &bgra[4 * (size_t(y) * extent.width + x)];
            if (p[0] + p[1] + p[2] > 6) {
                ++covered;
                c.mean += glm::vec3(p[2], p[1], p[0]) / 255.0f;
            }
        }
    }
    c.share = float(covered) / float((y1 - y0) * extent.width);
    c.mean /= float(std::max(covered, 1u));
    return c;
}

} // namespace

class TreeGrassTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
    }
    void TearDown() override { frontend.reset(); }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
};

TEST(TreeGrass, grassFollowsTheScale) {
    for (float s : {0.5f, 0.75f, 1.0f}) {
        const BonsaiScene scene = sceneAt(s);
        const TreeGrassParams g = makeTreeGrass(scene);
        const float mpu = scene.config.scale.metersPerUnit();
        EXPECT_NEAR(g.grid.x * mpu, 0.25f, 1e-4f) << s;
        EXPECT_NEAR(g.grid.y * mpu, 40.0f, 1e-3f) << s;
        EXPECT_FLOAT_EQ(g.grid.w, mpu) << s;
        // The grid holds the cells of all points within the radius of any eye.
        for (const glm::vec3 eye :
             {glm::vec3(0.0f), glm::vec3(0.1234f, 0.0f, -0.4321f), glm::vec3(-0.7f, 0.0f, 0.3f)}) {
            const glm::ivec2 o = treeGrassGridOrigin(g, eye);
            const int n = int(g.counts.x);
            for (const glm::vec2 d : {glm::vec2(-1, 0), glm::vec2(1, 0), glm::vec2(0, -1), glm::vec2(0, 1)}) {
                const float x = eye.x + d.x * g.grid.y, z = eye.z + d.y * g.grid.y;
                const int cx = int(std::floor(x / g.grid.x)), cz = int(std::floor(z / g.grid.x));
                EXPECT_GE(cx, o.x);
                EXPECT_LT(cx, o.x + n);
                EXPECT_GE(cz, o.y);
                EXPECT_LT(cz, o.y + n);
            }
        }
    }
}

TEST(TreeGrass, grassGrowsOnlyOnOpenLand) {
    const BonsaiScene scene = sceneAt(0.75f);
    const TreeGrassParams g = makeTreeGrass(scene);
    int clumps = 0, forest = 0, bare = 0;
    for (const glm::vec3 eye : {openLand(scene), glm::vec3(0.0f), glm::vec3(0.5f, 0.0f, 0.2f)}) {
        const glm::ivec2 o = treeGrassGridOrigin(g, eye);
        for (int j = 0; j < int(g.counts.x); j += 3) {
            for (int i = 0; i < int(g.counts.x); i += 3) {
                const int cx = o.x + i, cz = o.y + j;
                const TreeGrassClump c = tree::treeGrassCell(scene.terrain, g, cx, cz, eye, kSummer);
                const float x = c.posHeight.x, z = c.posHeight.z;
                if (c.kind.x <= 0.0f) {
                    continue;
                }
                ++clumps;
                ASSERT_GE(x, cx * g.grid.x);
                ASSERT_LE(x, (cx + 1) * g.grid.x);
                ASSERT_GE(z, cz * g.grid.x);
                ASSERT_LE(z, (cz + 1) * g.grid.x);
                ASSERT_NEAR(c.posHeight.y, tree::treeTerrainHeight(scene.terrain, x, z), 1e-6f);
                ASSERT_GT(c.posHeight.w * g.grid.w, 0.1f);
                ASSERT_LT(c.posHeight.w * g.grid.w, 1.2f);
                const TreeLandCover cover = tree::treeLandCover(scene.terrain, x, z);
                ASSERT_GT(openWeight(cover), 0.0f);
                forest += cover.vegetation.x > 0.999f;
                bare += cover.bare.x + cover.bare.y > 0.999f;
            }
        }
    }
    EXPECT_GT(clumps, 1000);
    EXPECT_EQ(forest, 0);
    EXPECT_EQ(bare, 0);
}

TEST(TreeGrass, clumpsStayWhileTheCameraMoves) {
    const BonsaiScene scene = sceneAt(0.75f);
    const TreeGrassParams g = makeTreeGrass(scene);
    TreeGrassParams other = g;
    other.counts.z += 1u;
    const glm::vec3 eye = openLand(scene);
    const glm::vec3 moved = eye + glm::vec3(0.3f * g.grid.y, 0.0f, -0.2f * g.grid.y);
    const glm::ivec2 o = treeGrassGridOrigin(g, eye);
    int compared = 0, differ = 0;
    for (int j = 0; j < int(g.counts.x); j += 2) {
        for (int i = 0; i < int(g.counts.x); i += 2) {
            const TreeGrassClump a = tree::treeGrassCell(scene.terrain, g, o.x + i, o.y + j, eye, kSummer);
            const TreeGrassClump b = tree::treeGrassCell(scene.terrain, g, o.x + i, o.y + j, moved, kSummer);
            if (a.kind.x <= 0.0f || b.kind.x <= 0.0f) {
                continue;
            }
            ++compared;
            ASSERT_EQ(a.posHeight, b.posHeight);
            ASSERT_EQ(a.kind.y, b.kind.y);
            ASSERT_EQ(a.kind.w, b.kind.w);
            const TreeGrassClump c = tree::treeGrassCell(scene.terrain, other, o.x + i, o.y + j, eye, kSummer);
            differ += c.kind.x <= 0.0f || c.posHeight != a.posHeight;
        }
    }
    EXPECT_GT(compared, 1000);
    EXPECT_GT(differ, compared * 9 / 10);
}

TEST(TreeGrass, fewerBladesWithDistance) {
    const BonsaiScene scene = sceneAt(0.75f);
    const TreeGrassParams g = makeTreeGrass(scene);
    const glm::vec3 eye = openLand(scene);
    const std::vector<TreeGrassClump> clumps = treeGrassAround(scene.terrain, g, eye, kSummer);
    ASSERT_GT(clumps.size(), 10000u);
    float farthest = 0.0f;
    int thinned = 0;
    for (const TreeGrassClump& c : clumps) {
        const float d = glm::length(glm::vec2(c.posHeight.x - eye.x, c.posHeight.z - eye.z));
        farthest = std::max(farthest, d);
        ASSERT_LE(d, g.grid.y);
        ASSERT_GE(c.kind.x, 1.0f);
        ASSERT_LE(c.kind.x, g.grid.z);
        if (d < g.grid.y / 3.0f) {
            ASSERT_EQ(c.kind.x, g.grid.z);
        } else if (d > 0.9f * g.grid.y) {
            ASSERT_LE(c.kind.x, 0.1f * g.grid.z + 1.0f);
        }
        thinned += c.kind.x < g.grid.z;
    }
    EXPECT_GT(farthest, 0.95f * g.grid.y);
    EXPECT_GT(thinned, int(clumps.size()) / 2);
}

TEST(TreeGrass, densityFollowsTheOpenLand) {
    const BonsaiScene scene = sceneAt(0.75f);
    const TreeGrassParams g = makeTreeGrass(scene);
    const glm::vec3 eye = openLand(scene);
    const glm::ivec2 o = treeGrassGridOrigin(g, eye);
    double expected = 0.0;
    int found = 0;
    const int n = int(g.counts.x);
    for (int j = n / 4; j < 3 * n / 4; ++j) {
        for (int i = n / 4; i < 3 * n / 4; ++i) {
            const TreeGrassClump c = tree::treeGrassCell(scene.terrain, g, o.x + i, o.y + j, eye, kSummer);
            const float x = (o.x + i + 0.5f) * g.grid.x, z = (o.y + j + 0.5f) * g.grid.x;
            const TreeLandCover cover = tree::treeLandCover(scene.terrain, x, z);
            const bool ploughed = tree::treeFieldCropHeight(glm::vec2(x, z) * scene.terrain.cover.y, kSummer,
                                                            scene.terrain.seed.x) <= 0.0f;
            expected += std::min(1.0f, openWeight(cover) - (ploughed ? cover.vegetation.z : 0.0f));
            found += c.kind.x > 0.0f;
        }
    }
    ASSERT_GT(expected, 1000.0);
    EXPECT_NEAR(found / expected, 1.0, 0.05);
}

TEST_F(TreeGrassTest, visibleClumpsAreThoseInTheView) {
    const BonsaiScene scene = sceneAt(0.75f);
    GrassScene g(*vc, scene, {640, 360});
    const Eye e = standingInTheMeadow(scene);
    g.render(e);
    std::vector<TreeGrassVisible> gpu = g.grass->readVisible();
    std::vector<TreeGrassClump> cpu = g.visibleOnTheCpu(e);
    auto key = [](glm::vec4 p) { return std::make_pair(p.x, p.z); };
    std::vector<std::pair<float, float>> cpuKeys;
    for (const TreeGrassClump& c : cpu) {
        cpuKeys.push_back(key(c.posHeight));
    }
    std::sort(cpuKeys.begin(), cpuKeys.end());
    std::printf("visible clumps: %zu on the GPU, %zu on the CPU\n", gpu.size(), cpu.size());
    EXPECT_GT(cpu.size(), 5000u);
    EXPECT_NEAR(double(gpu.size()), double(cpu.size()), 0.002 * cpu.size() + 2.0);
    size_t matched = 0;
    for (const TreeGrassVisible& v : gpu) {
        matched += std::binary_search(cpuKeys.begin(), cpuKeys.end(), key(v.posHeight)) ? 1 : 0;
        // The slope of the ground at the clump, and the albedo of open land: green, not black, not white.
        const float e = 0.5f * g.params.grid.x;
        const float dx = (tree::treeTerrainHeight(scene.terrain, v.posHeight.x + e, v.posHeight.z) - v.posHeight.y) / e;
        ASSERT_NEAR(v.ground.x, dx, 0.02f + 0.02f * std::abs(dx));
        ASSERT_GT(v.albedo.y, 0.03f);
        ASSERT_LT(v.albedo.y, 0.4f);
        ASSERT_GE(v.kind.z, 0.0f);
        ASSERT_LE(v.kind.z, 1.0f);
    }
    EXPECT_GT(matched, gpu.size() * 99 / 100);
    EXPECT_EQ(g.grass->readCounter()[1], 0u);
}

TEST_F(TreeGrassTest, lookingUpShowsNoGrass) {
    const BonsaiScene scene = sceneAt(0.75f);
    GrassScene g(*vc, scene, {320, 180});
    Eye e = standingInTheMeadow(scene);
    e.target = e.eye + glm::vec3(0.001f, 1.0f, 0.0f);
    g.render(e);
    EXPECT_EQ(g.grass->readCounter()[0], 0u);
}

TEST_F(TreeGrassTest, aFullListCountsTheRest) {
    const BonsaiScene scene = sceneAt(0.75f);
    TreeGrassConfig config;
    config.clumps = 1000;
    GrassScene g(*vc, scene, {320, 180}, config);
    g.render(standingInTheMeadow(scene));
    const std::vector<uint32_t> counter = g.grass->readCounter();
    EXPECT_GT(counter[0], 1000u);
    EXPECT_EQ(counter[1], counter[0] - 1000u);
    EXPECT_EQ(g.grass->readVisible().size(), 1000u);
}

TEST_F(TreeGrassTest, noGrassFromHighAbove) {
    const BonsaiScene scene = sceneAt(0.75f);
    GrassScene g(*vc, scene, {320, 180});
    const Eye low = standingInTheMeadow(scene);
    Eye high = low;
    const glm::vec3 up(0.0f, 100.0f / scene.config.scale.metersPerUnit(), 0.0f);
    high.eye += up;
    high.target += up;
    g.grass->setView(high.eye);
    g.render(high);
    EXPECT_EQ(g.grass->readCounter()[0], 0u);
    g.grass->setView(low.eye);
    g.render(low);
    EXPECT_GT(g.grass->readCounter()[0], 5000u);
}

TEST_F(TreeGrassTest, grassRendersBothWays) {
    const BonsaiScene scene = sceneAt(0.75f);
    const VkExtent2D extent{960, 540};
    GrassScene g(*vc, scene, extent);
    g.addGaussians();
    const Eye e = standingInTheMeadow(scene);
    setSeason(*g.settings, kSummer);
    const std::vector<uint8_t> classical = g.render(e);
    const std::vector<uint8_t> gaussian = g.renderGaussians(e);
    writePpm("grass_classical", classical, extent);
    writePpm("grass_gaussian", gaussian, extent);
    std::printf("grass: %u clumps, %u Gaussian slots\n", g.grass->readCounter()[0], g.grassGaussians->readTaken());
    for (const auto& [name, ms] : g.gaussianGraph->getProfilingResults()) {
        std::printf("  %s: %.2f ms\n", name.c_str(), ms);
    }
    EXPECT_LE(g.grassGaussians->readTaken(), 262144u);

    // Below the horizon of the grass (rows from the middle down to 4/5 of the height), in blocks of 16 x 16 pixels:
    // the mean colour of all pixels, black ones included, i.e. the light the grass puts there, as the blades are
    // placed alike but shaped differently (hard-edged strips, soft Gaussians).
    auto blockMean = [&](const std::vector<uint8_t>& bgra, uint32_t bx, uint32_t by) {
        glm::vec3 sum(0.0f);
        for (uint32_t y = by; y < by + 16; ++y) {
            for (uint32_t x = bx; x < bx + 16; ++x) {
                const uint8_t* p = &bgra[4 * (size_t(y) * extent.width + x)];
                sum += glm::vec3(p[2], p[1], p[0]) / 255.0f;
            }
        }
        return sum / 256.0f;
    };
    double difference = 0.0;
    glm::vec3 meanClassical(0.0f), meanGaussian(0.0f);
    int blocks = 0;
    for (uint32_t by = extent.height / 2; by + 16 <= extent.height * 4 / 5; by += 16) {
        for (uint32_t bx = 0; bx + 16 <= extent.width; bx += 16) {
            const glm::vec3 a = blockMean(classical, bx, by), b = blockMean(gaussian, bx, by);
            difference += std::abs(a.y - b.y);
            meanClassical += a;
            meanGaussian += b;
            ++blocks;
        }
    }
    difference /= blocks;
    meanClassical /= float(blocks);
    meanGaussian /= float(blocks);
    std::printf("mean of all pixels: classical (%.3f, %.3f, %.3f), Gaussian (%.3f, %.3f, %.3f); green %.3f apart per "
                "block\n",
                meanClassical.x, meanClassical.y, meanClassical.z, meanGaussian.x, meanGaussian.y, meanGaussian.z,
                difference);
    EXPECT_GT(meanClassical.y, 0.1f);
    EXPECT_LT(difference, 0.05);
    EXPECT_LT(glm::length(meanGaussian - meanClassical), 0.03f);
    EXPECT_GT(meanGaussian.y, meanGaussian.x);
    EXPECT_GT(meanGaussian.y, meanGaussian.z);
}

TEST_F(TreeGrassTest, bladesCoverTheGround) {
    const BonsaiScene scene = sceneAt(0.75f);
    const VkExtent2D extent{960, 540};
    GrassScene g(*vc, scene, extent);
    const Eye e = standingInTheMeadow(scene);
    setSeason(*g.settings, kSummer);
    const std::vector<uint8_t> summer = g.render(e);
    std::printf("grass: %u clumps\n", g.grass->readCounter()[0]);
    for (const auto& [name, ms] : g.graph->getProfilingResults()) {
        std::printf("  %s: %.2f ms\n", name.c_str(), ms);
    }
    setSeason(*g.settings, 0.0f);
    const std::vector<uint8_t> winter = g.render(e);
    writePpm("grass_summer", summer, extent);
    writePpm("grass_winter", winter, extent);

    const Coverage sky = coverageOf(summer, extent, 0, extent.height / 10);
    const Coverage near = coverageOf(summer, extent, extent.height * 4 / 5, extent.height);
    const Coverage middle = coverageOf(summer, extent, extent.height / 2, extent.height * 3 / 5);
    std::printf("coverage: %.3f of the sky, %.2f near, %.2f in the middle; mean near (%.2f, %.2f, %.2f)\n", sky.share,
                near.share, middle.share, near.mean.x, near.mean.y, near.mean.z);
    EXPECT_LT(sky.share, 0.01f);
    EXPECT_GT(near.share, 0.4f);
    EXPECT_GT(middle.share, 0.5f);
    EXPECT_GT(near.mean.y, near.mean.x);
    EXPECT_GT(near.mean.y, near.mean.z);
    const Coverage snowy = coverageOf(winter, extent, extent.height / 2, extent.height);
    const Coverage green = coverageOf(summer, extent, extent.height / 2, extent.height);
    std::printf("mean in winter (%.2f, %.2f, %.2f)\n", snowy.mean.x, snowy.mean.y, snowy.mean.z);
    EXPECT_GT(snowy.mean.x + snowy.mean.y + snowy.mean.z, 1.3f * (green.mean.x + green.mean.y + green.mean.z));
}

TEST(TreeGrass, cropsFollowTheSeason) {
    const BonsaiScene scene = sceneAt(0.75f);
    const TreeGrassParams g = makeTreeGrass(scene);
    // Cells of open land across the region, sorted into fields (by their parcel) and meadows.
    double grain[2] = {0.0, 0.0}, meadow[2] = {0.0, 0.0};
    int grainCount[2] = {0, 0}, meadowCount[2] = {0, 0}, ploughed = 0;
    int ploughedCells = 0;
    const float seasons[2] = {kSummer, 0.0f};
    for (int j = -200; j < 200; j += 3) {
        for (int i = -300; i < 300; i += 3) {
            const int cx = int(std::floor(0.004f * i / g.grid.x)), cz = int(std::floor(0.004f * j / g.grid.x));
            for (int k = 0; k < 2; ++k) {
                const glm::vec3 eye((cx + 0.5f) * g.grid.x, 0.0f, (cz + 0.5f) * g.grid.x);
                const TreeGrassClump c = tree::treeGrassCell(scene.terrain, g, cx, cz, eye, seasons[k]);
                const TreeLandCover cover = tree::treeLandCover(scene.terrain, eye.x, eye.z);
                const float crop = tree::treeFieldCropHeight(glm::vec2(eye.x, eye.z) * scene.terrain.cover.y, kSummer,
                                                             scene.terrain.seed.x);
                if (cover.vegetation.z > 0.95f && crop <= 0.0f) {
                    ++ploughedCells;
                    ploughed += c.kind.x > 0.0f;
                } else if (cover.vegetation.z > 0.95f && crop > 1.0f && c.kind.x > 0.0f) {
                    grain[k] += c.posHeight.w * g.grid.w;
                    ++grainCount[k];
                } else if (cover.vegetation.y > 0.95f && c.kind.x > 0.0f) {
                    meadow[k] += c.posHeight.w * g.grid.w;
                    ++meadowCount[k];
                }
            }
        }
    }
    ASSERT_GT(grainCount[0], 50);
    ASSERT_GT(ploughedCells, 10);
    ASSERT_GT(meadowCount[0], 50);
    const float grainSummer = grain[0] / grainCount[0], grainWinter = grain[1] / std::max(grainCount[1], 1);
    const float meadowSummer = meadow[0] / meadowCount[0], meadowWinter = meadow[1] / std::max(meadowCount[1], 1);
    std::printf("grain %.2f m in summer, %.2f m in winter; meadow %.2f m, %.2f m\n", grainSummer, grainWinter,
                meadowSummer, meadowWinter);
    // Early summer (S = 1.2): the grain is half grown.
    EXPECT_GT(grainSummer, 0.3f);
    EXPECT_LT(grainWinter, 0.1f);
    EXPECT_NEAR(meadowWinter, meadowSummer, 0.02f);
    EXPECT_EQ(ploughed, 0);
}

TEST_F(TreeGrassTest, theRenderersDrawGrass) {
    const BonsaiScene scene = sceneAt(0.75f);
    const VkExtent2D extent{480, 270};
    const Eye low = standingInTheMeadow(scene);
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
        ASSERT_NE(renderer.getGrass(), nullptr);
        ASSERT_EQ(renderer.getGrassGaussians() != nullptr, mode == BonsaiRenderMode::Gaussian);
        for (const glm::vec3 lift : {glm::vec3(0.0f), up}) {
            const glm::vec3 eye = low.eye + lift;
            setCamera(renderer.getCamera()->ubo, eye, low.target + lift, 50.0f, 16.0f / 9.0f, 1e-5f, 100.0f);
            renderer.getCamera()->update(0);
            renderer.setView(eye);
            graph.submitAndWait(vc->getGraphicsQueue(), 0);
            const uint32_t clumps = renderer.getGrass()->readCounter()[0];
            if (lift == glm::vec3(0.0f)) {
                EXPECT_GT(clumps, 5000u);
                if (renderer.getGrassGaussians()) {
                    EXPECT_GT(renderer.getGrassGaussians()->readTaken(), clumps);
                    EXPECT_LE(renderer.getGrassGaussians()->readTaken(), config.grassPool);
                }
                writePpm(mode == BonsaiRenderMode::Classical ? "grass_renderer_classical" : "grass_renderer_gaussian",
                         readImage(*vc, target->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent), extent);
            } else {
                EXPECT_EQ(clumps, 0u);
            }
        }
    }
    const BonsaiScene planting = sceneAt(0.21f);
    auto target = std::make_shared<OffscreenTarget>(*vc, extent, 1);
    BonsaiRenderConfig config;
    config.extent = extent;
    BonsaiRenderer renderer(*vc, planting, target, config);
    EXPECT_EQ(renderer.getGrass(), nullptr);
}
