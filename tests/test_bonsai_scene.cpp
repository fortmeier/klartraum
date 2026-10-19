// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - treesStandInThePot: 8, 13 and 20 trees stand on the soil inside the pot, apart from each other and from the
 *   lantern
 * - treesAreFourToNineLanternsTall: the generated trees are 4 to 9 times as tall as the lantern
 * - soilLiesInThePot: the soil rises from below the rim at the pot's edge towards the middle
 * - potWallsFaceOutwards: the mesh is a triangle list; outer wall normals point away from the middle, the soil's up
 * - meshGaussiansCoverTheMesh: sampleMeshGaussians places about area / spacing^2 Gaussians on the triangles
 * - lanternStandsOnTheSoil: the placed lantern capture's ground lies on the soil, its moss inside the pot
 * - classicalGeometryIsHiddenBehindSplats: a quad behind an opaque block of splats is hidden by it, a quad in
 *   front covers it, with the splats' depth output
 * - bonsaiRendersBothWays: the scene renders classically and as Gaussians, written to build/TestingOutput/tree/
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/tree/bonsai_renderer.hpp"

#include "tree_test_helpers.hpp"

using namespace klartraum;
using namespace klartraum::tree_test;

namespace {

constexpr float kLanternHeight = 0.28f;

TreeSceneVertex vertex(glm::vec3 p, glm::vec3 n, glm::vec3 color) {
    TreeSceneVertex v{};
    v.pos = glm::vec4(p, 1.0f);
    v.normal = glm::vec4(n, 0.0f);
    v.color = glm::vec4(color, TREE_MATERIAL_PLAIN);
    return v;
}

// An axis-aligned quad in the plane z, facing +z.
void addQuad(std::vector<TreeSceneVertex>& mesh, float x0, float x1, float y0, float y1, float z, glm::vec3 color) {
    const glm::vec3 n(0.0f, 0.0f, 1.0f);
    const TreeSceneVertex a = vertex({x0, y0, z}, n, color), b = vertex({x1, y0, z}, n, color),
                          c = vertex({x1, y1, z}, n, color), d = vertex({x0, y1, z}, n, color);
    mesh.insert(mesh.end(), {a, b, c, a, c, d});
}

} // namespace

TEST(BonsaiScene, treesStandInThePot) {
    for (uint32_t count : {8u, 13u, 20u}) {
        BonsaiSceneConfig config;
        config.treeCount = count;
        const BonsaiScene scene = makeBonsaiScene(config);
        ASSERT_EQ(scene.trees.size(), count);
        const glm::vec2 lantern(config.lanternPosition.x, config.lanternPosition.z);
        for (size_t i = 0; i < scene.trees.size(); ++i) {
            const glm::vec4 p = scene.trees[i].posScale;
            EXPECT_LT(scene.potMeasure(p.x, p.z, 0.9f), 1.0f) << i;
            EXPECT_NEAR(p.y, scene.soilHeight(p.x, p.z) - 0.01f, 1e-5f) << i;
            EXPECT_GE(glm::length(glm::vec2(p.x, p.z) - lantern), 0.38f) << i;
            for (size_t j = 0; j < i; ++j) {
                const glm::vec4 q = scene.trees[j].posScale;
                EXPECT_GE(glm::length(glm::vec2(p.x - q.x, p.z - q.z)), 0.3f - 1e-5f) << i << " " << j;
            }
        }
    }
}

TEST(BonsaiScene, treesAreFourToNineLanternsTall) {
    const BonsaiScene scene = makeBonsaiScene();
    const TreeRecords records = generateTreesReference(scene.types, scene.trees, defaultTreeFrame());
    std::vector<float> top(scene.trees.size(), 0.0f);
    for (const SegmentRecord& d : records.segments) {
        top[d.ids.z] = std::max(top[d.ids.z], d.toPosRadius.y);
    }
    for (size_t i = 0; i < top.size(); ++i) {
        const float height = top[i] - scene.trees[i].posScale.y;
        EXPECT_GE(height / kLanternHeight, 4.0f) << i;
        EXPECT_LE(height / kLanternHeight, 9.0f) << i;
    }
}

TEST(BonsaiScene, soilLiesInThePot) {
    const BonsaiScene scene = makeBonsaiScene();
    const BonsaiPot& pot = scene.config.pot;
    // At the inner rim the soil is below the rim; in the middle it is mounded.
    EXPECT_LT(scene.soilHeight(pot.halfLength - pot.wall, 0.0f), pot.height);
    EXPECT_NEAR(scene.soilHeight(0.0f, 0.0f), pot.soilHeight + pot.soilMound, 0.01f);
    EXPECT_GT(scene.soilHeight(0.0f, 0.0f), scene.soilHeight(1.0f, 0.4f));
}

TEST(BonsaiScene, potWallsFaceOutwards) {
    const BonsaiScene scene = makeBonsaiScene();
    ASSERT_EQ(scene.mesh.size() % 3, 0u);
    size_t outerWall = 0, soil = 0;
    for (const TreeSceneVertex& v : scene.mesh) {
        const glm::vec3 p(v.pos), n(v.normal);
        EXPECT_NEAR(glm::length(n), 1.0f, 1e-4f);
        if (v.color.w == TREE_MATERIAL_GLAZE && p.y < 0.5f * scene.config.pot.height) {
            EXPECT_GT(glm::dot(n, glm::vec3(p.x, 0.0f, p.z)), 0.0f);
            ++outerWall;
        }
        if (v.color.w == TREE_MATERIAL_MOSS) {
            EXPECT_GT(n.y, 0.8f);
            ++soil;
        }
    }
    EXPECT_GT(outerWall, 100u);
    EXPECT_GT(soil, 100u);
}

TEST(BonsaiScene, meshGaussiansCoverTheMesh) {
    std::vector<TreeSceneVertex> mesh;
    addQuad(mesh, 0.0f, 2.0f, 0.0f, 1.0f, 0.5f, glm::vec3(0.5f));
    const float spacing = 0.02f;
    const auto gaussians = sampleMeshGaussians(mesh, spacing, defaultTreeRenderSettings({64, 64}), 3);
    EXPECT_NEAR(double(gaussians.size()), 2.0 / (spacing * spacing), 0.02 * 2.0 / (spacing * spacing));
    for (const Gaussian3D& g : gaussians) {
        ASSERT_NEAR(g.position[2], 0.5f, 1e-5f);
        ASSERT_GE(g.position[0], 0.0f);
        ASSERT_LE(g.position[0], 2.0f);
        ASSERT_GE(g.position[1], 0.0f);
        ASSERT_LE(g.position[1], 1.0f);
        // The thin axis along the normal: the rotated z axis is +-z.
        const glm::quat q(g.rotation[3], g.rotation[0], g.rotation[1], g.rotation[2]);
        ASSERT_NEAR(std::abs((q * glm::vec3(0.0f, 0.0f, 1.0f)).z), 1.0f, 1e-4f);
    }
}

TEST(BonsaiScene, lanternStandsOnTheSoil) {
    const BonsaiScene scene = makeBonsaiScene();
    const std::vector<Gaussian3D> lantern = BonsaiRenderer::loadLantern(scene, "data/lantern.spz");
    ASSERT_GT(lantern.size(), 10000u);
    std::vector<float> heights;
    for (const Gaussian3D& g : lantern) {
        const glm::vec3 p(g.position[0], g.position[1], g.position[2]);
        EXPECT_LT(scene.potMeasure(p.x, p.z, 0.95f), 1.0f);
        const float r =
            glm::length(glm::vec2(p.x, p.z) - glm::vec2(scene.lanternTranslation.x, scene.lanternTranslation.z));
        if (r > 0.12f) {
            heights.push_back(p.y - scene.soilHeight(p.x, p.z));
        }
    }
    // The moss around the lantern lies on the soil, within the capture's own unevenness.
    std::nth_element(heights.begin(), heights.begin() + heights.size() / 2, heights.end());
    EXPECT_NEAR(heights[heights.size() / 2], 0.0f, 0.02f);
}

class BonsaiRenderingTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
    }
    void TearDown() override { frontend.reset(); }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
};

TEST_F(BonsaiRenderingTest, classicalGeometryIsHiddenBehindSplats) {
    const VkExtent2D extent{128, 128};
    // An opaque red block of Gaussians around the origin, 0.4 wide.
    std::vector<Gaussian3D> block;
    const float c0 = 0.28209479f;
    for (int i = 0; i < 8000; ++i) {
        Gaussian3D g{};
        g.position = {0.02f * float(i % 20) - 0.19f, 0.02f * float(i / 20 % 20) - 0.19f,
                      0.02f * float(i / 400) - 0.19f};
        g.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
        g.scale = {0.015f, 0.015f, 0.015f};
        g.color = {0.5f / c0, -0.5f / c0, -0.5f / c0};
        g.alpha = 0.99f;
        block.push_back(g);
    }
    auto model = std::make_shared<GaussianDataStandard>(*vc, std::move(block));
    // A blue quad behind the block on the left, a green quad in front on the right.
    std::vector<TreeSceneVertex> mesh;
    addQuad(mesh, -0.6f, 0.0f, -0.6f, 0.6f, -0.5f, glm::vec3(0.0f, 0.0f, 1.0f));
    addQuad(mesh, 0.1f, 0.6f, -0.6f, 0.6f, 0.5f, glm::vec3(0.0f, 1.0f, 0.0f));

    for (GsplatBackend backend : {GsplatBackend::Raster, GsplatBackend::Compute}) {
        SCOPED_TRACE(backend == GsplatBackend::Raster ? "raster" : "compute");
        auto target = std::make_shared<OffscreenTarget>(*vc, extent, 1);
        auto camera = std::make_shared<CameraUboType>();
        GsplatConfig config;
        config.outputDepth = true;
        auto splatting = createGaussianSplatting(*vc, backend, target, camera, model, config);
        auto toGeneral = std::make_shared<ImageToGeneral>();
        toGeneral->setInput(target, 0);
        toGeneral->addDependency(splatting);
        TreeRenderSettings values = defaultTreeRenderSettings(extent);
        values.occlusion.x = 1.0f;
        // Unlit: full ambient from every side, no sun.
        values.sun.w = 0.0f;
        values.sky = values.ground = glm::vec4(1.0f);
        auto settings = std::make_shared<HostValues<TreeRenderSettings>>(*vc, std::vector<TreeRenderSettings>{values});
        SceneMeshRenderer meshes(*vc, mesh, settings, extent, {getGaussianSplattingDepth(splatting), -1});
        auto pass = std::make_shared<RenderPass>(vc->getSwapChainImageFormat(), extent);
        pass->setInput(toGeneral, 0);
        pass->setInput(camera, 1);
        pass->setLoadExisting(true);
        pass->setDepthAttachment(true);
        pass->addComputeDependency(settings);
        pass->addDrawComponent(meshes.getDrawComponent());
        ComputeGraph graph(*vc, 1);
        graph.compileFrom(pass);
        setCamera(camera->ubo, glm::vec3(0.0f, 0.0f, 2.0f), glm::vec3(0.0f), 40.0f, 1.0f);
        camera->update(0);
        graph.submitAndWait(vc->getGraphicsQueue(), 0);
        const auto image = readImage(*vc, target->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
        writePpm(std::string("composite_") + (backend == GsplatBackend::Raster ? "raster" : "compute"), image, extent);

        auto pixel = [&](float x, float y) {
            const glm::vec4 clip = camera->ubo.proj * camera->ubo.view * glm::vec4(x, y, 0.0f, 1.0f);
            const uint32_t px = uint32_t((clip.x / clip.w * 0.5f + 0.5f) * extent.width);
            const uint32_t py = uint32_t((clip.y / clip.w * 0.5f + 0.5f) * extent.height);
            const size_t i = (size_t(py) * extent.width + px) * 4;
            return glm::ivec3(image[i + 2], image[i + 1], image[i]); // rgb
        };
        // Left: the block in front of the blue quad.
        const glm::ivec3 left = pixel(-0.1f, 0.0f);
        EXPECT_GT(left.r, 200);
        EXPECT_LT(left.b, 60);
        // Right: the green quad in front of the block.
        const glm::ivec3 right = pixel(0.15f, 0.0f);
        EXPECT_GT(right.g, 200);
        EXPECT_LT(right.r, 60);
        // Outside the block: the blue quad.
        const glm::ivec3 behind = pixel(-0.4f, 0.0f);
        EXPECT_GT(behind.b, 200);
    }
}

TEST_F(BonsaiRenderingTest, bonsaiRendersBothWays) {
    const BonsaiScene scene = makeBonsaiScene();
    const VkExtent2D extent{480, 270};
    for (BonsaiRenderMode mode : {BonsaiRenderMode::Classical, BonsaiRenderMode::Gaussian}) {
        const char* name = mode == BonsaiRenderMode::Classical ? "classical" : "gaussian";
        SCOPED_TRACE(name);
        auto target = std::make_shared<OffscreenTarget>(*vc, extent, 1);
        BonsaiRenderConfig config;
        config.extent = extent;
        config.mode = mode;
        config.meshGaussianSpacing = 0.03f;
        BonsaiRenderer bonsai(*vc, scene, target, config);
        ComputeGraph graph(*vc, 1);
        graph.compileFrom(bonsai.getOutput());
        bonsai.initialize(1);
        setCamera(bonsai.getCamera()->ubo, scene.focus + glm::vec3(1.4f, 1.3f, 3.8f), scene.focus, 40.0f,
                  float(extent.width) / float(extent.height));
        bonsai.getCamera()->update(0);
        graph.submitAndWait(vc->getGraphicsQueue(), 0);
        const auto image = readImage(*vc, target->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
        writePpm(std::string("bonsai_") + name, image, extent);
        // Trees, pot and table fill most of the frame; the sky above stays black.
        EXPECT_GT(coveredCount(image, extent), extent.width * extent.height / 2);
        EXPECT_EQ(bonsai.getGenerator().readCounters()[kTreeCounterOverflow], 0u);
    }
}
