// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - treesStandInThePot: 8, 13 and 20 trees stand on the soil inside the pot, apart from each other and from the
 *   lantern
 * - treesAreFourToNineLanternsTall: the generated trees are 4 to 9 times as tall as the lantern
 * - mossCoversPartOfTheSoil: moss grows on 30 to 80 % of the soil in patches a few centimetres wide, greener
 *   than the bare soil, and fading the fine detail keeps the mean colour but smooths it
 * - soilLiesInThePot: the soil rises from below the rim at the pot's edge towards the middle
 * - potWallsFaceOutwards: the mesh is a triangle list; outer wall normals point away from the middle, the
 *   bottom's down, the soil's up
 * - meshSurfelsCoverTheMesh: sampleMeshSurfels places about area / spacing^2 samples on the triangles, with
 *   their normal and spacing
 * - mossPatchesAreGreenSectors: a synthetic ring of green and brown ground gives patches only of its green
 *   sectors, centred at the origin with their ground at 0 and faded towards the cuts
 * - mossPatchesFromTheCapture: the lantern capture gives several mostly green patches
 * - scatteredMossLiesOnTheSoil: placed copies of patches lie inside the pot, away from the lantern, on the plane
 *   touching the soil, at their height above it
 * - capturedGaussiansTakeSnow: treeGaussianNormal is the thinnest axis turned up; treeCapturedColor covers what
 *   faces up with snow in winter only and scales by tint and shade
 * - lanternStandsOnTheSoil: the placed lantern capture's ground lies on the soil, its moss inside the pot
 * - classicalGeometryIsHiddenBehindSplats: a quad behind an opaque block of splats is hidden by it, a quad in
 *   front covers it, with the splats' depth output
 * - supersamplingSmoothsEdges: the classical render at 2x supersampling differs less between neighbouring pixels
 *   than without, at the same mean brightness
 * - nightIsDarkerAndTheLanternGlows: at midnight the scene is darker than at noon, and the moss next to the
 *   lantern is lit warmer than at noon, in both renderers
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

TEST(BonsaiScene, mossCoversPartOfTheSoil) {
    const glm::vec3 base(0.075f, 0.11f, 0.025f);
    int moss = 0, samples = 0;
    glm::vec3 meanFine(0.0f), meanSmooth(0.0f), mossColour(0.0f), soilColour(0.0f);
    for (float x = -1.0f; x < 1.0f; x += 0.01f) {
        for (float z = -0.6f; z < 0.6f; z += 0.01f) {
            const glm::vec3 p(x, 0.3f, z);
            const float coverage = tree::treeMossCoverage(p);
            moss += coverage > 0.5f ? 1 : 0;
            ++samples;
            const glm::vec3 a = tree::treeMossAlbedo(base, p, 1.0f);
            meanFine += a;
            meanSmooth += tree::treeMossAlbedo(base, p, 0.0f);
            mossColour += coverage > 0.95f ? a : glm::vec3(0.0f);
            soilColour += coverage < 0.05f ? a : glm::vec3(0.0f);
        }
    }
    const float fraction = float(moss) / float(samples);
    EXPECT_GT(fraction, 0.3f);
    EXPECT_LT(fraction, 0.8f);
    EXPECT_NEAR(meanFine.g / meanSmooth.g, 1.0f, 0.05f);
    // Moss is greener than bare soil.
    EXPECT_GT(mossColour.g / mossColour.r, 1.2f * soilColour.g / soilColour.r);

    // Along a densely sampled line the fine structure (cushions, grit) changes the colour between neighbouring
    // samples; without detail it is smooth.
    double fine = 0.0, smooth = 0.0;
    glm::vec3 previousFine(0.0f), previousSmooth(0.0f);
    for (int i = 0; i <= 4000; ++i) {
        const glm::vec3 p(-1.0f + 0.0005f * float(i), 0.3f, 0.1f);
        const glm::vec3 a = tree::treeMossAlbedo(base, p, 1.0f);
        const glm::vec3 b = tree::treeMossAlbedo(base, p, 0.0f);
        if (i > 0) {
            fine += glm::length(a - previousFine);
            smooth += glm::length(b - previousSmooth);
        }
        previousFine = a;
        previousSmooth = b;
    }
    EXPECT_LT(smooth, 0.7 * fine);
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
        if (v.color.w == TREE_MATERIAL_GLAZE && p.y == 0.0f && std::abs(n.y) > 0.5f) {
            EXPECT_LT(n.y, -0.99f); // the bottom
        } else if (v.color.w == TREE_MATERIAL_GLAZE && p.y < 0.5f * scene.config.pot.height) {
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

TEST(BonsaiScene, meshSurfelsCoverTheMesh) {
    std::vector<TreeSceneVertex> mesh;
    addQuad(mesh, 0.0f, 2.0f, 0.0f, 1.0f, 0.5f, glm::vec3(0.5f));
    const float spacing = 0.02f;
    const auto surfels = sampleMeshSurfels(mesh, spacing, 3);
    EXPECT_NEAR(double(surfels.size()), 2.0 / (spacing * spacing), 0.02 * 2.0 / (spacing * spacing));
    for (const TreeSceneVertex& v : surfels) {
        ASSERT_NEAR(v.pos.z, 0.5f, 1e-5f);
        ASSERT_GE(v.pos.x, 0.0f);
        ASSERT_LE(v.pos.x, 2.0f);
        ASSERT_GE(v.pos.y, 0.0f);
        ASSERT_LE(v.pos.y, 1.0f);
        ASSERT_EQ(v.pos.w, spacing);
        ASSERT_NEAR(v.normal.z, 1.0f, 1e-5f);
    }
}

namespace {

// A flat Gaussian of the given colour (degree 0 coefficients) at (x, y, z).
Gaussian3D groundGaussian(float x, float y, float z, glm::vec3 color) {
    Gaussian3D g{};
    g.position = {x, y, z};
    g.rotation = {0.0f, 0.0f, 0.0f, 1.0f};
    g.scale = {0.004f, 0.0005f, 0.004f};
    g.color = {color.x, color.y, color.z};
    g.alpha = 1.0f;
    return g;
}

} // namespace

TEST(BonsaiScene, mossPatchesAreGreenSectors) {
    // Ground at y = 0.01 around the origin: green for x < 0, brown for x > 0.
    std::vector<Gaussian3D> capture;
    for (float x = -0.25f; x <= 0.25f; x += 0.004f) {
        for (float z = -0.25f; z <= 0.25f; z += 0.004f) {
            const glm::vec3 color = x < 0.0f ? glm::vec3(-0.5f, 0.5f, -0.8f) : glm::vec3(0.4f, 0.0f, -0.6f);
            capture.push_back(groundGaussian(x, 0.01f, z, color));
        }
    }
    capture.push_back(groundGaussian(0.18f, 0.2f, 0.0f, glm::vec3(1.0f))); // too high: the lantern
    const std::vector<MossPatch> patches = extractMossPatches(capture, 12);
    // Of the 12 sectors of 30 degrees, those within 90 to 270 degrees are green.
    ASSERT_GE(patches.size(), 4u);
    ASSERT_LE(patches.size(), 6u);
    for (const MossPatch& patch : patches) {
        glm::vec3 mean(0.0f);
        float minAlpha = 1.0f, maxAlpha = 0.0f;
        for (const Gaussian3D& g : patch) {
            mean += glm::vec3(g.position[0], g.position[1], g.position[2]);
            EXPECT_GT(g.color[1], g.color[0]);
            minAlpha = std::min(minAlpha, g.alpha);
            maxAlpha = std::max(maxAlpha, g.alpha);
        }
        mean /= float(patch.size());
        EXPECT_NEAR(mean.x, 0.0f, 1e-3f);
        EXPECT_NEAR(mean.z, 0.0f, 1e-3f);
        EXPECT_NEAR(mean.y, 0.0f, 1e-5f);
        EXPECT_EQ(maxAlpha, 1.0f);
        EXPECT_LT(minAlpha, 0.3f); // faded at the cuts
    }
}

TEST(BonsaiScene, mossPatchesFromTheCapture) {
    const std::vector<MossPatch> patches = extractMossPatches(loadGaussiansSpz("data/lantern.spz", true));
    EXPECT_GE(patches.size(), 3u);
    for (const MossPatch& patch : patches) {
        EXPECT_GT(patch.size(), 500u);
        size_t green = 0;
        for (const Gaussian3D& g : patch) {
            green += g.color[1] > g.color[0] && g.color[1] > g.color[2] ? 1 : 0;
        }
        EXPECT_GE(float(green), 0.6f * float(patch.size()));
    }
}

TEST(BonsaiScene, scatteredMossLiesOnTheSoil) {
    const BonsaiScene scene = makeBonsaiScene();
    MossPatch patch;
    for (float x = -0.04f; x <= 0.04f; x += 0.01f) {
        patch.push_back(groundGaussian(x, 0.003f, 0.0f, glm::vec3(0.0f, 0.5f, 0.0f)));
    }
    const std::vector<TreeLodInstance> copies = placeMossPatches(scene, 1, 50, 3, 1.0f);
    ASSERT_EQ(copies.size(), 50u);
    const std::vector<Gaussian3D> moss = scatterMossPatches({patch}, copies);
    EXPECT_EQ(moss.size(), 50u * patch.size());
    const glm::vec2 lantern(scene.lanternTranslation.x, scene.lanternTranslation.z);
    for (const Gaussian3D& g : moss) {
        const glm::vec3 p(g.position[0], g.position[1], g.position[2]);
        EXPECT_LT(scene.potMeasure(p.x, p.z, 0.95f), 1.0f);
        EXPECT_GT(glm::length(glm::vec2(p.x, p.z) - lantern), 0.25f);
        // On the plane that touches the soil at the copy's centre: as high above the soil as above the patch's
        // ground, within the soil's curvature over the patch.
        EXPECT_NEAR(p.y - scene.soilHeight(p.x, p.z), 0.003f + 0.002f, 1e-3f);
    }
}

TEST(BonsaiScene, capturedGaussiansTakeSnow) {
    // Thinnest axis y, turned upside down: the normal points up again.
    const glm::vec3 n = tree::treeGaussianNormal(glm::vec4(1.0f, 0.0f, 0.0f, 0.0f), glm::vec3(0.1f, 0.01f, 0.1f));
    EXPECT_NEAR(n.y, 1.0f, 1e-6f);
    EXPECT_NEAR(tree::treeGaussianNormal(glm::vec4(0, 0, 0, 1), glm::vec3(0.01f, 0.1f, 0.1f)).x, 1.0f, 1e-6f);

    TreeRenderSettings s = defaultTreeRenderSettings({64, 64});
    const glm::vec3 green(0.2f, 0.6f, 0.2f);
    const glm::vec3 up(0.0f, 1.0f, 0.0f), p(0.1f, 0.3f, 0.2f);
    s.season.y = 0.0f;
    EXPECT_NEAR(glm::length(tree::treeCapturedColor(green, up, p, 0.5f, s) - 0.5f * green), 0.0f, 1e-6f);
    s.season.y = 1.0f;
    const glm::vec3 snowy = tree::treeCapturedColor(green, up, p, 1.0f, s);
    EXPECT_GT(snowy.r, 0.8f);
    const glm::vec3 side = tree::treeCapturedColor(green, glm::vec3(1.0f, 0.0f, 0.0f), p, 1.0f, s);
    EXPECT_NEAR(glm::length(side - green), 0.0f, 1e-6f);
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
        SceneMeshRenderer meshes(*vc, mesh, settings, extent,
                                 TreeRenderInputs{BufferRef{getGaussianSplattingDepth(splatting), -1}, {}});
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

TEST_F(BonsaiRenderingTest, supersamplingSmoothsEdges) {
    const BonsaiScene scene = makeBonsaiScene();
    const VkExtent2D extent{480, 270};
    std::vector<double> variation, brightness;
    for (uint32_t supersample : {1u, 2u}) {
        auto target = std::make_shared<OffscreenTarget>(*vc, extent, 1);
        BonsaiRenderConfig config;
        config.extent = extent;
        config.supersample = supersample;
        BonsaiRenderer bonsai(*vc, scene, target, config);
        ComputeGraph graph(*vc, 1);
        graph.compileFrom(bonsai.getOutput());
        bonsai.initialize(1);
        setCamera(bonsai.getCamera()->ubo, scene.focus + glm::vec3(1.4f, 1.3f, 3.8f), scene.focus, 40.0f,
                  float(extent.width) / float(extent.height));
        bonsai.getCamera()->update(0);
        graph.submitAndWait(vc->getGraphicsQueue(), 0);
        const auto image = readImage(*vc, target->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
        writePpm("bonsai_supersample_" + std::to_string(supersample), image, extent);
        double v = 0.0, b = 0.0;
        for (uint32_t y = 0; y < extent.height; ++y) {
            for (uint32_t x = 1; x < extent.width; ++x) {
                const size_t i = (size_t(y) * extent.width + x) * 4;
                for (int c = 0; c < 3; ++c) {
                    v += std::abs(int(image[i + c]) - int(image[i - 4 + c]));
                    b += image[i + c];
                }
            }
        }
        variation.push_back(v);
        brightness.push_back(b);
    }
    std::cout << "neighbour variation " << variation[0] << " -> " << variation[1] << std::endl;
    EXPECT_LT(variation[1], 0.85 * variation[0]);
    EXPECT_NEAR(brightness[1] / brightness[0], 1.0, 0.1);
}

TEST_F(BonsaiRenderingTest, nightIsDarkerAndTheLanternGlows) {
    const BonsaiScene scene = makeBonsaiScene();
    const VkExtent2D extent{480, 270};
    for (BonsaiRenderMode mode : {BonsaiRenderMode::Classical, BonsaiRenderMode::Gaussian}) {
        const char* name = mode == BonsaiRenderMode::Classical ? "classical" : "gaussian";
        SCOPED_TRACE(name);
        auto target = std::make_shared<OffscreenTarget>(*vc, extent, 1);
        BonsaiRenderConfig config;
        config.extent = extent;
        config.mode = mode;
        config.meshGaussianSpacing = 0.02f;
        BonsaiRenderer bonsai(*vc, scene, target, config);
        ComputeGraph graph(*vc, 1);
        graph.compileFrom(bonsai.getOutput());
        bonsai.initialize(1);
        setCamera(bonsai.getCamera()->ubo, scene.focus + glm::vec3(1.4f, 1.3f, 3.8f), scene.focus, 40.0f,
                  float(extent.width) / float(extent.height));
        bonsai.getCamera()->update(0);
        // The moss just in front of the lantern, in pixels.
        const glm::vec3 moss = scene.lanternTranslation + glm::vec3(0.0f, 0.0f, 0.2f);
        const glm::vec4 clip = bonsai.getCamera()->ubo.proj * bonsai.getCamera()->ubo.view * glm::vec4(moss, 1.0f);
        const int px = int((clip.x / clip.w * 0.5f + 0.5f) * extent.width);
        const int py = int((clip.y / clip.w * 0.5f + 0.5f) * extent.height);
        auto render = [&](float timeOfDay) {
            bonsai.setTime(1.0f, timeOfDay, 2.0f, glm::vec4(1.0f, 0.0f, 0.0f, 0.0f));
            graph.submitAndWait(vc->getGraphicsQueue(), 0);
            const auto image = readImage(*vc, target->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
            writePpm(std::string("bonsai_") + name + (timeOfDay < 0.25f ? "_night" : "_noon"), image, extent);
            glm::dvec3 total(0.0), near(0.0);
            for (uint32_t y = 0; y < extent.height; ++y) {
                for (uint32_t x = 0; x < extent.width; ++x) {
                    const size_t i = (size_t(y) * extent.width + x) * 4;
                    const glm::dvec3 rgb(image[i + 2], image[i + 1], image[i]);
                    total += rgb;
                    if (std::abs(int(x) - px) <= 3 && std::abs(int(y) - py) <= 3) {
                        near += rgb;
                    }
                }
            }
            return std::make_pair(total, near);
        };
        const auto noon = render(0.5f);
        const auto night = render(0.0f);
        const double noonBrightness = noon.first.r + noon.first.g + noon.first.b;
        const double nightBrightness = night.first.r + night.first.g + night.first.b;
        EXPECT_LT(nightBrightness, 0.6 * noonBrightness);
        // Warmer: more red than blue next to the lantern at night, compared with noon.
        EXPECT_GT(night.second.r / std::max(night.second.b, 1.0), noon.second.r / std::max(noon.second.b, 1.0));
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
        // Trees and pot fill a good part of the frame; the background stays black.
        const size_t coverage = coveredCount(image, extent);
        EXPECT_GT(coverage, extent.width * extent.height / 5);
        EXPECT_LT(coverage, extent.width * extent.height * 4 / 5);
        EXPECT_EQ(bonsai.getGenerator().readCounters()[kTreeCounterOverflow], 0u);
    }
}
