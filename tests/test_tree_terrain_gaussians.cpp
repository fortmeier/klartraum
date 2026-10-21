// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - gaussiansLieOnTheTerrain: every visible terrain Gaussian lies on the surface the classical renderer draws
 *   within a third of its size, inside the soil's outline, and the pool's unused slots are transparent
 * - poolDoesNotOverflow: the default pool holds the nodes of views from outside the pot and at ground level
 * - gaussianTerrainMatchesTheClassical: rendered as Gaussians, the terrain covers nearly the same pixels as the
 *   classical terrain (IoU above 0.8, their soft edges count most in small views) with nearly the same mean colour;
 *   images are written to
 *   build/TestingOutput/tree/
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>

#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/tree/bonsai_scene.hpp"
#include "klartraum/tree/tree_gaussians.hpp"
#include "klartraum/tree/tree_terrain_gaussians.hpp"
#include "klartraum/tree/tree_terrain_renderer.hpp"

#include "tree_test_helpers.hpp"

using namespace klartraum;
using namespace klartraum::tree_test;

namespace {

constexpr uint32_t kPool = 65536;

BonsaiScene sceneAt(float s) {
    BonsaiSceneConfig config;
    config.scale.s = s;
    config.treeCount = s > 0.3f ? 12u : 0u;
    return makeBonsaiScene(config);
}

struct Eye {
    glm::vec3 eye;
    glm::vec3 target;
};

std::vector<Eye> views(const BonsaiScene& scene) {
    const glm::vec3 focus(0.0f, 0.8f, 0.0f);
    const float floor = scene.terrain.soil.x, relief = scene.terrain.relief.x;
    return {
        {focus +
             4.2f * glm::vec3(std::cos(0.32f) * std::sin(0.35f), std::sin(0.32f), std::cos(0.32f) * std::cos(0.35f)),
         focus},
        {{0.6f, 1.4f, 1.9f}, {0.0f, 0.5f, 0.0f}},
        {{1.25f, floor + 1.1f * relief, 0.05f}, {-1.3f, floor + 0.3f * relief, -0.1f}},
    };
}

// The height of node `k`'s surface at (x, z): its two triangles per quad share the diagonal from (i, j) to
// (i + 1, j + 1).
float surfaceHeight(const std::vector<TreeTerrainNode>& nodes, const std::vector<glm::vec4>& vertices, size_t k,
                    float x, float z) {
    const int quads = 32, row = quads + 1;
    const TreeTerrainNode& n = nodes[k];
    const float u = std::clamp((x - n.x) / n.z * quads, 0.0f, float(quads));
    const float v = std::clamp((z - n.y) / n.z * quads, 0.0f, float(quads));
    const int i = std::min(int(u), quads - 1), j = std::min(int(v), quads - 1);
    const float fu = u - i, fv = v - j;
    auto h = [&](int a, int b) { return vertices[k * row * row + b * row + a].w; };
    if (fu >= fv) {
        return h(i, j) + fu * (h(i + 1, j) - h(i, j)) + fv * (h(i + 1, j + 1) - h(i + 1, j));
    }
    return h(i, j) + fv * (h(i, j + 1) - h(i, j)) + fu * (h(i + 1, j + 1) - h(i, j + 1));
}

// The terrain both ways: classically into one target, as Gaussians into another.
struct BothTerrains {
    BothTerrains(VulkanContext& vc, const BonsaiScene& scene, VkExtent2D extent)
        : vc(vc),
          extent(extent) {
        camera = std::make_shared<CameraUboType>();
        settings = std::make_shared<HostValues<TreeRenderSettings>>(
            vc, std::vector<TreeRenderSettings>{defaultTreeRenderSettings(extent)});
        terrain = std::make_unique<TreeTerrain>(vc, scene.terrain, camera);

        renderer = std::make_unique<TreeTerrainRenderer>(vc, *terrain, settings, extent);
        classicalTarget = std::make_shared<OffscreenTarget>(vc, extent, 1);
        pass = std::make_shared<RenderPass>(vc.getSwapChainImageFormat(), extent);
        pass->setInput(classicalTarget, 0);
        pass->setInput(camera, 1);
        pass->setDepthAttachment(true);
        pass->addComputeDependency(terrain->getElement());
        pass->addComputeDependency(settings);
        pass->addDrawComponent(renderer->getDrawComponent());
        classicalGraph = std::make_unique<ComputeGraph>(vc, 1);
        classicalGraph->compileFrom(pass);

        // A generator with a trunk only, far away; its Gaussians are not part of the buffers.
        TreeTypeParams p = tree_presets::quakingAspen();
        p.ratios.x = 1.0f;
        p.leafCount.x = 0.0f;
        const std::vector<TreeTypeParams> types = {p};
        const std::vector<TreeInstance> trees = {makeInstance(0, 1, glm::vec3(10.0f, 0.0f, 0.0f), 0.01f)};
        generator = std::make_shared<TreeGenerator>(vc, types, trees,
                                                    TreeCapacity::fromReference(types, trees, defaultTreeFrame()));
        TreeGaussianSources sources;
        sources.trees = false;
        sources.extraSlots = kPool;
        gaussians = std::make_unique<TreeGaussians>(vc, *generator, camera, settings, std::move(sources));
        terrainGaussians = std::make_unique<TreeTerrainGaussians>(vc, *terrain, gaussians->getGaussians(),
                                                                  gaussians->getExtraOffset(), kPool, settings, camera);
        gaussianTarget = std::make_shared<OffscreenTarget>(vc, extent, 1);
        splatting = createGaussianSplatting(vc, GsplatBackend::Raster, gaussianTarget, camera,
                                            terrainGaussians->getGaussians());
        gaussianGraph = std::make_unique<ComputeGraph>(vc, 1);
        gaussianGraph->compileFrom(splatting);
        gaussians->initializeBuffers(1);
    }

    void setView(const Eye& e) {
        setCamera(camera->ubo, e.eye, e.target, 40.0f, float(extent.width) / float(extent.height), 1e-5f, 100.0f);
        camera->update(0);
    }

    std::vector<uint8_t> renderClassical(const Eye& e) {
        setView(e);
        classicalGraph->submitAndWait(vc.getGraphicsQueue(), 0);
        return readImage(vc, classicalTarget->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
    }

    std::vector<uint8_t> renderGaussians(const Eye& e) {
        setView(e);
        gaussianGraph->submitAndWait(vc.getGraphicsQueue(), 0);
        return readImage(vc, gaussianTarget->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
    }

    template <typename T>
    std::vector<T> read(const BufferRef& ref) {
        auto buffer = std::dynamic_pointer_cast<BufferElement<VulkanBuffer<T>>>(ref.buffer());
        std::vector<T> values(buffer->getBuffer(0).getSize());
        buffer->getBuffer(0).memcopyTo(values);
        return values;
    }

    VulkanContext& vc;
    VkExtent2D extent;
    std::shared_ptr<CameraUboType> camera;
    std::shared_ptr<HostValues<TreeRenderSettings>> settings;
    std::unique_ptr<TreeTerrain> terrain;
    std::unique_ptr<TreeTerrainRenderer> renderer;
    std::shared_ptr<OffscreenTarget> classicalTarget;
    std::shared_ptr<RenderPass> pass;
    std::unique_ptr<ComputeGraph> classicalGraph;
    std::shared_ptr<TreeGenerator> generator;
    std::unique_ptr<TreeGaussians> gaussians;
    std::unique_ptr<TreeTerrainGaussians> terrainGaussians;
    std::shared_ptr<OffscreenTarget> gaussianTarget;
    std::shared_ptr<ComputeGraphElement> splatting;
    std::unique_ptr<ComputeGraph> gaussianGraph;
};

} // namespace

class TreeTerrainGaussiansTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
    }
    void TearDown() override { frontend.reset(); }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
};

TEST_F(TreeTerrainGaussiansTest, gaussiansLieOnTheTerrain) {
    const BonsaiScene scene = sceneAt(1.0f);
    BothTerrains both(*vc, scene, {320, 180});
    const Eye e = views(scene)[1];
    both.renderGaussians(e);
    const std::vector<TreeTerrainNode> nodes = both.terrain->readNodes();
    const std::vector<glm::vec4> vertices = both.terrain->readVertices();
    ASSERT_FALSE(nodes.empty());
    const std::vector<glm::vec3> pos = both.read<glm::vec3>(both.terrainGaussians->getGaussians().pos);
    const std::vector<glm::vec3> scale = both.read<glm::vec3>(both.terrainGaussians->getGaussians().scale);
    const std::vector<glm::vec4> colAlpha = both.read<glm::vec4>(both.terrainGaussians->getGaussians().colAlpha);
    const uint32_t offset = both.gaussians->getExtraOffset();
    const uint32_t perNode = 16u * 16u;
    int visible = 0;
    for (uint32_t k = 0; k < kPool; ++k) {
        const uint32_t i = offset + k;
        if (k >= nodes.size() * perNode) {
            EXPECT_EQ(colAlpha[i].w, 0.0f) << k;
            continue;
        }
        if (colAlpha[i].w == 0.0f) {
            continue;
        }
        ++visible;
        const glm::vec3 p = pos[i];
        EXPECT_LT(tree::treeTerrainRadius(scene.terrain, p.x, p.z), 1.0f);
        // The jitter in the tangent plane of the vertex normal moves it off the triangles across ridges: measured
        // up to 0.28 of its size.
        const float h = surfaceHeight(nodes, vertices, k / perNode, p.x, p.z);
        EXPECT_LT(std::abs(p.y - h), 0.34f * std::max(scale[i].x, scale[i].y) + 1e-5f) << k;
    }
    EXPECT_GT(visible, int(nodes.size() * perNode / 2));
}

TEST_F(TreeTerrainGaussiansTest, poolDoesNotOverflow) {
    const BonsaiScene scene = sceneAt(1.0f);
    BothTerrains both(*vc, scene, {320, 180});
    std::vector<Eye> eyes = views(scene);
    // Standing on the ground in the valley, looking along it.
    const glm::vec2 at(0.35f, 0.15f);
    const glm::vec3 ground(at.x, scene.soilHeight(at.x, at.y) + 2.0f / scene.config.scale.metersPerUnit(), at.y);
    eyes.push_back({ground, glm::vec3(-0.9f, ground.y, -0.3f)});
    for (const Eye& e : eyes) {
        both.renderGaussians(e);
        EXPECT_EQ(both.terrainGaussians->readOverflow(), 0u);
        EXPECT_LE(both.terrain->readNodes().size() * 256u, kPool);
    }
}

TEST_F(TreeTerrainGaussiansTest, gaussianTerrainMatchesTheClassical) {
    const BonsaiScene scene = sceneAt(1.0f);
    const VkExtent2D extent{480, 270};
    BothTerrains both(*vc, scene, extent);
    int index = 0;
    for (const Eye& e : views(scene)) {
        const std::vector<uint8_t> classical = both.renderClassical(e);
        const std::vector<uint8_t> splatted = both.renderGaussians(e);
        writePpm("terrain_classical_" + std::to_string(index), classical, extent);
        writePpm("terrain_gaussian_" + std::to_string(index), splatted, extent);
        size_t both_ = 0, either = 0;
        glm::dvec3 sumA(0.0), sumB(0.0);
        for (uint32_t y = 0; y < extent.height; ++y) {
            for (uint32_t x = 0; x < extent.width; ++x) {
                const bool a = covered(classical, extent, x, y), b = covered(splatted, extent, x, y);
                either += (a || b) ? 1 : 0;
                if (a && b) {
                    ++both_;
                    const size_t i = (size_t(y) * extent.width + x) * 4;
                    sumA += glm::dvec3(classical[i + 2], classical[i + 1], classical[i]) / 255.0;
                    sumB += glm::dvec3(splatted[i + 2], splatted[i + 1], splatted[i]) / 255.0;
                }
            }
        }
        const double iou = double(both_) / double(std::max<size_t>(either, 1));
        const glm::dvec3 difference = glm::abs(sumA - sumB) / double(std::max<size_t>(both_, 1));
        std::printf("view %d: IoU %.3f, mean colour difference %.3f %.3f %.3f\n", index, iou, difference.x,
                    difference.y, difference.z);
        // Measured 0.83 to 0.98, and 0.05 summed over the channels.
        EXPECT_GT(iou, 0.8) << index;
        EXPECT_LT(difference.x + difference.y + difference.z, 0.08) << index;
        ++index;
    }
}
