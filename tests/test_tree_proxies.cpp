// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - visibleTreesAreThoseInTheView: the visible trees are exactly the resident trees whose box is in the view and
 *   that span at least minPixels pixels, as the CPU finds them
 * - lookingAwayShowsNoTrees: looking up into the sky, no tree is visible
 * - aFullListCountsTheRest: with a small capacity the list is full, the rest is counted, and the indirect draw
 *   holds the capacity
 * - theRendererDrawsNearTreesAndProxies: at scale 0.75 the renderer generates the trees nearest to the camera in
 *   full and draws the others as proxies, none of them within the near trees' radius
 * - canopyAndProxiesCrossfade: looking down on a forest at scale 1 from farther and farther away, as its trees shrink
 *   from 30 to 2 pixels and the canopy takes over from the proxies, the forest's mean colour changes gradually in
 *   both renderers and nothing of it turns black
 * - proxiesRenderBothWays: looking down onto a forest edge in summer, the proxies cover nearly the same pixels
 *   classically and as Gaussians (IoU above 0.6); images and the GPU time of a view over the forest are written
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>

#include "klartraum/gaussian_splatting_factory.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/tree/bonsai_renderer.hpp"
#include "klartraum/tree/tree_gaussians.hpp"
#include "klartraum/tree/tree_proxies.hpp"

#include "tree_test_helpers.hpp"

using namespace klartraum;
using namespace klartraum::tree_test;

namespace {

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

// About 60 m above a forested slope at scale 1, looking along it.
Eye overTheForest(const BonsaiScene& scene) {
    const glm::vec2 at(-0.75f, 0.2f);
    const glm::vec3 eye(at.x, scene.soilHeight(at.x, at.y) + 60.0f / scene.config.scale.metersPerUnit(), at.y);
    return {eye, eye + glm::vec3(0.3f, -0.02f, -0.1f)};
}

// About 250 m above a forest edge at scale 1, looking down onto it.
Eye downOntoAnEdge(const BonsaiScene& scene) {
    const glm::vec2 at(0.83f, -0.15f);
    const glm::vec3 ground(at.x, scene.soilHeight(at.x, at.y), at.y);
    return {ground + glm::vec3(-0.04f, 250.0f / scene.config.scale.metersPerUnit(), 0.04f), ground};
}

void setSeason(HostValues<TreeRenderSettings>& settings, float season) {
    TreeRenderSettings values = settings.values().at(0);
    values.season = glm::vec4(season, tree::treeSnowCover(season), 0.0f, 0.0f);
    settings.set(0, values);
}

// Tiles, proxies and their Gaussians, drawn alone into two black targets.
struct ProxyScene {
    ProxyScene(VulkanContext& vc, const BonsaiScene& scene, VkExtent2D extent, TreeProxyConfig config = {})
        : vc(vc),
          scene(scene),
          extent(extent) {
        camera = std::make_shared<CameraUboType>();
        settings = std::make_shared<HostValues<TreeRenderSettings>>(
            vc, std::vector<TreeRenderSettings>{defaultTreeRenderSettings(extent)});
        TreeTileLayout layout;
        layout.range = 0.3f;
        placement = makeTreePlacement(scene);
        tiles = std::make_unique<TreePlacementTiles>(vc, scene.terrain, placement, layout);
        proxies = std::make_unique<TreeProxies>(vc, *tiles, camera, settings, extent, TreeRenderInputs{}, config);
        classicalTarget = std::make_shared<OffscreenTarget>(vc, extent, 1);
        pass = std::make_shared<RenderPass>(vc.getSwapChainImageFormat(), extent);
        pass->setInput(classicalTarget, 0);
        pass->setInput(camera, 1);
        pass->setDepthAttachment(true);
        pass->addComputeDependency(proxies->getElement());
        pass->addComputeDependency(settings);
        pass->addDrawComponent(proxies->getDrawComponent());
        classicalGraph = std::make_unique<ComputeGraph>(vc, 1);
        classicalGraph->enableProfiling();
        classicalGraph->compileFrom(pass);

        TreeTypeParams p = tree_presets::quakingAspen();
        p.ratios.x = 1.0f;
        p.leafCount.x = 0.0f;
        const std::vector<TreeTypeParams> types = {p};
        const std::vector<TreeInstance> trees = {makeInstance(0, 1, glm::vec3(10.0f, 0.0f, 0.0f), 0.01f)};
        generator = std::make_shared<TreeGenerator>(vc, types, trees,
                                                    TreeCapacity::fromReference(types, trees, defaultTreeFrame()));
        TreeGaussianSources sources;
        sources.trees = false;
        sources.extraSlots = 4u * config.capacity;
        gaussians = std::make_unique<TreeGaussians>(vc, *generator, camera, settings, std::move(sources));
        proxyGaussians = std::make_unique<TreeProxyGaussians>(
            vc, *proxies, gaussians->getGaussians(), gaussians->getExtraOffset(), config.capacity, settings, camera);
        gaussianTarget = std::make_shared<OffscreenTarget>(vc, extent, 1);
        splatting =
            createGaussianSplatting(vc, GsplatBackend::Raster, gaussianTarget, camera, proxyGaussians->getGaussians());
        gaussianGraph = std::make_unique<ComputeGraph>(vc, 1);
        gaussianGraph->compileFrom(splatting);
        gaussians->initializeBuffers(1);
    }

    void setView(const Eye& e) {
        setCamera(camera->ubo, e.eye, e.target, 40.0f, float(extent.width) / float(extent.height), 1e-5f, 100.0f);
        camera->update(0);
        tiles->update(e.eye);
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

    // The resident trees that the cull pass keeps, found on the CPU.
    std::vector<TreeSite> visibleOnTheCpu(const Eye& e) {
        const glm::mat4 viewProj = camera->ubo.proj * camera->ubo.view;
        const float focal = std::abs(camera->ubo.proj[1][1]) * 0.5f * float(extent.height);
        const int n = int(tiles->getTileConfig().tiles.x);
        std::vector<TreeSite> result;
        for (const glm::ivec4& slot : tiles->getCache().getSlots()) {
            if (slot.w == 0) {
                continue;
            }
            for (const TreeSite& site :
                 treeSitesIn(scene.terrain, placement, slot.x * n, slot.y * n, (slot.x + 1) * n, (slot.y + 1) * n)) {
                const float h = site.posHeight.w;
                const glm::vec4 shape = tree::treeProxyShape(site.kind.x);
                const glm::vec3 foot(site.posHeight), r(shape.z * h, 0.0f, shape.z * h);
                if (tree::treeBoxOutsideFrustum(viewProj, foot - r, foot + r + glm::vec3(0.0f, h, 0.0f))) {
                    continue;
                }
                if (tree::treePixelDiameter(e.eye, foot + glm::vec3(0.0f, 0.5f * h, 0.0f), 0.5f * h, focal) <
                    proxies->getConfig().minPixels) {
                    continue;
                }
                result.push_back(site);
            }
        }
        return result;
    }

    VulkanContext& vc;
    const BonsaiScene& scene;
    VkExtent2D extent;
    TreePlacementParams placement{};
    std::shared_ptr<CameraUboType> camera;
    std::shared_ptr<HostValues<TreeRenderSettings>> settings;
    std::unique_ptr<TreePlacementTiles> tiles;
    std::unique_ptr<TreeProxies> proxies;
    std::shared_ptr<OffscreenTarget> classicalTarget;
    std::shared_ptr<RenderPass> pass;
    std::unique_ptr<ComputeGraph> classicalGraph;
    std::shared_ptr<TreeGenerator> generator;
    std::unique_ptr<TreeGaussians> gaussians;
    std::unique_ptr<TreeProxyGaussians> proxyGaussians;
    std::shared_ptr<OffscreenTarget> gaussianTarget;
    std::shared_ptr<ComputeGraphElement> splatting;
    std::unique_ptr<ComputeGraph> gaussianGraph;
};

} // namespace

class TreeProxiesTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
    }
    void TearDown() override { frontend.reset(); }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
};

TEST_F(TreeProxiesTest, visibleTreesAreThoseInTheView) {
    const BonsaiScene scene = sceneAt(1.0f);
    ProxyScene p(*vc, scene, {640, 360});
    const Eye e = overTheForest(scene);
    p.renderClassical(e);
    std::vector<TreeSite> gpu = p.proxies->readVisible();
    std::vector<TreeSite> cpu = p.visibleOnTheCpu(e);
    auto less = [](const TreeSite& a, const TreeSite& b) {
        return std::tie(a.posHeight.x, a.posHeight.z) < std::tie(b.posHeight.x, b.posHeight.z);
    };
    std::sort(gpu.begin(), gpu.end(), less);
    std::sort(cpu.begin(), cpu.end(), less);
    std::printf("visible: %zu on the GPU, %zu on the CPU\n", gpu.size(), cpu.size());
    EXPECT_GT(cpu.size(), 1000u);
    // Heights from the GPU's sines and cosines can move a tree at the edge of the view or of minPixels.
    EXPECT_NEAR(double(gpu.size()), double(cpu.size()), 0.002 * cpu.size() + 2.0);
    size_t matched = 0;
    for (const TreeSite& g : gpu) {
        matched += std::binary_search(cpu.begin(), cpu.end(), g, less) ? 1 : 0;
    }
    EXPECT_GT(matched, gpu.size() * 99 / 100);
    EXPECT_EQ(p.proxies->readCounter()[1], 0u);
}

TEST_F(TreeProxiesTest, lookingAwayShowsNoTrees) {
    const BonsaiScene scene = sceneAt(1.0f);
    ProxyScene p(*vc, scene, {320, 180});
    Eye e = overTheForest(scene);
    e.target = e.eye + glm::vec3(0.01f, 1.0f, 0.0f);
    p.renderClassical(e);
    EXPECT_EQ(p.proxies->readCounter()[0], 0u);
}

TEST_F(TreeProxiesTest, aFullListCountsTheRest) {
    const BonsaiScene scene = sceneAt(1.0f);
    TreeProxyConfig config;
    config.capacity = 500;
    ProxyScene p(*vc, scene, {320, 180}, config);
    p.renderClassical(overTheForest(scene));
    const std::vector<uint32_t> counter = p.proxies->readCounter();
    EXPECT_GT(counter[0], 500u);
    EXPECT_EQ(counter[1], counter[0] - 500u);
    EXPECT_EQ(p.proxies->readVisible().size(), 500u);
}

TEST_F(TreeProxiesTest, theRendererDrawsNearTreesAndProxies) {
    BonsaiSceneConfig sceneConfig;
    sceneConfig.scale.s = 0.75f;
    const BonsaiScene scene = makeBonsaiScene(sceneConfig);
    const VkExtent2D extent{480, 270};
    auto target = std::make_shared<OffscreenTarget>(*vc, extent, 1);
    BonsaiRenderConfig config;
    config.extent = extent;
    BonsaiRenderer renderer(*vc, scene, target, config);
    ComputeGraph graph(*vc, 1);
    graph.compileFrom(renderer.getOutput());
    renderer.initialize(1);
    const glm::vec3 eye(0.9f, 1.2f, 1.6f);
    setCamera(renderer.getCamera()->ubo, eye, glm::vec3(0.0f, 0.4f, 0.0f), 40.0f, 16.0f / 9.0f);
    renderer.getCamera()->update(0);
    renderer.setView(eye);
    for (int frame = 0; frame < 8 && renderer.getPendingTiles() > 0; ++frame) {
        graph.submitAndWait(vc->getGraphicsQueue(), 0);
        renderer.setView(eye);
    }
    graph.submitAndWait(vc->getGraphicsQueue(), 0);
    EXPECT_EQ(renderer.getPendingTiles(), 0u);
    writePpm("bonsai_placed_s75", readImage(*vc, target->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent),
             extent);
    // The near trees are the nearest sites.
    const std::vector<TreeSite> near = nearestTreeSites(scene.terrain, scene.placement, eye, sceneConfig.nearTrees);
    const std::vector<TreeInstance>& instances = renderer.getGenerator().getInstances().values();
    ASSERT_EQ(instances.size(), near.size());
    for (size_t i = 0; i < near.size(); ++i) {
        EXPECT_EQ(glm::vec3(instances[i].posScale), glm::vec3(near[i].posHeight));
    }
    // The proxies are the others.
    const float radius = glm::length(glm::vec3(near.back().posHeight) - eye);
    const std::vector<TreeSite> visible = renderer.getProxies()->readVisible();
    EXPECT_GT(visible.size(), 1000u);
    EXPECT_EQ(renderer.getProxies()->readCounter()[1], 0u);
    for (const TreeSite& site : visible) {
        EXPECT_GT(glm::length(glm::vec3(site.posHeight) - eye), radius);
    }
}

TEST_F(TreeProxiesTest, proxiesRenderBothWays) {
    const BonsaiScene scene = sceneAt(1.0f);
    const VkExtent2D extent{640, 360};
    ProxyScene p(*vc, scene, extent);
    setSeason(*p.settings, 2.0f);
    p.renderClassical(overTheForest(scene));
    p.classicalGraph->resetProfiling();
    writePpm("proxies_over_the_forest", p.renderClassical(overTheForest(scene)), extent);
    std::printf("%u trees visible, proxies take %.2f ms of GPU time at 640 x 360 (tiles resident)\n",
                p.proxies->readCounter()[0], p.classicalGraph->getGpuSpanMs());
    const Eye e = downOntoAnEdge(scene);
    const std::vector<uint8_t> classical = p.renderClassical(e);
    const std::vector<uint8_t> splatted = p.renderGaussians(e);
    writePpm("proxies_classical", classical, extent);
    writePpm("proxies_gaussian", splatted, extent);
    // Covered where brighter than the faint tails of the Gaussians: 0.1 summed over the channels.
    auto solid = [&](const std::vector<uint8_t>& image, uint32_t x, uint32_t y) {
        const size_t i = (size_t(y) * extent.width + x) * 4;
        return int(image[i]) + int(image[i + 1]) + int(image[i + 2]) > 25;
    };
    size_t both = 0, either = 0;
    for (uint32_t y = 0; y < extent.height; ++y) {
        for (uint32_t x = 0; x < extent.width; ++x) {
            const bool a = solid(classical, x, y), b = solid(splatted, x, y);
            both += (a && b) ? 1 : 0;
            either += (a || b) ? 1 : 0;
        }
    }
    const double iou = double(both) / double(std::max<size_t>(either, 1));
    std::printf("proxies classical vs Gaussian: IoU %.3f, covering %.2f of the image\n", iou,
                double(either) / double(extent.width * extent.height));
    EXPECT_GT(iou, 0.6);
    EXPECT_LT(either, size_t(extent.width * extent.height * 9 / 10));
}

TEST_F(TreeProxiesTest, canopyAndProxiesCrossfade) {
    BonsaiSceneConfig sceneConfig;
    sceneConfig.scale.s = 1.0f;
    const BonsaiScene scene = makeBonsaiScene(sceneConfig);
    // A forest in a valley (all forest within 0.04 units of it).
    const glm::vec2 at(-0.62f, 0.32f);
    const glm::vec3 ground(at.x, scene.soilHeight(at.x, at.y), at.y);
    const VkExtent2D extent{320, 180};
    for (int gaussian = 0; gaussian < 2; ++gaussian) {
        auto target = std::make_shared<OffscreenTarget>(*vc, extent, 1);
        BonsaiRenderConfig config;
        config.extent = extent;
        config.mode = gaussian ? BonsaiRenderMode::Gaussian : BonsaiRenderMode::Classical;
        BonsaiRenderer renderer(*vc, scene, target, config);
        ComputeGraph graph(*vc, 1);
        graph.compileFrom(renderer.getOutput());
        renderer.initialize(1);
        renderer.setSeason(2.0f);
        std::vector<glm::vec3> means;
        std::vector<float> pixels;
        for (int k = 0; k <= 16; ++k) {
            // From about 30 down to 2 pixels per tree.
            const float distance = 0.07f * std::pow(1.2f, float(k));
            const glm::vec3 eye = ground + distance * glm::normalize(glm::vec3(0.3f, 1.0f, 0.2f));
            setCamera(renderer.getCamera()->ubo, eye, ground, 40.0f, 16.0f / 9.0f, 1e-5f, 100.0f);
            renderer.getCamera()->update(0);
            renderer.setView(eye);
            for (int frame = 0; frame < 8 && renderer.getPendingTiles() > 0; ++frame) {
                graph.submitAndWait(vc->getGraphicsQueue(), 0);
                renderer.setView(eye);
            }
            graph.submitAndWait(vc->getGraphicsQueue(), 0);
            const std::vector<uint8_t> image =
                readImage(*vc, target->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
            if (k % 4 == 0) {
                writePpm(std::string(gaussian ? "crossfade_gaussian_" : "crossfade_classical_") + std::to_string(k),
                         image, extent);
            }
            // The middle of the image, which stays on the soil.
            glm::dvec3 sum(0.0);
            int black = 0, count = 0;
            for (uint32_t y = extent.height / 4; y < 3 * extent.height / 4; ++y) {
                for (uint32_t x = extent.width / 4; x < 3 * extent.width / 4; ++x) {
                    const size_t i = (size_t(y) * extent.width + x) * 4;
                    sum += glm::dvec3(image[i + 2], image[i + 1], image[i]) / 255.0;
                    black += covered(image, extent, x, y) ? 0 : 1;
                    ++count;
                }
            }
            EXPECT_EQ(black, 0) << gaussian << " " << k;
            means.push_back(glm::vec3(sum / double(count)));
            const float focal = 0.5f * float(extent.height) / std::tan(glm::radians(20.0f));
            pixels.push_back(scene.config.scale.treeHeightMeters() / scene.config.scale.metersPerUnit() * focal /
                             distance);
        }
        float largest = 0.0f;
        for (size_t k = 0; k < means.size(); ++k) {
            std::printf("%s: tree %5.1f px, mean %.3f %.3f %.3f\n", gaussian ? "Gaussian" : "classical", pixels[k],
                        means[k].x, means[k].y, means[k].z);
            if (k > 0) {
                const glm::vec3 d = glm::abs(means[k] - means[k - 1]);
                largest = std::max(largest, d.x + d.y + d.z);
            }
        }
        // The view's content changes with the distance too: steps of 20 % in distance change the mean by at most
        // a little, also where the canopy takes over (measured 0.024 classically with impostors, 0.045 as Gaussians).
        EXPECT_LT(largest, 0.08f) << gaussian;
    }
}
