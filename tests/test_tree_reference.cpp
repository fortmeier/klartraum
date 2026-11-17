// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - depthCarriesOverBetweenPasses: two trees drawn by two render passes, the second loading the first's image and
 *   depth (RenderPass::setDepthFrom()), give the image of both drawn in one pass, whichever is drawn first
 * - batchesComposeLikeOnePass: a forest view of TreeReferenceRenderer with batches of 1024 and of 64 trees draws
 *   hundreds of far trees covering the same pixels (masks equal) and the same image outside them; inside, only
 *   the trees' tints differ (the classical leaves vary their colour by their tree's slot in its batch)
 **/

#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>

#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/tree/bonsai_scene.hpp"
#include "klartraum/tree/tree_reference.hpp"
#include "klartraum/tree/tree_leaf_renderer.hpp"
#include "klartraum/tree/tree_stem_renderer.hpp"

#include "tree_test_helpers.hpp"

using namespace klartraum;
using namespace klartraum::tree_test;

namespace {

// One tree's stems and leaves drawn into `target` by a pass of its own, which either clears the image and keeps its
// depth for others, or draws over the image with the depth of `depthFrom`.
struct TreeLayer {
    TreeLayer(VulkanContext& vc, const std::vector<TreeTypeParams>& types, std::vector<TreeInstance> instances,
              std::shared_ptr<OffscreenTarget> target, std::shared_ptr<CameraUboType> camera,
              std::shared_ptr<RenderPass> depthFrom)
        : vc(vc) {
        const TreeFrame frame = defaultTreeFrame();
        const VkExtent2D extent = target->getImageExtent(0);
        generator = std::make_shared<TreeGenerator>(vc, types, instances,
                                                    TreeCapacity::fromReference(types, instances, frame), frame);
        settings = std::make_shared<HostValues<TreeRenderSettings>>(
            vc, std::vector<TreeRenderSettings>{defaultTreeRenderSettings(extent)});
        pass = std::make_shared<RenderPass>(vc.getSwapChainImageFormat(), extent);
        if (depthFrom) {
            auto toGeneral = std::make_shared<ImageToGeneral>();
            toGeneral->setInput(std::make_shared<OffscreenTargetView>(target), 0);
            pass->setInput(toGeneral, 0);
            pass->setLoadExisting(true);
            pass->setDepthFrom(depthFrom);
        } else {
            pass->setInput(target, 0);
            pass->setDepthAttachment(true);
            pass->setStoreDepth(true);
        }
        pass->setInput(camera, 1);
        stems = std::make_unique<TreeStemRenderer>(vc, *generator, camera, settings, extent);
        leaves = std::make_unique<TreeLeafRenderer>(vc, *generator, types, camera, settings, extent);
        pass->addComputeDependency(stems->getClassify());
        pass->addComputeDependency(leaves->getClassify());
        pass->addDrawComponent(stems->getDrawComponent());
        pass->addDrawComponent(leaves->getDrawComponent());
        graph = std::make_unique<ComputeGraph>(vc, 1);
        graph->compileFrom(pass);
    }

    VulkanContext& vc;
    std::shared_ptr<TreeGenerator> generator;
    std::shared_ptr<HostValues<TreeRenderSettings>> settings;
    std::unique_ptr<TreeStemRenderer> stems;
    std::unique_ptr<TreeLeafRenderer> leaves;
    std::shared_ptr<RenderPass> pass;
    std::unique_ptr<ComputeGraph> graph;
};

} // namespace

class TreeReferenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
    }
    void TearDown() override { frontend.reset(); }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
};

TEST_F(TreeReferenceTest, depthCarriesOverBetweenPasses) {
    const std::vector<TreeTypeParams> types = {tree_presets::sassafras()};
    // The second tree stands behind the first, half hidden.
    const TreeInstance front = makeInstance(0, 2, glm::vec3(0.0f), 0.1f);
    const TreeInstance back = makeInstance(0, 5, glm::vec3(0.4f, 0.0f, -1.5f), 0.1f);
    const VkExtent2D extent{256, 256};
    const glm::vec3 eye(0.0f, 1.0f, 4.0f), lookAt(0.0f, 1.0f, 0.0f);

    TreeScene both(*vc, types, {front, back}, extent, true, true);
    const std::vector<uint8_t> expected = both.render(eye, lookAt);
    ASSERT_GT(coveredCount(expected, extent), 2000u);

    for (bool frontFirst : {true, false}) {
        auto target = std::make_shared<OffscreenTarget>(*vc, extent, 1);
        auto camera = std::make_shared<CameraUboType>();
        setCamera(camera->ubo, eye, lookAt, 40.0f, 1.0f);
        TreeLayer first(*vc, types, {frontFirst ? front : back}, target, camera, nullptr);
        TreeLayer second(*vc, types, {frontFirst ? back : front}, target, camera, first.pass);
        camera->update(0);
        first.graph->submitAndWait(vc->getGraphicsQueue(), 0);
        second.graph->submitAndWait(vc->getGraphicsQueue(), 0);
        const std::vector<uint8_t> image =
            readImage(*vc, target->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
        int largest = 0;
        size_t differing = 0;
        for (size_t i = 0; i < image.size(); ++i) {
            const int d = std::abs(int(image[i]) - int(expected[i]));
            largest = std::max(largest, d);
            differing += d > 2;
        }
        writePpm(std::string("reference_depth_") + (frontFirst ? "front_first" : "back_first"), image, extent);
        writePpm("reference_depth_one_pass", expected, extent);
        std::printf("%s tree first: largest difference %d, %zu values differ by more than 2\n",
                    frontFirst ? "front" : "back", largest, differing);
        EXPECT_LE(differing, image.size() / 1000) << frontFirst;
    }
}

TEST_F(TreeReferenceTest, batchesComposeLikeOnePass) {
    BonsaiSceneConfig sceneConfig;
    sceneConfig.scale.s = 1.0f;
    sceneConfig.seed = 7;
    const BonsaiScene scene = makeBonsaiScene(sceneConfig);
    const float m = 1.0f / scene.config.scale.metersPerUnit();
    // 30 m above the soil of the scene's focus, looking down across it.
    const glm::vec3 foot(scene.focus.x, scene.soilHeight(scene.focus.x, scene.focus.z), scene.focus.z);
    const glm::vec3 eye = foot + glm::vec3(0.0f, 30.0f * m, 0.0f);
    const glm::vec3 lookAt = eye + glm::vec3(0.8f, -0.1f, -0.6f);
    const VkExtent2D extent{320, 180};
    std::vector<uint8_t> images[2], masks[2];
    uint32_t trees[2] = {0, 0};
    for (int k = 0; k < 2; ++k) {
        TreeReferenceConfig config;
        config.batch = k == 0 ? 1024u : 64u;
        config.rangeMeters = 400.0f;
        TreeReferenceRenderer reference(*vc, scene, extent, config);
        const TreeReferenceRenderer::Frame frame = reference.render(eye, lookAt, 50.0f, 1.0f * m, 50.0f, 2.0f, 0.5f);
        images[k] = frame.image;
        masks[k] = frame.mask;
        trees[k] = frame.trees;
        size_t masked = 0;
        for (uint8_t v : frame.mask) {
            masked += v > 0;
        }
        std::printf("batches of %u: %u far trees in %u batches, %zu pixels masked\n", config.batch, frame.trees,
                    frame.batches, masked);
        EXPECT_GT(frame.trees, 200u);
        EXPECT_GT(masked, frame.mask.size() / 20);
        writePpm(std::string("reference_batches_") + std::to_string(config.batch), frame.image, extent);
    }
    EXPECT_EQ(trees[0], trees[1]);
    size_t maskDiffers = 0, outsideDiffers = 0, inside = 0;
    double insideDifference = 0.0;
    for (size_t p = 0; p < masks[0].size(); ++p) {
        maskDiffers += masks[0][p] != masks[1][p];
        int largest = 0;
        for (int c = 0; c < 3; ++c) {
            largest = std::max(largest, std::abs(int(images[0][4 * p + c]) - int(images[1][4 * p + c])));
        }
        if (masks[0][p] || masks[1][p]) {
            insideDifference += largest;
            ++inside;
        } else {
            outsideDiffers += largest > 2;
        }
    }
    const double meanInside = inside > 0 ? insideDifference / double(inside) : 0.0;
    std::printf("masks differ in %zu pixels; outside them %zu pixels differ; inside, the mean difference is %.2f\n",
                maskDiffers, outsideDiffers, meanInside);
    EXPECT_LE(maskDiffers, masks[0].size() / 1000);
    EXPECT_LE(outsideDiffers, masks[0].size() / 1000);
    EXPECT_LT(meanInside, 6.0);
}
