// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - gridCoversTheBounds: configureTreeLightGrid sizes the grid to cover the bounds with the voxel size and enables it
 * - shadeLitScalesSunAndSky: treeShadeLit with full light equals treeShade; no sun light leaves the ambient term,
 *   no sky light the sun term
 * - emptyGridIsFullyLit: with the tree outside the grid, every voxel sees the sun and the whole sky
 * - shadowFallsBelowTheCrown: with the sun straight above, the ground under the crown gets little sunlight and the
 *   ground beside it all; above the tree everything is lit
 * - shadowFollowsTheSun: with the sun low in the +x direction, the shadow falls towards -x
 * - crownInsideSeesLessSky: voxels in the middle of the crown see less sky than voxels beside it
 * - staticOccludersCastShadows: an opaque static plate (treeStaticOcclusion()) shadows what is below it but not
 *   itself, and the trees-only channel ignores it
 * - capturedShadeWeighsSunAndSky: treeCapturedShade is 1 in full light, the sky's share in the sun's shadow, and
 *   0 without light
 * - groundIsDarkerInTheShadow: a ground plane drawn with the light volume is darker under the tree than beside it
 **/

#include <gtest/gtest.h>

#include <memory>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/tree/scene_mesh_renderer.hpp"
#include "klartraum/tree/tree_light_volume.hpp"

#include "tree_test_helpers.hpp"

using namespace klartraum;
using namespace klartraum::tree;
using namespace klartraum::tree_test;

namespace {

// A dense round crown about 1 unit wide on a 1 unit trunk: the sassafras preset at scale 0.08.
std::vector<TreeTypeParams> crownTypes() {
    TreeTypeParams p = tree_presets::sassafras();
    p.leafCount.x = 30.0f;
    return {p};
}

struct LightScene {
    LightScene(VulkanContext& vc, glm::vec3 sun, glm::vec3 treePosition = glm::vec3(0.0f),
               const std::vector<TreeSceneVertex>& occluders = {}) {
        const std::vector<TreeTypeParams> types = crownTypes();
        const std::vector<TreeInstance> instances = {makeInstance(0, 4, treePosition, 0.08f)};
        const TreeFrame frame = defaultTreeFrame();
        generator = std::make_shared<TreeGenerator>(vc, types, instances,
                                                    TreeCapacity::fromReference(types, instances, frame), frame);
        values = defaultTreeRenderSettings({64, 64});
        values.sun = glm::vec4(glm::normalize(sun), 1.0f);
        configureTreeLightGrid(values, glm::vec3(-1.0f, 0.0f, -1.0f), glm::vec3(1.0f, 1.6f, 1.0f), 0.05f);
        settings = std::make_shared<HostValues<TreeRenderSettings>>(vc, std::vector<TreeRenderSettings>{values});
        volume = std::make_unique<TreeLightVolume>(vc, *generator, settings,
                                                   occluders.empty() ? std::vector<float>{}
                                                                     : treeStaticOcclusion(values, occluders, {}));
        graph = std::make_unique<ComputeGraph>(vc, 1);
        graph->compileFrom(volume->getElement());
        graph->submitAndWait(vc.getGraphicsQueue(), 0);
        light = volume->readLight();
    }

    // The light of the voxel around p, through everything.
    glm::vec2 at(glm::vec3 p) const { return glm::vec2(all(p)); }

    // xy through everything, zw through the trees only.
    glm::vec4 all(glm::vec3 p) const {
        const glm::ivec3 c = glm::ivec3(glm::floor((p - glm::vec3(values.lightGridMin)) / values.lightGridMin.w));
        return light[treeLightIndex(c.x, c.y, c.z, values.lightGridSize)];
    }

    TreeRenderSettings values;
    std::shared_ptr<TreeGenerator> generator;
    std::shared_ptr<HostValues<TreeRenderSettings>> settings;
    std::unique_ptr<TreeLightVolume> volume;
    std::unique_ptr<ComputeGraph> graph;
    std::vector<glm::vec4> light;
};

} // namespace

TEST(TreeLight, gridCoversTheBounds) {
    TreeRenderSettings s = defaultTreeRenderSettings({64, 64});
    EXPECT_EQ(s.lightGridSize.w, 0u);
    configureTreeLightGrid(s, glm::vec3(-1.0f, 0.0f, -0.5f), glm::vec3(1.0f, 2.0f, 0.52f), 0.1f);
    EXPECT_EQ(s.lightGridSize, glm::uvec4(20u, 20u, 11u, 1u));
    EXPECT_EQ(glm::vec3(s.lightGridMin), glm::vec3(-1.0f, 0.0f, -0.5f));
    EXPECT_EQ(s.lightGridMin.w, 0.1f);
}

TEST(TreeLight, shadeLitScalesSunAndSky) {
    const TreeRenderSettings s = defaultTreeRenderSettings({64, 64});
    const glm::vec3 albedo(0.5f, 0.4f, 0.3f);
    const glm::vec3 normal = glm::vec3(s.sun);
    const glm::vec3 full = treeShade(albedo, normal, s);
    EXPECT_EQ(treeShadeLit(albedo, normal, s, glm::vec2(1.0f)), full);
    const glm::vec3 ambient = treeShadeLit(albedo, normal, s, glm::vec2(0.0f, 1.0f));
    const glm::vec3 sun = treeShadeLit(albedo, normal, s, glm::vec2(1.0f, 0.0f));
    EXPECT_NEAR(glm::length(ambient + sun - full), 0.0f, 1e-6f);
    EXPECT_NEAR(glm::length(sun - albedo * s.sun.w), 0.0f, 1e-6f);
}

class TreeLightTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
    }
    void TearDown() override { frontend.reset(); }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
};

TEST_F(TreeLightTest, emptyGridIsFullyLit) {
    const LightScene scene(*vc, glm::vec3(0.3f, 1.0f, 0.2f), glm::vec3(10.0f, 0.0f, 0.0f));
    for (const glm::vec4& l : scene.light) {
        ASSERT_EQ(l, glm::vec4(1.0f));
    }
}

TEST_F(TreeLightTest, shadowFallsBelowTheCrown) {
    const LightScene scene(*vc, glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::vec2 under = scene.at(glm::vec3(0.1f, 0.02f, 0.1f));
    const glm::vec2 beside = scene.at(glm::vec3(0.9f, 0.02f, 0.9f));
    const glm::vec2 above = scene.at(glm::vec3(0.0f, 1.58f, 0.0f));
    std::cout << "sun under " << under.x << ", beside " << beside.x << ", above " << above.x << std::endl;
    EXPECT_LT(under.x, 0.5f);
    EXPECT_GT(beside.x, 0.99f);
    EXPECT_GT(above.x, 0.99f);
}

TEST_F(TreeLightTest, shadowFollowsTheSun) {
    const LightScene scene(*vc, glm::vec3(1.0f, 1.0f, 0.0f));
    // The crown is about 0.6 to 1.2 high: its shadow on the ground lies about that far towards -x.
    const float towards = scene.at(glm::vec3(-0.8f, 0.02f, 0.0f)).x;
    const float away = scene.at(glm::vec3(0.8f, 0.02f, 0.0f)).x;
    std::cout << "sun at -x " << towards << ", at +x " << away << std::endl;
    EXPECT_LT(towards, away - 0.3f);
}

TEST_F(TreeLightTest, crownInsideSeesLessSky) {
    const LightScene scene(*vc, glm::vec3(0.3f, 1.0f, 0.2f));
    // The middle of the crown and a point at the same height outside it.
    float inside = 1.0f;
    for (float y = 0.5f; y < 1.2f; y += 0.05f) {
        inside = std::min(inside, scene.at(glm::vec3(0.0f, y, 0.0f)).y);
    }
    const float outside = scene.at(glm::vec3(0.95f, 0.8f, 0.95f)).y;
    std::cout << "sky inside " << inside << ", outside " << outside << std::endl;
    EXPECT_LT(inside, 0.6f);
    EXPECT_GT(outside, 0.9f);
}

TEST_F(TreeLightTest, staticOccludersCastShadows) {
    // A plate 0.6 wide at height 1.4 over x in [0.2, 0.8], the tree far away.
    std::vector<TreeSceneVertex> plate;
    for (float x = 0.2f; x < 0.8f; x += 0.01f) {
        for (float z = -0.3f; z < 0.3f; z += 0.01f) {
            TreeSceneVertex v{};
            v.pos = glm::vec4(x, 1.4f, z, 0.01f);
            v.normal = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
            plate.push_back(v);
        }
    }
    const LightScene scene(*vc, glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(10.0f, 0.0f, 0.0f), plate);
    EXPECT_LT(scene.at(glm::vec3(0.5f, 0.5f, 0.0f)).x, 0.1f);   // under the plate
    EXPECT_GT(scene.at(glm::vec3(-0.5f, 0.5f, 0.0f)).x, 0.99f); // beside it
    EXPECT_GT(scene.at(glm::vec3(0.5f, 1.41f, 0.0f)).x, 0.99f); // the plate itself
    EXPECT_GT(scene.all(glm::vec3(0.5f, 0.5f, 0.0f)).z, 0.99f); // the trees-only channel ignores it
}

TEST(TreeLight, capturedShadeWeighsSunAndSky) {
    const TreeRenderSettings s = defaultTreeRenderSettings({64, 64});
    EXPECT_NEAR(treeCapturedShade(s, glm::vec2(1.0f, 1.0f)), 1.0f, 1e-6f);
    EXPECT_EQ(treeCapturedShade(s, glm::vec2(0.0f, 0.0f)), 0.0f);
    const float shadow = treeCapturedShade(s, glm::vec2(0.0f, 1.0f));
    EXPECT_GT(shadow, 0.1f);
    EXPECT_LT(shadow, 0.6f);
}

TEST_F(TreeLightTest, groundIsDarkerInTheShadow) {
    const VkExtent2D extent{128, 128};
    const std::vector<TreeTypeParams> types = crownTypes();
    const std::vector<TreeInstance> instances = {makeInstance(0, 4, glm::vec3(0.0f), 0.08f)};
    const TreeFrame frame = defaultTreeFrame();
    auto generator = std::make_shared<TreeGenerator>(*vc, types, instances,
                                                     TreeCapacity::fromReference(types, instances, frame), frame);
    TreeRenderSettings values = defaultTreeRenderSettings(extent);
    values.sun = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
    configureTreeLightGrid(values, glm::vec3(-1.0f, -0.05f, -1.0f), glm::vec3(1.0f, 1.6f, 1.0f), 0.05f);
    auto settings = std::make_shared<HostValues<TreeRenderSettings>>(*vc, std::vector<TreeRenderSettings>{values});
    TreeLightVolume volume(*vc, *generator, settings);

    TreeSceneVertex v{};
    v.normal = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
    v.color = glm::vec4(0.5f, 0.5f, 0.5f, TREE_MATERIAL_PLAIN);
    std::vector<TreeSceneVertex> ground;
    for (glm::vec2 c :
         {glm::vec2(-1, -1), glm::vec2(1, -1), glm::vec2(1, 1), glm::vec2(-1, -1), glm::vec2(1, 1), glm::vec2(-1, 1)}) {
        v.pos = glm::vec4(c.x, 0.0f, c.y, 1.0f);
        ground.push_back(v);
    }
    SceneMeshRenderer meshes(*vc, ground, settings, extent, TreeRenderInputs{{}, volume.getLight()});
    auto target = std::make_shared<OffscreenTarget>(*vc, extent, 1);
    auto camera = std::make_shared<CameraUboType>();
    auto pass = std::make_shared<RenderPass>(vc->getSwapChainImageFormat(), extent);
    pass->setInput(target, 0);
    pass->setInput(camera, 1);
    pass->setDepthAttachment(true);
    pass->addComputeDependency(volume.getElement());
    pass->addDrawComponent(meshes.getDrawComponent());
    ComputeGraph graph(*vc, 1);
    graph.compileFrom(pass);
    // From straight below the ground would hide the tree; look down from above, slightly tilted.
    setCamera(camera->ubo, glm::vec3(0.0f, 4.0f, 0.01f), glm::vec3(0.0f), 30.0f, 1.0f);
    camera->update(0);
    graph.submitAndWait(vc->getGraphicsQueue(), 0);
    const auto image = readImage(*vc, target->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
    writePpm("light_ground", image, extent);

    auto brightness = [&](glm::vec3 p) {
        const glm::vec4 clip = camera->ubo.proj * camera->ubo.view * glm::vec4(p, 1.0f);
        const uint32_t x = uint32_t((clip.x / clip.w * 0.5f + 0.5f) * extent.width);
        const uint32_t y = uint32_t((clip.y / clip.w * 0.5f + 0.5f) * extent.height);
        const size_t i = (size_t(y) * extent.width + x) * 4;
        return int(image[i]) + int(image[i + 1]) + int(image[i + 2]);
    };
    const int shadow = brightness(glm::vec3(0.15f, 0.0f, 0.15f));
    const int lit = brightness(glm::vec3(0.9f, 0.0f, 0.9f));
    std::cout << "ground brightness in the shadow " << shadow << ", beside " << lit << std::endl;
    EXPECT_LT(shadow, 0.8 * lit);
}
