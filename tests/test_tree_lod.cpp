// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - reducedTypesHaveFewerRecords: a reduced type has one stem level less, and its trees far fewer segments and
 *   leaves with at least the same leaf area (more, where branches fall short of their full count; the larger
 *   leaves overlap more, see reducedTreesLookAlike)
 * - reducedTreesLookAlike: at the size where trees switch to the reduced type (96 pixels tall), a reduced tree
 *   covers about as many pixels as the full one (within 20 %) with nearly the same outline (IoU of 2 x 2 blocks above
 *   0.7, the leaves speckle differently) and mean colour; images are written to build/TestingOutput/tree/
 * - gaussianLibraryHasTheTrees: the library holds per type and season a tree of the reduced type, as tall as
 *   reported, its Gaussians within the tree's reach, the broadleaf's crown sparse in winter
 * - gaussianForestCopiesTheVisibleTrees: at scale 0.75 the Gaussian renderer copies the library for the visible
 *   trees within its pool, none left out
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>

#include <glm/gtc/quaternion.hpp>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/tree/bonsai_renderer.hpp"
#include "klartraum/tree/tree_library.hpp"

#include "tree_test_helpers.hpp"

using namespace klartraum;
using namespace klartraum::tree_test;

namespace {

std::vector<TreeTypeParams> forestTypes() {
    BonsaiSceneConfig config;
    config.scale.s = 0.75f;
    return makeBonsaiScene(config).types;
}

TreeInstance upright(uint32_t type, float scale) {
    const glm::vec4 up = treeUpRotation(glm::vec3(0.0f, 1.0f, 0.0f));
    const glm::quat rotation = glm::angleAxis(0.7f, glm::vec3(0.0f, 1.0f, 0.0f)) * glm::quat(up.w, up.x, up.y, up.z);
    TreeInstance inst{};
    inst.posScale = glm::vec4(0.0f, 0.0f, 0.0f, scale);
    inst.rot = glm::vec4(rotation.x, rotation.y, rotation.z, rotation.w);
    inst.ids = glm::uvec4(type, 1u, 0u, 0u);
    return inst;
}

double leafArea(const TreeRecords& r) {
    double area = 0.0;
    for (const LeafRecord& l : r.leaves) {
        area += double(l.posScale.w) * l.posScale.w;
    }
    return area;
}

} // namespace

TEST(TreeLod, reducedTypesHaveFewerRecords) {
    const std::vector<TreeTypeParams> types = forestTypes();
    for (uint32_t type = 0; type < types.size(); ++type) {
        std::vector<TreeTypeParams> reduced = types;
        reduced[type] = treeReducedType(types[type]);
        EXPECT_EQ(reduced[type].ratios.x, types[type].ratios.x - 1.0f);
        const TreeRecords full = generateTreesReference(types, {upright(type, 1.0f)}, defaultTreeFrame());
        const TreeRecords less = generateTreesReference(reduced, {upright(type, 1.0f)}, defaultTreeFrame());
        std::printf("type %u: segments %zu -> %zu, leaves %zu -> %zu, leaf area %.1f -> %.1f\n", type,
                    full.segments.size(), less.segments.size(), full.leaves.size(), less.leaves.size(), leafArea(full),
                    leafArea(less));
        EXPECT_LT(less.segments.size(), full.segments.size() / 2);
        EXPECT_LT(less.leaves.size(), full.leaves.size() / 2);
        // Measured 1.4 to 1.6 times the area.
        EXPECT_GE(leafArea(less), leafArea(full));
        EXPECT_LT(leafArea(less), 2.0 * leafArea(full));
    }
}

class TreeLodTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
    }
    void TearDown() override { frontend.reset(); }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
};

TEST_F(TreeLodTest, reducedTreesLookAlike) {
    const std::vector<TreeTypeParams> types = forestTypes();
    const VkExtent2D extent{256, 256};
    for (uint32_t type = 0; type < types.size(); ++type) {
        std::vector<TreeTypeParams> reduced = types;
        reduced[type] = treeReducedType(types[type]);
        // About 13 m tall at scale 1: 1 unit at this scale.
        const float scale = 1.0f / types[type].shapeBase.z;
        TreeScene full(*vc, types, {upright(type, scale)}, extent, true, true);
        TreeScene less(*vc, reduced, {upright(type, scale)}, extent, true, true);
        for (float elevation : {0.1f, 0.7f}) {
            const glm::vec3 target(0.0f, 0.5f, 0.0f);
            // 96 pixels tall: 256 / (2 tan 20 degrees * 3.66).
            const glm::vec3 eye =
                target + 3.66f * glm::vec3(std::cos(elevation) * 0.6f, std::sin(elevation), std::cos(elevation) * 0.8f);
            const std::vector<uint8_t> a = full.render(eye, target);
            const std::vector<uint8_t> b = less.render(eye, target);
            const std::string name = "lod1_type" + std::to_string(type) + "_e" + std::to_string(int(elevation * 10));
            writePpm(name + "_full", a, extent);
            writePpm(name + "_reduced", b, extent);
            size_t both = 0, either = 0;
            for (uint32_t by = 0; by < extent.height / 2; ++by) {
                for (uint32_t bx = 0; bx < extent.width / 2; ++bx) {
                    bool pa = false, pb = false;
                    for (uint32_t k = 0; k < 4; ++k) {
                        pa = pa || covered(a, extent, 2 * bx + k % 2, 2 * by + k / 2);
                        pb = pb || covered(b, extent, 2 * bx + k % 2, 2 * by + k / 2);
                    }
                    both += (pa && pb) ? 1 : 0;
                    either += (pa || pb) ? 1 : 0;
                }
            }
            glm::dvec3 ca(0.0), cb(0.0);
            size_t na = 0, nb = 0;
            for (uint32_t y = 0; y < extent.height; ++y) {
                for (uint32_t x = 0; x < extent.width; ++x) {
                    const size_t i = (size_t(y) * extent.width + x) * 4;
                    if (covered(a, extent, x, y)) {
                        ca += glm::dvec3(a[i + 2], a[i + 1], a[i]) / 255.0;
                        ++na;
                    }
                    if (covered(b, extent, x, y)) {
                        cb += glm::dvec3(b[i + 2], b[i + 1], b[i]) / 255.0;
                        ++nb;
                    }
                }
            }
            const double iou = double(both) / double(std::max<size_t>(either, 1));
            const glm::dvec3 d = glm::abs(ca / double(std::max<size_t>(na, 1)) - cb / double(std::max<size_t>(nb, 1)));
            std::printf("type %u, elevation %.1f: IoU %.3f, covered %zu vs %zu, colour difference %.3f %.3f %.3f\n",
                        type, elevation, iou, na, nb, d.x, d.y, d.z);
            // Measured IoU 0.73 to 0.81, coverage within 12 %, colour within 0.01.
            EXPECT_GT(iou, 0.7) << type << " " << elevation;
            EXPECT_NEAR(double(nb), double(na), 0.2 * double(na)) << type << " " << elevation;
            EXPECT_LT(d.x + d.y + d.z, 0.1) << type << " " << elevation;
        }
    }
}

TEST(TreeLod, gaussianLibraryHasTheTrees) {
    const std::vector<TreeTypeParams> types = forestTypes();
    std::vector<float> heights;
    const std::vector<float> seasons = {0.2f, 2.0f};
    const std::vector<MossPatch> library = buildTreeGaussianLibrary(types, seasons, heights);
    ASSERT_EQ(library.size(), types.size() * seasons.size());
    for (uint32_t type = 0; type < types.size(); ++type) {
        // As tall as the type at scale 1 (Weber and Penn's scale, give or take its variation).
        EXPECT_NEAR(heights[type], types[type].shapeBase.z, 0.5f * types[type].shapeBase.z);
        for (size_t k = 0; k < seasons.size(); ++k) {
            const MossPatch& tree = library[type * seasons.size() + k];
            float top = 0.0f;
            for (const Gaussian3D& g : tree) {
                EXPECT_GT(g.position[1], -0.05f * heights[type]);
                EXPECT_LT(g.position[1], 1.05f * heights[type]);
                EXPECT_LT(glm::length(glm::vec2(g.position[0], g.position[2])), heights[type]);
                top = std::max(top, g.position[1]);
            }
            EXPECT_GT(top, 0.8f * heights[type]);
            std::printf("type %u, season %.1f: %zu Gaussians\n", type, seasons[k], tree.size());
        }
    }
    // The broadleaf is bare in winter.
    EXPECT_LT(library[0].size(), library[1].size() / 2);
}

TEST_F(TreeLodTest, gaussianForestCopiesTheVisibleTrees) {
    BonsaiSceneConfig sceneConfig;
    sceneConfig.scale.s = 0.75f;
    const BonsaiScene scene = makeBonsaiScene(sceneConfig);
    const VkExtent2D extent{480, 270};
    auto target = std::make_shared<OffscreenTarget>(*vc, extent, 1);
    BonsaiRenderConfig config;
    config.extent = extent;
    config.mode = BonsaiRenderMode::Gaussian;
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
    writePpm("gaussian_forest_s75", readImage(*vc, target->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent),
             extent);
    const uint32_t visible = renderer.getProxies()->readCounter()[0];
    const std::vector<uint32_t> copies = renderer.getForest()->getLod().readCounter();
    std::printf("%u visible trees, %u Gaussians written, %u trees left out\n", visible, copies[0], copies[1]);
    EXPECT_GT(visible, 1000u);
    EXPECT_GT(copies[0], 10u * visible);
    EXPECT_EQ(copies[1], 0u);
}
