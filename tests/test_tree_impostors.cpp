// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - hemiOctahedronRoundTrips: directions of the upper hemisphere map into [-1, 1]^2 and back, directions below
 *   the horizon map to the horizon, and the cards' axes are orthonormal with up as close to +y as possible
 * - atlasViewsShowTheTree: every view of the atlas shows the tree in its middle, covering a reasonable share of
 *   the view, with normals of unit length; winter crowns cover less than summer ones; the atlases are written to
 *   build/TestingOutput/tree/
 * - impostorMatchesTheTree: from the side, at 40 and at 70 degrees above, an impostor has nearly the silhouette of
 *   the generated tree at the size impostors are used at (48 pixels tall): as many pixels covered (within 10 %), the
 *   same outline (IoU of blocks of 2 x 2 pixels above 0.65; the thinned leaves speckle differently) and a similar
 *   mean colour; images are written to build/TestingOutput/tree/
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>

#include <glm/gtc/quaternion.hpp>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/tree/bonsai_scene.hpp"
#include "klartraum/tree/tree_impostors.hpp"

#include "tree_test_helpers.hpp"

using namespace klartraum;
using namespace klartraum::tree_test;

namespace {

// The tree types of the scene at scale 0.75 (natural, not bonsai-styled).
std::vector<TreeTypeParams> forestTypes() {
    BonsaiSceneConfig config;
    config.scale.s = 0.75f;
    return makeBonsaiScene(config).types;
}

glm::vec4 unpack(uint32_t v) {
    return glm::vec4(float(v & 255u), float((v >> 8) & 255u), float((v >> 16) & 255u), float(v >> 24)) / 255.0f;
}

} // namespace

TEST(TreeImpostors, hemiOctahedronRoundTrips) {
    for (int k = 0; k < 2000; ++k) {
        const float a = 2.0f * 3.14159265f * tree::treeRand(5u, 2u * k);
        const float e = 0.5f * 3.14159265f * tree::treeRand(5u, 2u * k + 1u);
        const glm::vec3 d(std::cos(e) * std::cos(a), std::sin(e), std::cos(e) * std::sin(a));
        const glm::vec2 uv = tree::treeHemiOctEncode(d);
        EXPECT_LE(std::abs(uv.x), 1.0f + 1e-5f);
        EXPECT_LE(std::abs(uv.y), 1.0f + 1e-5f);
        const glm::vec3 back = tree::treeHemiOctDecode(uv);
        EXPECT_NEAR(glm::length(back - d), 0.0f, 1e-4f) << k;
        const glm::vec3 right = tree::treeImpostorRight(d), up = tree::treeImpostorUp(d);
        EXPECT_NEAR(glm::length(right), 1.0f, 1e-4f);
        EXPECT_NEAR(glm::length(up), 1.0f, 1e-4f);
        EXPECT_NEAR(glm::dot(right, up), 0.0f, 1e-4f);
        EXPECT_NEAR(glm::dot(right, d), 0.0f, 1e-4f);
        EXPECT_NEAR(glm::dot(up, d), 0.0f, 1e-4f);
        // Up on the screen points up, except looking (nearly) straight down, where any up will do.
        if (d.y < 0.999f) {
            EXPECT_GE(up.y, -1e-5f);
        }
    }
    // Below the horizon: on it.
    const glm::vec3 below = glm::normalize(glm::vec3(0.6f, -0.5f, 0.3f));
    const glm::vec3 horizon = tree::treeHemiOctDecode(tree::treeHemiOctEncode(below));
    EXPECT_NEAR(horizon.y, 0.0f, 1e-5f);
    EXPECT_NEAR(glm::length(glm::vec2(horizon.x, horizon.z) - glm::normalize(glm::vec2(below.x, below.z))), 0.0f,
                1e-4f);
}

class TreeImpostorsTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
    }
    void TearDown() override { frontend.reset(); }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
};

TEST_F(TreeImpostorsTest, atlasViewsShowTheTree) {
    TreeImpostorConfig config;
    config.seasons = {0.2f, 2.0f};
    const TreeImpostorAtlas atlas(*vc, forestTypes(), config);
    const TreeImpostorParams& p = atlas.getParams();
    const uint32_t views = p.grid.x, pixels = p.grid.y, side = atlas.side();
    for (uint32_t type = 0; type < p.grid.w; ++type) {
        EXPECT_GT(p.bounds[type].w, 0.0f);
        EXPECT_GT(p.heights[type], p.bounds[type].y);
        double covered[2] = {};
        for (uint32_t season = 0; season < 2; ++season) {
            const size_t layer = atlas.layer(type, season);
            for (uint32_t v = 0; v < views * views; ++v) {
                double coverage = 0.0;
                glm::dvec2 centroid(0.0);
                for (uint32_t y = 0; y < pixels; ++y) {
                    for (uint32_t x = 0; x < pixels; ++x) {
                        const size_t i = layer + size_t((v / views) * pixels + y) * side + (v % views) * pixels + x;
                        const glm::vec4 a = unpack(atlas.getAlbedo()[i]);
                        coverage += a.w;
                        centroid += double(a.w) * glm::dvec2(x + 0.5, y + 0.5);
                        if (a.w > 0.0f) {
                            const glm::vec3 n = 2.0f * glm::vec3(unpack(atlas.getNormals()[i])) - glm::vec3(1.0f);
                            EXPECT_NEAR(glm::length(n), 1.0f, 0.03f);
                        }
                    }
                }
                const double share = coverage / double(pixels * pixels);
                if (season == 1) {
                    // A crown in leaf from any view: neither a speck nor the whole view.
                    EXPECT_GT(share, 0.05) << type << " " << v;
                    EXPECT_LT(share, 0.85) << type << " " << v;
                }
                if (coverage > 0.0) {
                    centroid /= coverage;
                    EXPECT_LT(glm::length(centroid - glm::dvec2(0.5 * pixels)), 0.25 * pixels) << type << " " << v;
                }
                covered[season] += coverage;
            }
        }
        for (uint32_t season = 0; season < 2; ++season) {
            // Albedo over black, scaled by coverage.
            std::vector<uint8_t> image(size_t(side) * side * 4);
            for (size_t i = 0; i < size_t(side) * side; ++i) {
                const glm::vec4 a = unpack(atlas.getAlbedo()[atlas.layer(type, season) + i]);
                for (int c = 0; c < 3; ++c) {
                    image[i * 4 + 2 - c] = uint8_t(std::lround(255.0f * a[c] * a.w));
                }
            }
            writePpm("impostor_atlas_type" + std::to_string(type) + "_season" + std::to_string(season), image,
                     {side, side});
        }
        // The broadleaf is bare in winter; the pine keeps its needles.
        if (type == 0) {
            EXPECT_LT(covered[0], 0.8 * covered[1]);
        }
        std::printf("type %u: coverage winter %.0f, summer %.0f texels\n", type, covered[0], covered[1]);
    }
}

TEST_F(TreeImpostorsTest, impostorMatchesTheTree) {
    const std::vector<TreeTypeParams> types = forestTypes();
    TreeImpostorConfig config;
    config.seasons = {2.0f};
    const TreeImpostorAtlas atlas(*vc, types, config);
    const VkExtent2D extent{256, 256};
    for (uint32_t type = 0; type < 2; ++type) {
        // The tree: 1 unit tall, turned by 0.7 radians.
        const float height = 1.0f;
        const float scale = height / atlas.getParams().heights[type];
        const float yaw = 0.7f;
        const glm::vec4 up = treeUpRotation(glm::vec3(0.0f, 1.0f, 0.0f));
        const glm::quat rotation = glm::angleAxis(yaw, glm::vec3(0.0f, 1.0f, 0.0f)) * glm::quat(up.w, up.x, up.y, up.z);
        TreeInstance inst{};
        inst.posScale = glm::vec4(0.0f, 0.0f, 0.0f, scale);
        inst.rot = glm::vec4(rotation.x, rotation.y, rotation.z, rotation.w);
        inst.ids = glm::uvec4(type, 1u, 0u, 0u);
        TreeScene real(*vc, types, {inst}, extent, true, true);

        // The impostor of the same tree.
        TreeSite site{};
        site.posHeight = glm::vec4(0.0f, 0.0f, 0.0f, height);
        site.kind = glm::vec4(type == 1 ? TREE_SITE_CONIFER : TREE_SITE_BROADLEAF, yaw, 1.0f, 1.0f);
        auto sites = std::make_shared<BufferElementSinglePath<VulkanBuffer<TreeSite>>>(*vc, 1u);
        sites->getBuffer().memcopyFrom(std::vector<TreeSite>{site});
        auto args = std::make_shared<BufferElementSinglePath<VulkanBuffer<VkDrawIndirectCommand>>>(
            *vc, 1u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
        args->getBuffer().memcopyFrom(std::vector<VkDrawIndirectCommand>{{6u, 1u, 0u, 0u}});
        auto camera = std::make_shared<CameraUboType>();
        auto settings = std::make_shared<HostValues<TreeRenderSettings>>(
            *vc, std::vector<TreeRenderSettings>{defaultTreeRenderSettings(extent)});
        TreeImpostorRenderer impostor(*vc, {sites, -1}, {args, -1}, 0, atlas, settings, extent);
        auto target = std::make_shared<OffscreenTarget>(*vc, extent, 1);
        auto pass = std::make_shared<RenderPass>(vc->getSwapChainImageFormat(), extent);
        pass->setInput(target, 0);
        pass->setInput(camera, 1);
        pass->setDepthAttachment(true);
        pass->addComputeDependency(settings);
        pass->addDrawComponent(impostor.getDrawComponent());
        ComputeGraph graph(*vc, 1);
        graph.compileFrom(pass);

        for (float elevation : {0.1f, 0.7f, 1.2f}) {
            const glm::vec3 target3(0.0f, 0.5f * height, 0.0f);
            // 48 pixels tall: 256 / (2 tan 20 degrees * 7.3).
            const glm::vec3 eye =
                target3 + 7.3f * glm::vec3(std::cos(elevation) * 0.6f, std::sin(elevation), std::cos(elevation) * 0.8f);
            const std::vector<uint8_t> a = real.render(eye, target3);

            setCamera(camera->ubo, eye, target3, 40.0f, 1.0f);
            camera->update(0);
            graph.submitAndWait(vc->getGraphicsQueue(), 0);
            const std::vector<uint8_t> b =
                readImage(*vc, target->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
            const std::string name =
                "impostor_type" + std::to_string(type) + "_e" + std::to_string(int(elevation * 10));
            writePpm(name + "_tree", a, extent);
            writePpm(name + "_impostor", b, extent);
            // Silhouettes: blocks of 2 x 2 pixels with any pixel covered.
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
                    const bool pa = covered(a, extent, x, y), pb = covered(b, extent, x, y);
                    const size_t i = (size_t(y) * extent.width + x) * 4;
                    if (pa) {
                        ca += glm::dvec3(a[i + 2], a[i + 1], a[i]) / 255.0;
                        ++na;
                    }
                    if (pb) {
                        cb += glm::dvec3(b[i + 2], b[i + 1], b[i]) / 255.0;
                        ++nb;
                    }
                }
            }
            const double iou = double(both) / double(std::max<size_t>(either, 1));
            const glm::dvec3 d = glm::abs(ca / double(std::max<size_t>(na, 1)) - cb / double(std::max<size_t>(nb, 1)));
            std::printf("type %u, elevation %.1f: silhouette IoU %.3f, covered pixels %zu vs %zu, mean colour "
                        "difference %.3f %.3f %.3f\n",
                        type, elevation, iou, na, nb, d.x, d.y, d.z);
            // Measured: coverage within 7 %, IoU 0.70 to 0.88, colour differences up to 0.17 (the impostor's
            // normals are averaged over the crown, the tree's are per leaf).
            EXPECT_NEAR(double(nb), double(na), 0.1 * double(na)) << type << " " << elevation;
            EXPECT_GT(iou, 0.65) << type << " " << elevation;
            EXPECT_LT(d.x + d.y + d.z, 0.2) << type << " " << elevation;
        }
    }
}
