// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - verticesLieOnTheTerrain: unmorphed vertices (close to the camera) have the terrain's height and its normal
 * - sharedEdgesMatch: along every edge two selected nodes share, also between levels, both nodes' surfaces have the
 *   same heights (no cracks), from outside the pot and at ground level
 * - geomorphIsContinuous: while the camera moves in small steps, the surface at fixed points changes far less
 *   than when levels switch without morphing
 * - terrainRendersWithoutHoles: rendered from above the peaks at either end of the pot against a black
 *   background, every column of the image is covered from the bottom up to the terrain's silhouette
 * - smallScalesRenderAsBefore: at scales 0 and 0.21 the bonsai with the terrain looks like the one with the soil
 *   mesh (PSNR above 40 dB); images are written to build/TestingOutput/tree/
 * - materialsFollowTheLandCover: seen from straight above at scale 1 in summer, rock is grey, meadows and forest
 *   green, fields more yellow than meadows, and rock brighter than forest
 * - snowLiesAboveTheSnowLine: in fall only gentle ground above the snow line is white; in winter most ground
 * - distantDetailDoesNotAlias: from outside the pot at scale 1 the image varies between neighbouring pixels about as
 *   much as one rendered at twice the resolution and filtered down
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/tree/bonsai_renderer.hpp"
#include "klartraum/tree/tree_terrain_renderer.hpp"

#include "tree_test_helpers.hpp"

using namespace klartraum;
using namespace klartraum::tree_test;

namespace {

BonsaiScene sceneAt(float s, uint32_t seed = 7) {
    BonsaiSceneConfig config;
    config.scale.s = s;
    config.seed = seed;
    config.treeCount = s > 0.3f ? 12u : 0u;
    return makeBonsaiScene(config);
}

struct Eye {
    glm::vec3 eye;
    glm::vec3 target;
};

Eye outsideEye() {
    const glm::vec3 focus(0.0f, 0.8f, 0.0f);
    return {focus +
                4.2f * glm::vec3(std::cos(0.32f) * std::sin(0.35f), std::sin(0.32f), std::cos(0.32f) * std::cos(0.35f)),
            focus};
}

// `meters` above the ground (2 m: standing on it), looking along the ground.
Eye groundEye(const BonsaiScene& scene, glm::vec2 at = glm::vec2(0.35f, 0.15f),
              glm::vec2 towards = glm::vec2(-0.9f, -0.3f), float meters = 2.0f) {
    const float above = meters / scene.config.scale.metersPerUnit();
    const glm::vec3 eye(at.x, scene.soilHeight(at.x, at.y) + above, at.y);
    return {eye, glm::vec3(towards.x, eye.y, towards.y)};
}

// The terrain alone, drawn into a black target.
struct TerrainScene {
    TerrainScene(VulkanContext& vc, const BonsaiScene& scene, VkExtent2D extent, TreeTerrainLodConfig lod = {})
        : vc(vc),
          extent(extent) {
        camera = std::make_shared<CameraUboType>();
        settings = std::make_shared<HostValues<TreeRenderSettings>>(
            vc, std::vector<TreeRenderSettings>{defaultTreeRenderSettings(extent)});
        terrain = std::make_unique<TreeTerrain>(vc, scene.terrain, camera, lod);
        renderer = std::make_unique<TreeTerrainRenderer>(vc, *terrain, settings, extent);
        target = std::make_shared<OffscreenTarget>(vc, extent, 1);
        pass = std::make_shared<RenderPass>(vc.getSwapChainImageFormat(), extent);
        pass->setInput(target, 0);
        pass->setInput(camera, 1);
        pass->setDepthAttachment(true);
        pass->addComputeDependency(terrain->getElement());
        pass->addComputeDependency(settings);
        pass->addDrawComponent(renderer->getDrawComponent());
        graph = std::make_unique<ComputeGraph>(vc, 1);
        graph->compileFrom(pass);
    }

    std::vector<uint8_t> render(const Eye& e, float fov = 40.0f) {
        setCamera(camera->ubo, e.eye, e.target, fov, float(extent.width) / float(extent.height), 1e-5f, 100.0f);
        camera->update(0);
        graph->submitAndWait(vc.getGraphicsQueue(), 0);
        return readImage(vc, target->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
    }

    VulkanContext& vc;
    VkExtent2D extent;
    std::shared_ptr<CameraUboType> camera;
    std::shared_ptr<HostValues<TreeRenderSettings>> settings;
    std::unique_ptr<TreeTerrain> terrain;
    std::unique_ptr<TreeTerrainRenderer> renderer;
    std::shared_ptr<OffscreenTarget> target;
    std::shared_ptr<RenderPass> pass;
    std::unique_ptr<ComputeGraph> graph;
};

// The colour of pixel (x, y) in [0, 1].
glm::vec3 pixel(const std::vector<uint8_t>& bgra, VkExtent2D extent, uint32_t x, uint32_t y) {
    const size_t i = (size_t(y) * extent.width + x) * 4;
    return glm::vec3(bgra[i + 2], bgra[i + 1], bgra[i]) / 255.0f;
}

float saturation(glm::vec3 c) {
    const float hi = std::max(c.x, std::max(c.y, c.z)), lo = std::min(c.x, std::min(c.y, c.z));
    return hi > 0.0f ? (hi - lo) / hi : 0.0f;
}

// Nearly orthographic from high above: where the ray through pixel (x, y) meets the terrain.
struct TopView {
    Eye eye{{0.0f, 60.0f, 3.0f}, {0.0f, 0.0f, 0.0f}};
    float fov = 2.9f;

    glm::vec3 ground(const BonsaiScene& scene, VkExtent2D extent, uint32_t x, uint32_t y) const {
        // The ray from the camera's basis (unprojecting with the tiny near plane loses precision); image rows run
        // down.
        const glm::vec2 ndc((x + 0.5f) / extent.width * 2.0f - 1.0f, (y + 0.5f) / extent.height * 2.0f - 1.0f);
        const glm::vec3 forward = glm::normalize(eye.target - eye.eye);
        const glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));
        const glm::vec3 up = glm::cross(right, forward);
        const float tanY = std::tan(glm::radians(0.5f * fov));
        const float tanX = tanY * float(extent.width) / float(extent.height);
        const glm::vec3 from = eye.eye;
        const glm::vec3 dir = glm::normalize(forward + ndc.x * tanX * right - ndc.y * tanY * up);
        // March to the ground's height (the ray is nearly vertical).
        glm::vec3 p = from + dir * ((from.y - 0.5f) / -dir.y);
        for (int i = 0; i < 8; ++i) {
            p = from + dir * ((scene.soilHeight(p.x, p.z) - from.y) / dir.y);
        }
        return p;
    }
};

// The height of node `k`'s surface at (x, z) inside or on the edge of the node: its two triangles per quad share
// the diagonal from (i, j) to (i + 1, j + 1).
float surfaceHeight(const std::vector<TreeTerrainNode>& nodes, const std::vector<glm::vec4>& vertices, int quads,
                    size_t k, float x, float z) {
    const TreeTerrainNode& n = nodes[k];
    const float u = std::clamp((x - n.x) / n.z * quads, 0.0f, float(quads));
    const float v = std::clamp((z - n.y) / n.z * quads, 0.0f, float(quads));
    const int i = std::min(int(u), quads - 1), j = std::min(int(v), quads - 1);
    const float fu = u - i, fv = v - j;
    const int row = quads + 1;
    auto h = [&](int a, int b) { return vertices[k * row * row + b * row + a].w; };
    if (fu >= fv) {
        return h(i, j) + fu * (h(i + 1, j) - h(i, j)) + fv * (h(i + 1, j + 1) - h(i + 1, j));
    }
    return h(i, j) + fv * (h(i, j + 1) - h(i, j)) + fu * (h(i + 1, j + 1) - h(i, j + 1));
}

// The index of the node that contains (x, z), or -1.
int nodeAt(const std::vector<TreeTerrainNode>& nodes, float x, float z) {
    for (size_t k = 0; k < nodes.size(); ++k) {
        const TreeTerrainNode& n = nodes[k];
        if (x >= n.x && x < n.x + n.z && z >= n.y && z < n.y + n.z) {
            return int(k);
        }
    }
    return -1;
}

} // namespace

class TreeTerrainRenderingTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
    }
    void TearDown() override { frontend.reset(); }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
};

TEST_F(TreeTerrainRenderingTest, verticesLieOnTheTerrain) {
    const BonsaiScene scene = sceneAt(1.0f);
    TerrainScene t(*vc, scene, {320, 180});
    const Eye e = groundEye(scene);
    t.render(e);
    const std::vector<TreeTerrainNode> nodes = t.terrain->readNodes();
    const std::vector<glm::vec4> vertices = t.terrain->readVertices();
    const TreeTerrainConfig& c = t.terrain->getConfig();
    const int quads = int(c.lod.ranges.z), row = quads + 1;
    int checked = 0;
    for (size_t k = 0; k < nodes.size(); ++k) {
        const TreeTerrainNode& n = nodes[k];
        for (int j = 0; j <= quads; j += 4) {
            for (int i = 0; i <= quads; i += 4) {
                const float x = n.x + n.z * float(i) / quads, z = n.y + n.z * float(j) / quads;
                const glm::vec4 v = vertices[k * row * row + j * row + i];
                const float h = tree::treeTerrainHeight(c.terrain, x, z);
                if (tree::treeTerrainMorph(c.lod, uint32_t(n.w), glm::distance(e.eye, glm::vec3(x, h, z))) > 0.0f) {
                    continue;
                }
                // Within float rounding of positions far from the origin of the grid.
                EXPECT_NEAR(v.w, h, 2e-5f);
                const float spacing = n.z / quads;
                const glm::vec3 normal =
                    glm::normalize(glm::vec3(tree::treeTerrainHeight(c.terrain, x - spacing, z) -
                                                 tree::treeTerrainHeight(c.terrain, x + spacing, z),
                                             2.0f * spacing,
                                             tree::treeTerrainHeight(c.terrain, x, z - spacing) -
                                                 tree::treeTerrainHeight(c.terrain, x, z + spacing)));
                EXPECT_GT(glm::dot(glm::vec3(v), normal), 0.999f);
                ++checked;
            }
        }
    }
    EXPECT_GT(checked, 500);
}

TEST_F(TreeTerrainRenderingTest, sharedEdgesMatch) {
    const BonsaiScene scene = sceneAt(1.0f);
    TerrainScene t(*vc, scene, {320, 180});
    const int quads = int(t.terrain->getConfig().lod.ranges.z);
    for (const Eye& e : {outsideEye(), groundEye(scene), groundEye(scene, {-0.6f, 0.3f}, {0.8f, -0.2f})}) {
        t.render(e);
        const std::vector<TreeTerrainNode> nodes = t.terrain->readNodes();
        const std::vector<glm::vec4> vertices = t.terrain->readVertices();
        int edges = 0, betweenLevels = 0;
        float worst = 0.0f;
        for (size_t a = 0; a < nodes.size(); ++a) {
            for (size_t b = a + 1; b < nodes.size(); ++b) {
                const TreeTerrainNode& p = nodes[a];
                const TreeTerrainNode& q = nodes[b];
                const float eps = 1e-6f;
                const float x0 = std::max(p.x, q.x), x1 = std::min(p.x + p.z, q.x + q.z);
                const float z0 = std::max(p.y, q.y), z1 = std::min(p.y + p.z, q.y + q.z);
                const bool alongZ = std::abs(x1 - x0) < eps && z1 - z0 > eps;
                const bool alongX = std::abs(z1 - z0) < eps && x1 - x0 > eps;
                if (!alongZ && !alongX) {
                    continue;
                }
                ++edges;
                betweenLevels += p.w != q.w ? 1 : 0;
                // At every vertex of the finer node along the shared edge, and between them.
                const float spacing = std::min(p.z, q.z) / quads;
                const float length = alongZ ? z1 - z0 : x1 - x0;
                for (float d = 0.0f; d <= length + 0.25f * spacing; d += 0.5f * spacing) {
                    const float x = alongZ ? x0 : x0 + std::min(d, length);
                    const float z = alongZ ? z0 + std::min(d, length) : z0;
                    const float ha = surfaceHeight(nodes, vertices, quads, a, x, z);
                    const float hb = surfaceHeight(nodes, vertices, quads, b, x, z);
                    worst = std::max(worst, std::abs(ha - hb));
                }
            }
        }
        // Float rounding of the interpolation only: 1e-5 units are 3 cm at scale 1.
        EXPECT_LT(worst, 1e-5f);
        EXPECT_GT(edges, 20);
        // From outside the pot all nodes have one level; on the ground the levels change around the camera.
        if (e.eye.y < 2.0f) {
            EXPECT_GT(betweenLevels, 5);
        }
    }
}

TEST_F(TreeTerrainRenderingTest, geomorphIsContinuous) {
    const BonsaiScene scene = sceneAt(1.0f);
    // The largest change per step of the surface at fixed points while the camera walks along the ground.
    auto largestStep = [&](float morphStart) {
        TreeTerrainLodConfig lod;
        lod.morphStart = morphStart;
        TerrainScene t(*vc, scene, {320, 180}, lod);
        const int quads = int(lod.quads);
        Eye e = groundEye(scene);
        const glm::vec3 forward = glm::normalize(e.target - e.eye);
        std::vector<glm::vec2> points;
        for (int k = 0; k < 400; ++k) {
            const float d = 0.002f + 0.5f * std::pow(float(k) / 400.0f, 2.0f);
            const float side = 0.3f * d * std::sin(1.7f * float(k));
            points.emplace_back(e.eye.x + d * forward.x - side * forward.z, e.eye.z + d * forward.z + side * forward.x);
        }
        std::vector<float> previous;
        float largest = 0.0f;
        for (int step = 0; step < 60; ++step) {
            Eye moved = e;
            moved.eye += forward * (0.0004f * float(step));
            moved.target += forward * (0.0004f * float(step));
            t.render(moved);
            const std::vector<TreeTerrainNode> nodes = t.terrain->readNodes();
            const std::vector<glm::vec4> vertices = t.terrain->readVertices();
            std::vector<float> heights;
            for (const glm::vec2& p : points) {
                const int k = nodeAt(nodes, p.x, p.y);
                heights.push_back(k < 0 ? NAN : surfaceHeight(nodes, vertices, quads, size_t(k), p.x, p.y));
            }
            if (!previous.empty()) {
                for (size_t i = 0; i < points.size(); ++i) {
                    if (!std::isnan(heights[i]) && !std::isnan(previous[i])) {
                        largest = std::max(largest, std::abs(heights[i] - previous[i]));
                    }
                }
            }
            previous = heights;
        }
        return largest;
    };
    const float morphing = largestStep(0.66f);
    const float popping = largestStep(0.9999f);
    std::printf("largest step: morphing %g, popping %g\n", morphing, popping);
    EXPECT_GT(popping, 0.0f);
    EXPECT_LT(morphing, 0.3f * popping);
}

TEST_F(TreeTerrainRenderingTest, terrainRendersWithoutHoles) {
    const BonsaiScene scene = sceneAt(1.0f);
    const VkExtent2D extent{640, 360};
    TerrainScene t(*vc, scene, extent);
    int view = 0;
    // Looking along the pot's length, down into the landscape, with the far rim and sky above it.
    const float floor = scene.terrain.soil.x, relief = scene.terrain.relief.x;
    for (const Eye& e : {Eye{{1.25f, floor + 1.1f * relief, 0.05f}, {-1.3f, floor + 0.3f * relief, -0.1f}},
                         Eye{{-1.25f, floor + 1.1f * relief, -0.1f}, {1.3f, floor + 0.3f * relief, 0.15f}}}) {
        const std::vector<uint8_t> image = t.render(e);
        writePpm("terrain_ground_" + std::to_string(view++), image, extent);
        size_t holes = 0, covered = 0;
        for (uint32_t x = 0; x < extent.width; ++x) {
            // From the topmost covered pixel of the column down, everything is terrain.
            uint32_t top = extent.height;
            for (uint32_t y = 0; y < extent.height; ++y) {
                if (tree_test::covered(image, extent, x, y)) {
                    top = y;
                    break;
                }
            }
            for (uint32_t y = top; y < extent.height; ++y) {
                const bool c = tree_test::covered(image, extent, x, y);
                holes += c ? 0 : 1;
                covered += c ? 1 : 0;
            }
        }
        EXPECT_EQ(holes, 0u);
        EXPECT_GT(covered, extent.width * extent.height / 4);
        // Some sky, so the silhouette is part of the image.
        EXPECT_LT(covered, extent.width * extent.height * 19 / 20);
    }
}

TEST_F(TreeTerrainRenderingTest, smallScalesRenderAsBefore) {
    const VkExtent2D extent{480, 270};
    for (float s : {0.0f, BonsaiScale::kForest}) {
        const BonsaiScene scene = sceneAt(s);
        std::vector<uint8_t> images[2];
        for (int withTerrain = 0; withTerrain < 2; ++withTerrain) {
            auto target = std::make_shared<OffscreenTarget>(*vc, extent, 1);
            BonsaiRenderConfig config;
            config.extent = extent;
            config.terrainLod = withTerrain == 1;
            BonsaiRenderer renderer(*vc, scene, target, config);
            ComputeGraph graph(*vc, 1);
            graph.compileFrom(renderer.getOutput());
            renderer.initialize(1);
            const Eye e = outsideEye();
            setCamera(renderer.getCamera()->ubo, e.eye, e.target, 40.0f, float(extent.width) / float(extent.height));
            renderer.getCamera()->update(0);
            graph.submitAndWait(vc->getGraphicsQueue(), 0);
            images[withTerrain] = readImage(*vc, target->getImage(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, extent);
            writePpm(std::string(withTerrain ? "terrain" : "soil_mesh") + "_s" + std::to_string(int(s * 100.0f)),
                     images[withTerrain], extent);
        }
        double squared = 0.0;
        for (size_t i = 0; i < images[0].size(); i += 4) {
            for (int c = 0; c < 3; ++c) {
                const double d = double(images[0][i + c]) - double(images[1][i + c]);
                squared += d * d;
            }
        }
        const double mse = squared / double(extent.width * extent.height * 3);
        const double psnr = 10.0 * std::log10(255.0 * 255.0 / std::max(mse, 1e-12));
        std::printf("scale %.2f: PSNR %.1f dB\n", s, psnr);
        EXPECT_GT(psnr, 40.0) << s;
    }
}

namespace {

void setSeason(HostValues<TreeRenderSettings>& settings, float season) {
    TreeRenderSettings values = settings.values().at(0);
    values.season = glm::vec4(season, tree::treeSnowCover(season), 0.0f, 0.0f);
    settings.set(0, values);
}

} // namespace

TEST_F(TreeTerrainRenderingTest, materialsFollowTheLandCover) {
    const BonsaiScene scene = sceneAt(1.0f);
    const VkExtent2D extent{800, 480};
    TerrainScene t(*vc, scene, extent);
    setSeason(*t.settings, 2.0f);
    const TopView top;
    const std::vector<uint8_t> image = t.render(top.eye, top.fov);
    writePpm("terrain_top_summer", image, extent);
    // Mean colour per class over pixels whose ground is dominated by one class, also a little around them.
    glm::vec3 sum[5] = {};
    int count[5] = {};
    for (uint32_t y = 2; y < extent.height; y += 3) {
        for (uint32_t x = 2; x < extent.width; x += 3) {
            if (!tree_test::covered(image, extent, x, y)) {
                continue;
            }
            const glm::vec3 p = top.ground(scene, extent, x, y);
            int pure = -2;
            for (int d = 0; d < 5 && pure != -1; ++d) {
                const glm::vec2 o = d == 0 ? glm::vec2(0.0f) : 0.01f * glm::vec2(d % 2 ? 1 : -1, d < 3 ? 1 : -1);
                const TreeLandCover c = tree::treeLandCover(scene.terrain, p.x + o.x, p.z + o.y);
                const float w[5] = {c.vegetation.x, c.vegetation.y, c.vegetation.z, c.vegetation.w, c.bare.x};
                const int k = int(std::max_element(w, w + 5) - w);
                pure = (w[k] > 0.55f && (pure == -2 || pure == k)) ? k : -1;
            }
            if (pure >= 0) {
                sum[pure] += pixel(image, extent, x, y);
                ++count[pure];
            }
        }
    }
    glm::vec3 mean[5];
    const char* names[5] = {"forest", "meadow", "field", "scrub", "rock"};
    for (int k = 0; k < 5; ++k) {
        ASSERT_GT(count[k], 20) << names[k];
        mean[k] = sum[k] / float(count[k]);
        std::printf("%s: %d pixels, mean %.3f %.3f %.3f, saturation %.2f\n", names[k], count[k], mean[k].x, mean[k].y,
                    mean[k].z, saturation(mean[k]));
    }
    // Rock is grey, the vegetation green.
    EXPECT_LT(saturation(mean[4]), 0.25f);
    EXPECT_GT(saturation(mean[0]), saturation(mean[4]) + 0.15f);
    EXPECT_GT(saturation(mean[1]), saturation(mean[4]) + 0.15f);
    EXPECT_GT(mean[1].y, mean[1].x);
    EXPECT_GT(mean[0].y, mean[0].z);
    // Ripe fields are yellower than meadows.
    EXPECT_GT(mean[2].x / mean[2].y, mean[1].x / mean[1].y + 0.1f);
    // Rock is brighter than the forest floor.
    EXPECT_GT(mean[4].x + mean[4].y + mean[4].z, 1.3f * (mean[0].x + mean[0].y + mean[0].z));
}

TEST_F(TreeTerrainRenderingTest, snowLiesAboveTheSnowLine) {
    const BonsaiScene scene = sceneAt(1.0f);
    const VkExtent2D extent{800, 480};
    TerrainScene t(*vc, scene, extent);
    const TopView top;
    auto whiteShare = [&](float season, bool high) {
        setSeason(*t.settings, season);
        const std::vector<uint8_t> image = t.render(top.eye, top.fov);
        writePpm("terrain_top_season" + std::to_string(int(season * 10.0f)), image, extent);
        const float line = tree::treeSnowLine(season);
        int white = 0, total = 0;
        for (uint32_t y = 2; y < extent.height; y += 3) {
            for (uint32_t x = 2; x < extent.width; x += 3) {
                if (!tree_test::covered(image, extent, x, y)) {
                    continue;
                }
                const glm::vec3 p = top.ground(scene, extent, x, y);
                const float altitude = tree::treeTerrainAltitude(scene.terrain, p);
                const float e = 0.003f;
                const glm::vec3 n =
                    glm::normalize(glm::vec3(scene.soilHeight(p.x - e, p.z) - scene.soilHeight(p.x + e, p.z), 2.0f * e,
                                             scene.soilHeight(p.x, p.z - e) - scene.soilHeight(p.x, p.z + e)));
                // Ground clearly above the line and gentle, or clearly below it.
                const bool isHigh = altitude > line + 160.0f && n.y > 0.8f;
                const bool isLow = altitude < line - 150.0f;
                if (high ? !isHigh : !isLow) {
                    continue;
                }
                const glm::vec3 c = pixel(image, extent, x, y);
                // Snow's albedo of 0.8 maps above 0.85 in sun and shade; rock's below 0.75.
                white += (saturation(c) < 0.12f && std::min(c.x, std::min(c.y, c.z)) > 0.8f) ? 1 : 0;
                ++total;
            }
        }
        EXPECT_GT(total, 20) << season << " " << high;
        return total > 0 ? float(white) / float(total) : 0.0f;
    };
    const float fallHigh = whiteShare(3.0f, true);
    const float fallLow = whiteShare(3.0f, false);
    std::printf("fall: white above the line %.2f, below %.2f\n", fallHigh, fallLow);
    EXPECT_GT(fallHigh, 0.7f);
    EXPECT_LT(fallLow, 0.05f);
    const float winterLow = whiteShare(0.2f, false);
    std::printf("winter: white below the fall line %.2f\n", winterLow);
    EXPECT_GT(winterLow, 0.5f);
}

TEST_F(TreeTerrainRenderingTest, distantDetailDoesNotAlias) {
    const BonsaiScene scene = sceneAt(1.0f);
    const VkExtent2D extent{480, 270};
    const VkExtent2D large{960, 540};
    TerrainScene native(*vc, scene, extent);
    TerrainScene fine(*vc, scene, large);
    // Closer than the example's view, so that more of the image is terrain.
    const Eye e{{0.6f, 1.4f, 1.9f}, {0.0f, 0.5f, 0.0f}};
    const std::vector<uint8_t> a = native.render(e);
    const std::vector<uint8_t> b = fine.render(e);
    // b filtered down: the mean of 2 x 2 pixels; terrain only where all four are.
    auto variation = [&](auto at, auto inside) {
        double sum = 0.0;
        int n = 0;
        for (uint32_t y = 0; y < extent.height; ++y) {
            for (uint32_t x = 0; x + 1 < extent.width; ++x) {
                if (inside(x, y) && inside(x + 1, y)) {
                    const glm::vec3 d = at(x, y) - at(x + 1, y);
                    sum += std::abs(d.x) + std::abs(d.y) + std::abs(d.z);
                    ++n;
                }
            }
        }
        EXPECT_GT(n, int(extent.width * extent.height / 4));
        return sum / n;
    };
    const double native1 = variation([&](uint32_t x, uint32_t y) { return pixel(a, extent, x, y); },
                                     [&](uint32_t x, uint32_t y) { return tree_test::covered(a, extent, x, y); });
    auto down = [&](uint32_t x, uint32_t y) {
        return 0.25f * (pixel(b, large, 2 * x, 2 * y) + pixel(b, large, 2 * x + 1, 2 * y) +
                        pixel(b, large, 2 * x, 2 * y + 1) + pixel(b, large, 2 * x + 1, 2 * y + 1));
    };
    auto downInside = [&](uint32_t x, uint32_t y) {
        return tree_test::covered(b, large, 2 * x, 2 * y) && tree_test::covered(b, large, 2 * x + 1, 2 * y) &&
               tree_test::covered(b, large, 2 * x, 2 * y + 1) && tree_test::covered(b, large, 2 * x + 1, 2 * y + 1);
    };
    const double filtered = variation(down, downInside);
    writePpm("terrain_alias_native", a, extent);
    std::printf("neighbour variation: native %.4f, filtered %.4f\n", native1, filtered);
    EXPECT_LT(native1, 1.3 * filtered);
}
