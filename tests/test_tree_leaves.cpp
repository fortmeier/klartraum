// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - outlineHitsItsPoints: treeLeafOutline passes through base, widest point and tip, leaves the base at alpha
 *   and joins its curves smoothly
 * - halfAreaMatchesTheOutline: treeLeafHalfArea equals the area of the densely sampled outline, and of the
 *   polygon of the curve ends when flattened
 * - curveConvexityAndPolyline: convex curves contribute their chord's middle to the polyline, concave ones
 *   their control point; flattening moves control points onto the chord
 * - loopBlinnSidesOfTheCurve: the implicit form is zero on the curve, negative on the chord side, positive
 *   on the control side
 * - leafColoursVaryAroundTheBase: treeLeafAlbedo varies per leaf in hue and brightness, a little per tree, and
 *   averages near the base colour; undersides are paler and greyer
 * - leafLodIsContinuous: flatten goes from 0 to 1 between leafLod.x and .y, distant leaves are kept with the
 *   kept fraction and grow by 1/sqrt of it, and leaves near the threshold shrink smoothly
 * - singleLeafMatchesOutline: a rendered leaf facing the camera covers the pixels of its analytic outline
 *   (IoU > 0.95), for a convex and a concave outline
 * - threeLobesFanOut: a three-lobed leaf crosses a scanline below the side lobes' tips three times
 * - leafLodKeepsTheArea: the leaf's pixel area times the squared distance stays constant across the
 *   simplification band
 * - presetTreesWithLeaves: stems and leaves of the four presets, written to build/TestingOutput/tree/
 **/

#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include "klartraum/headless_frontend.hpp"

#include "tree_test_helpers.hpp"

using namespace klartraum;
using namespace klartraum::tree;
using namespace klartraum::tree_test;

namespace {

glm::vec2 bezier(glm::vec2 a, glm::vec2 b, glm::vec2 c, float t) {
    return (1.0f - t) * (1.0f - t) * a + 2.0f * (1.0f - t) * t * b + t * t * c;
}

// The closed outline of a lobe, densely sampled: the right half up, the left half down.
std::vector<glm::vec2> outlinePolygon(const TreeLeafOutline& o) {
    std::vector<glm::vec2> right;
    for (int c = 0; c < 4; ++c) {
        for (int i = 0; i < 64; ++i) {
            right.push_back(bezier(o.v[2 * c], o.v[2 * c + 1], o.v[2 * c + 2], float(i) / 64.0f));
        }
    }
    right.push_back(o.v[8]);
    std::vector<glm::vec2> polygon = right;
    for (auto it = right.rbegin(); it != right.rend(); ++it) {
        polygon.push_back(glm::vec2(-it->x, it->y));
    }
    return polygon;
}

bool insidePolygon(const std::vector<glm::vec2>& polygon, glm::vec2 p) {
    bool inside = false;
    for (size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++) {
        const glm::vec2 a = polygon[i], b = polygon[j];
        if ((a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x) {
            inside = !inside;
        }
    }
    return inside;
}

// A single leaf of length `leafLength` at (0, 1, 0), on the z = 0 plane: blade along x, pointing up +y.
TreeTypeParams singleLeaf(float leafLength, glm::vec4 shape, float width, glm::vec4 lobes) {
    TreeTypeParams p = tree_presets::quakingAspen();
    p.shapeBase = glm::vec4(3.0f, 0.0f, 2.0f, 0.0f);
    p.ratios = glm::vec4(1.0f, 0.001f, 1.0f, 0.0f);
    p.leafCount = glm::vec4(1.0f, leafLength, width, 0.0f);
    p.leafAngles = glm::vec4(0.0f);
    p.leafShape = shape;
    p.leafLobes = lobes;
    p.leafColor = glm::vec4(0.3f, 0.6f, 0.2f, 0.0f);
    p.nLength.x = 1.0f;
    p.nLengthV.x = 0.0f;
    p.nCurveRes.x = 1.0f;
    p.nCurve.x = 0.0f;
    p.nCurveV.x = 0.0f;
    return p;
}

// Pixel (x, y) as a point on the z = 0 plane.
glm::vec3 pixelOnPlane(const CameraMVP& ubo, VkExtent2D extent, uint32_t x, uint32_t y) {
    const glm::mat4 inverse = glm::inverse(ubo.proj * ubo.view);
    const float nx = 2.0f * (float(x) + 0.5f) / float(extent.width) - 1.0f;
    const float ny = 2.0f * (float(y) + 0.5f) / float(extent.height) - 1.0f;
    glm::vec4 a = inverse * glm::vec4(nx, ny, 0.0f, 1.0f);
    glm::vec4 b = inverse * glm::vec4(nx, ny, 0.5f, 1.0f);
    const glm::vec3 pa = glm::vec3(a) / a.w, pb = glm::vec3(b) / b.w;
    const float t = -pa.z / (pb.z - pa.z);
    return pa + t * (pb - pa);
}

} // namespace

TEST(TreeLeafShape, outlineHitsItsPoints) {
    const TreeLeafOutline o = treeLeafOutline(glm::vec4(60.0f, 10.0f, 40.0f, 0.4f), 0.3f);
    EXPECT_EQ(o.v[0], glm::vec2(0.0f, 0.0f));
    EXPECT_EQ(o.v[4], glm::vec2(0.3f, 0.4f));
    EXPECT_EQ(o.v[8], glm::vec2(0.0f, 1.0f));
    // Leaves the base at alpha against the axis.
    const glm::vec2 d = glm::normalize(o.v[1] - o.v[0]);
    EXPECT_NEAR(std::atan2(d.x, d.y), glm::radians(60.0f), 1e-5f);
    // Smooth joins: v2 and v6 lie on the line between their neighbouring control points, v4 between v3 and v5.
    EXPECT_NEAR(glm::length(o.v[2] - 0.5f * (o.v[1] + o.v[3])), 0.0f, 1e-6f);
    EXPECT_NEAR(glm::length(o.v[6] - 0.5f * (o.v[5] + o.v[7])), 0.0f, 1e-6f);
    const glm::vec2 in = glm::normalize(o.v[4] - o.v[3]), out = glm::normalize(o.v[5] - o.v[4]);
    EXPECT_NEAR(glm::dot(in, out), 1.0f, 1e-5f);
}

TEST(TreeLeafShape, halfAreaMatchesTheOutline) {
    for (const glm::vec4 shape : {glm::vec4(70.0f, 0.0f, 50.0f, 0.4f), glm::vec4(50.0f, 0.0f, -20.0f, 0.4f)}) {
        const TreeLeafOutline o = treeLeafOutline(shape, 0.4f);
        // Shoelace formula over the sampled outline of both halves.
        const std::vector<glm::vec2> polygon = outlinePolygon(o);
        double area = 0.0;
        for (size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++) {
            area += double(polygon[j].x) * polygon[i].y - double(polygon[i].x) * polygon[j].y;
        }
        EXPECT_NEAR(std::abs(area) * 0.5, 2.0 * treeLeafHalfArea(o, 0.0f), 1e-3);
        // Flattened: the polygon v0, v2, ..., v8 and the axis.
        double flat = 0.0;
        for (int i = 0; i < 8; i += 2) {
            flat += 0.5 * (o.v[i].x + o.v[i + 2].x) * (o.v[i + 2].y - o.v[i].y);
        }
        EXPECT_NEAR(flat, treeLeafHalfArea(o, 1.0f), 1e-6);
    }
}

TEST(TreeLeafShape, curveConvexityAndPolyline) {
    // A rounded leaf: all curves bulge outwards.
    const TreeLeafOutline round = treeLeafOutline(glm::vec4(70.0f, 0.0f, 50.0f, 0.4f), 0.5f);
    for (int c = 0; c < 4; ++c) {
        EXPECT_TRUE(treeLeafCurveConvex(round.v[2 * c], round.v[2 * c + 1], round.v[2 * c + 2])) << c;
        const glm::vec2 middle = 0.5f * (round.v[2 * c] + round.v[2 * c + 2]);
        EXPECT_NEAR(glm::length(treeLeafPolyline(round, 2 * c + 1, 0.0f) - middle), 0.0f, 1e-6f);
    }
    // A drawn-out tip (gamma < 0) is concave: the polyline takes the control point.
    const TreeLeafOutline pointed = treeLeafOutline(glm::vec4(50.0f, 0.0f, -20.0f, 0.4f), 0.4f);
    EXPECT_FALSE(treeLeafCurveConvex(pointed.v[6], pointed.v[7], pointed.v[8]));
    EXPECT_EQ(treeLeafPolyline(pointed, 7, 0.0f), pointed.v[7]);
    // Flattening moves control points onto the chord.
    const glm::vec2 flat = treeLeafControl(pointed, 7, 1.0f);
    EXPECT_NEAR(glm::length(flat - 0.5f * (pointed.v[6] + pointed.v[8])), 0.0f, 1e-6f);
    for (int i = 0; i <= 8; i += 2) {
        EXPECT_EQ(treeLeafPolyline(pointed, i, 0.3f), pointed.v[i]);
    }
}

TEST(TreeLeafShape, loopBlinnSidesOfTheCurve) {
    // Points of the curve u = t, v = t^2 in the coordinates (0, 0), (0.5, 0), (1, 1).
    for (float t : {0.0f, 0.25f, 0.5f, 1.0f}) {
        EXPECT_NEAR(treeLoopBlinn(glm::vec2(t, t * t)), 0.0f, 1e-6f);
    }
    EXPECT_LT(treeLoopBlinn(glm::vec2(0.5f, 0.5f)), 0.0f); // middle of the chord
    EXPECT_GT(treeLoopBlinn(glm::vec2(0.5f, 0.0f)), 0.0f); // the control point
}

TEST(TreeLeafShape, leafColoursVaryAroundTheBase) {
    const glm::vec3 base(0.08f, 0.15f, 0.035f);
    glm::vec3 sum(0.0f);
    float minLuma = 1.0f, maxLuma = 0.0f, minRatio = 10.0f, maxRatio = 0.0f;
    const int n = 4000;
    for (int i = 0; i < n; ++i) {
        const glm::vec3 c = treeLeafAlbedo(base, uint32_t(i), 7u);
        sum += c;
        const float luma = glm::dot(c, glm::vec3(0.3f, 0.59f, 0.11f));
        minLuma = std::min(minLuma, luma);
        maxLuma = std::max(maxLuma, luma);
        minRatio = std::min(minRatio, c.r / c.b);
        maxRatio = std::max(maxRatio, c.r / c.b);
    }
    const glm::vec3 mean = sum / float(n);
    EXPECT_NEAR(mean.g / base.g, 1.0f, 0.1f);
    EXPECT_GT(maxLuma / minLuma, 1.6f);   // brightness +-25 %
    EXPECT_GT(maxRatio / minRatio, 2.0f); // from yellower to bluer
    // Trees differ a little.
    EXPECT_NE(treeLeafAlbedo(base, 1u, 7u), treeLeafAlbedo(base, 1u, 8u));
    const glm::vec3 under = treeLeafUnderside(base);
    EXPECT_GT(glm::dot(under, glm::vec3(0.3f, 0.59f, 0.11f)), glm::dot(base, glm::vec3(0.3f, 0.59f, 0.11f)));
    EXPECT_LT(under.g / under.b, base.g / base.b);
}

TEST(TreeLeafShape, leafLodIsContinuous) {
    const glm::vec4 lod(24.0f, 8.0f, 4.0f, 0.25f);
    EXPECT_EQ(treeLeafLod(40.0f, lod, 0.5f), glm::vec2(0.0f, 1.0f));
    EXPECT_NEAR(treeLeafLod(16.0f, lod, 0.5f).x, 0.5f, 1e-6f);
    EXPECT_EQ(treeLeafLod(8.0f, lod, 0.5f).x, 1.0f);
    // Far away a quarter of the leaves stays, twice as large.
    EXPECT_NEAR(treeLeafLod(0.5f, lod, 0.1f).y, 2.0f, 1e-5f);
    EXPECT_EQ(treeLeafLod(0.5f, lod, 0.6f).y, 0.0f);
    // Continuous in the distance for a leaf near the threshold.
    float previous = treeLeafLod(4.0f, lod, 0.6f).y;
    for (float pixels = 4.0f; pixels > 0.5f; pixels -= 0.01f) {
        const float size = treeLeafLod(pixels, lod, 0.6f).y;
        EXPECT_LT(std::abs(size - previous), 0.05f) << pixels;
        previous = size;
    }
}

class TreeLeafTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
    }
    void TearDown() override { frontend.reset(); }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
};

TEST_F(TreeLeafTest, singleLeafMatchesOutline) {
    const VkExtent2D extent{256, 256};
    const float leafLength = 0.8f;
    struct Case {
        const char* name;
        glm::vec4 shape;
        float width;
    };
    for (const Case& c : {Case{"convex", glm::vec4(70.0f, 0.0f, 50.0f, 0.4f), 1.0f},
                          Case{"concave", glm::vec4(50.0f, 0.0f, -20.0f, 0.4f), 0.8f}}) {
        SCOPED_TRACE(c.name);
        const TreeTypeParams p = singleLeaf(leafLength, c.shape, c.width, glm::vec4(1.0f, 0.0f, 1.0f, 0.0f));
        TreeScene scene(*vc, {p}, {makeInstance(0, 1)}, extent, false, true);
        const auto image = scene.render(glm::vec3(0.0f, 1.4f, 3.0f), glm::vec3(0.0f, 1.4f, 0.0f), 30.0f);
        writePpm(std::string("leaf_") + c.name, image, extent);
        ASSERT_EQ(scene.leaves->readStats()[kTreeLeafStatDrawn], 1u);

        const std::vector<glm::vec2> polygon = outlinePolygon(treeLeafOutline(c.shape, 0.5f * c.width));
        size_t both = 0, either = 0;
        for (uint32_t y = 0; y < extent.height; ++y) {
            for (uint32_t x = 0; x < extent.width; ++x) {
                const glm::vec3 w = pixelOnPlane(scene.camera->ubo, extent, x, y);
                const bool expected = insidePolygon(polygon, glm::vec2(w.x, w.y - 1.0f) / leafLength);
                const bool drawn = covered(image, extent, x, y);
                both += expected && drawn ? 1 : 0;
                either += expected || drawn ? 1 : 0;
            }
        }
        ASSERT_GT(either, 2000u);
        EXPECT_GT(double(both) / double(either), 0.95);
    }
}

TEST_F(TreeLeafTest, threeLobesFanOut) {
    const VkExtent2D extent{256, 256};
    const float leafLength = 0.8f;
    const TreeTypeParams p =
        singleLeaf(leafLength, glm::vec4(70.0f, 0.0f, 50.0f, 0.4f), 0.3f, glm::vec4(3.0f, 60.0f, 0.75f, 0.0f));
    TreeScene scene(*vc, {p}, {makeInstance(0, 1)}, extent, false, true);
    const auto image = scene.render(glm::vec3(0.0f, 1.4f, 3.0f), glm::vec3(0.0f, 1.4f, 0.0f), 30.0f);
    writePpm("leaf_three_lobes", image, extent);

    // The row of the leaf point (0, 0.3).
    const glm::vec4 clip =
        scene.camera->ubo.proj * scene.camera->ubo.view * glm::vec4(0.0f, 1.0f + 0.3f * leafLength, 0.0f, 1.0f);
    const uint32_t row = uint32_t((clip.y / clip.w * 0.5f + 0.5f) * float(extent.height));
    int spans = 0;
    bool inside = false;
    for (uint32_t x = 0; x < extent.width; ++x) {
        const bool c = covered(image, extent, x, row);
        spans += c && !inside ? 1 : 0;
        inside = c;
    }
    EXPECT_EQ(spans, 3);
}

TEST_F(TreeLeafTest, leafLodKeepsTheArea) {
    const VkExtent2D extent{256, 256};
    const float leafLength = 0.2f;
    const TreeTypeParams p =
        singleLeaf(leafLength, glm::vec4(70.0f, 0.0f, 50.0f, 0.4f), 0.6f, glm::vec4(1.0f, 0.0f, 1.0f, 0.0f));
    TreeScene scene(*vc, {p}, {makeInstance(0, 1)}, extent, false, true);
    // At 256 pixels and 30 degrees, the leaf is 24 pixels long at distance 4 and 8 pixels at 12.
    double reference = 0.0;
    for (float distance = 2.0f; distance <= 14.0f; distance += 0.5f) {
        const glm::vec3 center(0.0f, 1.0f + 0.5f * leafLength, 0.0f);
        const auto image = scene.render(center + glm::vec3(0.0f, 0.0f, distance), center, 30.0f);
        const double area = double(coveredCount(image, extent)) * distance * distance;
        if (reference == 0.0) {
            reference = area;
        }
        EXPECT_NEAR(area / reference, 1.0, 0.15) << distance;
    }
}

TEST_F(TreeLeafTest, presetTreesWithLeaves) {
    const std::vector<TreeTypeParams> types = {tree_presets::quakingAspen(), tree_presets::blackTupelo(),
                                               tree_presets::sassafras(), tree_presets::pine()};
    const char* names[] = {"aspen", "tupelo", "sassafras", "pine"};
    const VkExtent2D extent{512, 512};
    for (uint32_t type = 0; type < types.size(); ++type) {
        const float scale = 1.0f / types[type].shapeBase.z;
        TreeScene scene(*vc, types, {makeInstance(type, 5, glm::vec3(0.0f), scale)}, extent, true, true);
        const auto image = scene.render(glm::vec3(0.0f, 0.5f, 1.8f), glm::vec3(0.0f, 0.5f, 0.0f));
        writePpm(std::string("tree_") + names[type], image, extent);
        const std::vector<uint32_t> stats = scene.leaves->readStats();
        std::cout << names[type] << ": " << stats[kTreeLeafStatDrawn] << " leaves drawn, "
                  << stats[kTreeLeafStatTriangles] << " leaf triangles, " << stats[kTreeLeafStatCulled] << " culled"
                  << std::endl;
        EXPECT_GT(stats[kTreeLeafStatDrawn], 100u) << names[type];
        EXPECT_GT(coveredCount(image, extent), 5000u) << names[type];
    }
}
