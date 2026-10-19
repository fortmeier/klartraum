// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - tessellationSpacingIsContinuous: treeTessSpacing has odd spacing, ends at 0 and 1, and its vertices move
 *   continuously as the factor passes an integer
 * - openingAngleFollowsThePaper: treeOpeningAngle is pi/2 for cylinders and grows when looking at a tapering
 *   stem from its tip side
 * - stemBucketsFitTheirFactors: treeStemBucket picks the smallest bucket with enough quads, else the largest
 * - straightCylinderHasProjectedWidth: a straight untapered trunk seen from the side is as wide in pixels as
 *   its projected diameter, without holes
 * - curvedTrunkHasNoCracks: every image row across the lower half of a curved, tapering trunk of 12 segments
 *   is one gap-free span, coarse or fine tessellation
 * - coarseTessellationKeepsTheSilhouette: 8 pixels per triangle edge covers nearly the same pixels as 1
 * - trianglesDecreaseWithDistance: the tessellation of a preset tree has fewer triangles the farther away it is
 * - culledOutsideTheView: a camera looking away draws no segment and leaves the image black
 * - presetTreeStems: the stems of the four presets, written to build/TestingOutput/tree/
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

// A trunk only: one level, no leaves, length `length` and base radius `radius`.
TreeTypeParams trunkOnly(float length, float radius) {
    TreeTypeParams p = tree_presets::quakingAspen();
    p.shapeBase = glm::vec4(3.0f, 0.0f, length, 0.0f);
    p.ratios = glm::vec4(1.0f, radius / length, 1.0f, 0.0f);
    p.leafCount = glm::vec4(0.0f, 0.1f, 1.0f, 0.0f);
    p.nLength.x = 1.0f;
    p.nLengthV.x = 0.0f;
    p.nCurveRes.x = 1.0f;
    p.nCurve.x = 0.0f;
    p.nCurveV.x = 0.0f;
    p.nSegSplits.x = 0.0f;
    p.barkColor = glm::vec4(0.6f, 0.5f, 0.4f, 0.0f);
    return p;
}

using StemScene = TreeScene;

// Covered pixels in row y: their number, and whether they form one span.
struct RowSpan {
    uint32_t count = 0;
    bool contiguous = true;
};

RowSpan rowSpan(const std::vector<uint8_t>& image, VkExtent2D extent, uint32_t y) {
    RowSpan span;
    int first = -1, last = -1;
    for (uint32_t x = 0; x < extent.width; ++x) {
        if (covered(image, extent, x, y)) {
            span.count++;
            first = first < 0 ? int(x) : first;
            last = int(x);
        }
    }
    span.contiguous = span.count == 0 || uint32_t(last - first + 1) == span.count;
    return span;
}

} // namespace

TEST(TreeTessellation, tessellationSpacingIsContinuous) {
    // Integer factors: even spacing.
    for (uint32_t i = 0; i <= 4; ++i) {
        EXPECT_NEAR(treeTessSpacing(i, 4.0f, 8u), float(i) / 4.0f, 1e-6f);
    }
    // Vertices past the last coincide with it; factors are clamped to [1, maxSegments].
    EXPECT_EQ(treeTessSpacing(7u, 4.0f, 8u), 1.0f);
    EXPECT_EQ(treeTessSpacing(1u, 0.2f, 8u), 1.0f);
    EXPECT_NEAR(treeTessSpacing(1u, 20.0f, 8u), 1.0f / 8.0f, 1e-6f);
    // Odd spacing: all segments 1 / F except the middle one.
    const float f = 4.5f;
    EXPECT_NEAR(treeTessSpacing(1u, f, 8u), 1.0f / f, 1e-6f);
    EXPECT_NEAR(treeTessSpacing(4u, f, 8u), 1.0f - 1.0f / f, 1e-6f);
    EXPECT_NEAR(treeTessSpacing(3u, f, 8u) - treeTessSpacing(2u, f, 8u), 1.0f - 4.0f / f, 1e-6f);
    // Continuity across the integer: at F slightly above 4 two vertices nearly coincide and the
    // others are where they are at F = 4.
    for (uint32_t i = 0; i <= 5; ++i) {
        const float above = treeTessSpacing(i, 4.0001f, 8u);
        const float at = treeTessSpacing(i > 2 ? i - 1 : i, 4.0f, 8u);
        EXPECT_NEAR(above, at, 1e-3f) << i;
    }
}

TEST(TreeTessellation, openingAngleFollowsThePaper) {
    EXPECT_NEAR(treeOpeningAngle(0.0f, 0.3f), 0.5f * TREE_PI, 1e-6f);
    EXPECT_NEAR(treeOpeningAngle(-0.1f, 0.0f), 0.5f * TREE_PI, 1e-6f);
    // A tapering stem seen from the tip side shows more than half its circumference.
    EXPECT_GT(treeOpeningAngle(-0.1f, 0.5f), 0.5f * TREE_PI);
    EXPECT_LT(treeOpeningAngle(-0.1f, -0.5f), 0.5f * TREE_PI);
    // theta = acos(dr/dz * x / sqrt(1 - x^2)).
    const float x = 0.6f;
    EXPECT_NEAR(treeOpeningAngle(-0.2f, x), std::acos(-0.2f * x / std::sqrt(1.0f - x * x)), 1e-6f);
    // Looking along the stem: clamped, not NaN.
    EXPECT_NEAR(treeOpeningAngle(-0.2f, 1.0f), TREE_PI, 1e-5f);
}

TEST(TreeTessellation, stemBucketsFitTheirFactors) {
    EXPECT_EQ(treeStemBucket(1.0f, 1.0f), 0u);
    EXPECT_EQ(treeStemBucket(1.5f, 1.0f), 1u);
    EXPECT_EQ(treeStemBucket(3.0f, 2.0f), 1u);
    EXPECT_EQ(treeStemBucket(3.0f, 3.0f), 2u);
    EXPECT_EQ(treeStemBucket(16.0f, 1.0f), 3u);
    EXPECT_EQ(treeStemBucket(17.0f, 1.0f), 4u);
    EXPECT_EQ(treeStemBucket(100.0f, 100.0f), TREE_STEM_BUCKETS - 1u);
}

class TreeRenderingTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
    }
    void TearDown() override { frontend.reset(); }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
};

TEST_F(TreeRenderingTest, straightCylinderHasProjectedWidth) {
    TreeTypeParams p = trunkOnly(2.0f, 0.1f);
    p.nTaper.x = 0.0f;
    p.nCurveRes.x = 4.0f;
    const VkExtent2D extent{256, 256};
    StemScene scene(*vc, {p}, {makeInstance(0, 1)}, extent);
    const auto image = scene.render(glm::vec3(0.0f, 1.0f, 4.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    writePpm("straight_cylinder", image, extent);

    const float focal = 0.5f * float(extent.height) / std::tan(glm::radians(20.0f));
    const float expected = 2.0f * 0.1f * focal / 4.0f;
    for (uint32_t y : {64u, 100u, 128u, 160u}) {
        const RowSpan span = rowSpan(image, extent, y);
        EXPECT_NEAR(float(span.count), expected, 1.5f) << "row " << y;
        EXPECT_TRUE(span.contiguous) << "row " << y;
    }
}

TEST_F(TreeRenderingTest, curvedTrunkHasNoCracks) {
    TreeTypeParams p = trunkOnly(2.0f, 0.12f);
    p.nCurveRes.x = 12.0f;
    p.nCurve.x = 70.0f;
    const VkExtent2D extent{256, 256};
    for (float pixelsPerEdge : {1.0f, 8.0f}) {
        SCOPED_TRACE(pixelsPerEdge);
        StemScene scene(*vc, {p}, {makeInstance(0, 1)}, extent);
        TreeRenderSettings settings = defaultTreeRenderSettings(extent);
        settings.viewport.z = pixelsPerEdge;
        scene.settings->set(0, settings);
        // The trunk bends in the plane seen from +x.
        const auto image = scene.render(glm::vec3(4.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.9f, 0.0f));
        writePpm("curved_trunk_" + std::to_string(int(pixelsPerEdge)), image, extent);

        int top = -1, bottom = -1;
        for (uint32_t y = 0; y < extent.height; ++y) {
            if (rowSpan(image, extent, y).count > 0) {
                top = top < 0 ? int(y) : top;
                bottom = int(y);
            }
        }
        ASSERT_GT(bottom - top, 100);
        // Rows across the lower half, where the trunk is still steep; it bends to nearly horizontal above.
        const int lowest = bottom - int(0.5f * float(bottom - top));
        // The lowest rows only touch the open end of the base ring.
        for (int y = bottom - 3; y >= lowest; --y) {
            const RowSpan span = rowSpan(image, extent, uint32_t(y));
            EXPECT_GE(span.count, 3u) << "row " << y;
            EXPECT_TRUE(span.contiguous) << "row " << y;
        }
    }
}

TEST_F(TreeRenderingTest, coarseTessellationKeepsTheSilhouette) {
    TreeTypeParams p = trunkOnly(2.0f, 0.15f);
    p.nCurveRes.x = 6.0f;
    p.nCurve.x = 50.0f;
    const VkExtent2D extent{256, 256};
    StemScene scene(*vc, {p}, {makeInstance(0, 1)}, extent);
    std::vector<std::vector<uint8_t>> images;
    for (float pixelsPerEdge : {1.0f, 8.0f}) {
        TreeRenderSettings settings = defaultTreeRenderSettings(extent);
        settings.viewport.z = pixelsPerEdge;
        scene.settings->set(0, settings);
        images.push_back(scene.render(glm::vec3(3.0f, 1.2f, 1.0f), glm::vec3(0.0f, 0.9f, 0.0f)));
    }
    size_t both = 0, either = 0;
    for (uint32_t y = 0; y < extent.height; ++y) {
        for (uint32_t x = 0; x < extent.width; ++x) {
            const bool a = covered(images[0], extent, x, y);
            const bool b = covered(images[1], extent, x, y);
            both += a && b ? 1 : 0;
            either += a || b ? 1 : 0;
        }
    }
    ASSERT_GT(either, 1000u);
    EXPECT_GT(double(both) / double(either), 0.95);
}

TEST_F(TreeRenderingTest, trianglesDecreaseWithDistance) {
    const VkExtent2D extent{512, 512};
    StemScene scene(*vc, {tree_presets::quakingAspen()}, {makeInstance(0, 3, glm::vec3(0.0f), 0.1f)}, extent);
    uint32_t previous = UINT32_MAX;
    for (float distance : {2.0f, 4.0f, 8.0f, 16.0f, 32.0f}) {
        scene.render(glm::vec3(0.0f, 0.8f, distance), glm::vec3(0.0f, 0.8f, 0.0f));
        const std::vector<uint32_t> stats = scene.stems->readStats();
        std::cout << "distance " << distance << ": " << stats[kTreeStemStatDrawn] << " segments, "
                  << stats[kTreeStemStatTriangles] << " triangles, " << stats[kTreeStemStatCulled] << " culled"
                  << std::endl;
        EXPECT_LT(stats[kTreeStemStatTriangles], previous) << distance;
        previous = stats[kTreeStemStatTriangles];
    }
    // Far away, the thin twigs are culled.
    EXPECT_GT(scene.stems->readStats()[kTreeStemStatCulled], 0u);
}

TEST_F(TreeRenderingTest, culledOutsideTheView) {
    const VkExtent2D extent{128, 128};
    StemScene scene(*vc, {tree_presets::quakingAspen()}, {makeInstance(0, 3, glm::vec3(0.0f), 0.1f)}, extent);
    const auto image = scene.render(glm::vec3(0.0f, 0.8f, 4.0f), glm::vec3(0.0f, 0.8f, 8.0f));
    EXPECT_EQ(scene.stems->readStats()[kTreeStemStatDrawn], 0u);
    EXPECT_EQ(coveredCount(image, extent), 0u);
}

TEST_F(TreeRenderingTest, presetTreeStems) {
    const std::vector<TreeTypeParams> types = {tree_presets::quakingAspen(), tree_presets::blackTupelo(),
                                               tree_presets::sassafras(), tree_presets::pine()};
    const char* names[] = {"aspen", "tupelo", "sassafras", "pine"};
    const VkExtent2D extent{512, 512};
    for (uint32_t type = 0; type < types.size(); ++type) {
        const float scale = 1.0f / types[type].shapeBase.z;
        StemScene scene(*vc, types, {makeInstance(type, 5, glm::vec3(0.0f), scale)}, extent);
        const auto image = scene.render(glm::vec3(0.0f, 0.5f, 1.8f), glm::vec3(0.0f, 0.5f, 0.0f));
        writePpm(std::string("stems_") + names[type], image, extent);
        EXPECT_GT(coveredCount(image, extent), 2000u) << names[type];
    }
}
