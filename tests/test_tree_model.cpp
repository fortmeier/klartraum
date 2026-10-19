// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - recordLayoutsMatchStd430: the record structs have the member offsets of their std430 GLSL layout
 * - hashIsDeterministic: treeHash/treeRand give fixed values, stay in [0, 1) and differ between keys
 * - quaternionsRotateAndCompose: qaxis/qrot turn +z as expected, qmul composes, qnlerp hits its ends
 * - treeUpRotationTurnsZToUp: the instance rotation from treeUpRotation() maps +z to the given up vector
 * - hermiteMatchesEndpointsAndTangents: treeHermite passes through its ends with the given tangents
 * - shapeRatioMatchesWeberPenn: the eight shape ratios at characteristic points
 * - stemRadiusTapersAndFlares: linear taper to the tip, flare only at the trunk base, slope = derivative
 * - trunkLobesAndBark: trunk lobes widen only the trunk's lowest sixth, by up to the lobe depth, and the bark
 *   relief stays in [0, 1] with furrows and ridges
 * - trunkOnlyTree: one level, no leaves: curveRes connected segments of the trunk's length and radius
 * - splitsCreateClones: segSplits = 1 doubles the clones at every ring, up to the 32 threads of a stem
 * - childCountsFollowWeberPenn: level 1 gets nBranches children, evenly spaced from the base size up
 * - leavesOnlyOnLastLevel: leaves come from the last stem level only, with the leaf scale applied
 * - generationIsDeterministicPerSeed: equal seeds give equal records, different seeds different trees
 * - attractionUpRaisesBranchTips: positive attractionUp raises level 2 tips compared with negative
 * - presetsGenerateFullTrees: every preset gives a tree with all levels and leaves in plausible numbers
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstddef>
#include <set>

#include "klartraum/tree/tree_model.hpp"

using namespace klartraum;
using namespace klartraum::tree;

namespace {

TreeInstance makeInstance(uint32_t type, uint32_t seed, glm::vec3 pos = glm::vec3(0.0f), float scale = 1.0f) {
    TreeInstance inst{};
    inst.posScale = glm::vec4(pos, scale);
    inst.rot = treeUpRotation(glm::vec3(0.0f, 1.0f, 0.0f));
    inst.ids = glm::uvec4(type, seed, 0u, 0u);
    return inst;
}

// A trunk only: one level, straight, no leaves.
TreeTypeParams trunkOnly() {
    TreeTypeParams p = tree_presets::quakingAspen();
    p.shapeBase = glm::vec4(3.0f, 0.0f, 10.0f, 0.0f);
    p.ratios = glm::vec4(1.0f, 0.02f, 1.0f, 0.0f);
    p.leafCount = glm::vec4(0.0f, 0.1f, 1.0f, 0.0f);
    p.nCurveRes = glm::vec4(4.0f, 1.0f, 1.0f, 1.0f);
    p.nCurveV = glm::vec4(0.0f);
    p.nCurve = glm::vec4(0.0f);
    p.nSegSplits = glm::vec4(0.0f);
    return p;
}

float distance3(const glm::vec4& a, const glm::vec4& b) { return glm::length(glm::vec3(a) - glm::vec3(b)); }

} // namespace

TEST(TreeModel, recordLayoutsMatchStd430) {
    EXPECT_EQ(offsetof(TreeTypeParams, shapeBase), 0u);
    EXPECT_EQ(offsetof(TreeTypeParams, leafColorFall), 8u * 16u);
    EXPECT_EQ(offsetof(TreeTypeParams, nLength), 9u * 16u);
    EXPECT_EQ(offsetof(TreeTypeParams, nBranches), 23u * 16u);
    EXPECT_EQ(offsetof(TreeTypeParams, trunk), 24u * 16u);
    EXPECT_EQ(offsetof(StemRecord, ids), 48u);
    EXPECT_EQ(offsetof(SegmentRecord, toPosRadius), 32u);
    EXPECT_EQ(offsetof(SegmentRecord, ids), 80u);
    EXPECT_EQ(offsetof(LeafRecord, ids), 32u);
    EXPECT_EQ(offsetof(TreeInstance, ids), 32u);
    EXPECT_EQ(offsetof(TreeFrame, timeSeason), 32u);
}

TEST(TreeModel, hashIsDeterministic) {
    EXPECT_EQ(treeHash(0u), treeHash(0u));
    EXPECT_NE(treeHash(0u), treeHash(1u));
    EXPECT_NE(treeHash2(1u, 2u), treeHash2(2u, 1u));
    std::set<uint32_t> values;
    for (uint32_t key = 0; key < 1000; ++key) {
        const float r = treeRand(42u, key);
        EXPECT_GE(r, 0.0f);
        EXPECT_LT(r, 1.0f);
        values.insert(treeHash2(42u, key));
    }
    EXPECT_EQ(values.size(), 1000u);
}

TEST(TreeModel, quaternionsRotateAndCompose) {
    const glm::vec4 identity(0.0f, 0.0f, 0.0f, 1.0f);
    const glm::vec3 f = qforward(identity);
    EXPECT_NEAR(f.z, 1.0f, 1e-6f);

    // A quarter turn about x turns +z to -y.
    const glm::vec4 qx = qaxis(glm::vec3(1.0f, 0.0f, 0.0f), 0.5f * TREE_PI);
    const glm::vec3 turned = qforward(qx);
    EXPECT_NEAR(turned.x, 0.0f, 1e-6f);
    EXPECT_NEAR(turned.y, -1.0f, 1e-6f);
    EXPECT_NEAR(turned.z, 0.0f, 1e-6f);

    // Two quarter turns compose to a half turn.
    const glm::vec3 half = qforward(qmul(qx, qx));
    EXPECT_NEAR(half.z, -1.0f, 1e-6f);

    const glm::vec4 a = qnlerp(identity, qx, 0.0f);
    const glm::vec4 b = qnlerp(identity, qx, 1.0f);
    EXPECT_NEAR(glm::length(a - identity), 0.0f, 1e-6f);
    EXPECT_NEAR(glm::length(b - qx), 0.0f, 1e-6f);
}

TEST(TreeModel, treeUpRotationTurnsZToUp) {
    for (const glm::vec3 up : {glm::vec3(0, 1, 0), glm::vec3(0, -1, 0), glm::vec3(0, 0, 1), glm::vec3(0, 0, -1),
                               glm::normalize(glm::vec3(1, 2, 3))}) {
        const glm::vec3 f = qforward(treeUpRotation(up));
        EXPECT_NEAR(glm::length(f - up), 0.0f, 1e-5f);
    }
}

TEST(TreeModel, hermiteMatchesEndpointsAndTangents) {
    const glm::vec3 p0(0, 0, 0), p1(1, 2, 3), m0(0, 0, 4), m1(4, 0, 0);
    EXPECT_NEAR(glm::length(treeHermite(p0, p1, m0, m1, 0.0f) - p0), 0.0f, 1e-6f);
    EXPECT_NEAR(glm::length(treeHermite(p0, p1, m0, m1, 1.0f) - p1), 0.0f, 1e-6f);
    EXPECT_NEAR(glm::length(treeHermiteTangent(p0, p1, m0, m1, 0.0f) - m0), 0.0f, 1e-6f);
    EXPECT_NEAR(glm::length(treeHermiteTangent(p0, p1, m0, m1, 1.0f) - m1), 0.0f, 1e-6f);
    // Straight line with matching tangents: the spline is the line.
    const glm::vec3 mid = treeHermite(p0, p1, p1 - p0, p1 - p0, 0.5f);
    EXPECT_NEAR(glm::length(mid - 0.5f * p1), 0.0f, 1e-6f);
}

TEST(TreeModel, shapeRatioMatchesWeberPenn) {
    EXPECT_NEAR(treeShapeRatio(0, 0.0f), 0.2f, 1e-6f);
    EXPECT_NEAR(treeShapeRatio(0, 1.0f), 1.0f, 1e-6f);
    EXPECT_NEAR(treeShapeRatio(1, 0.5f), 1.0f, 1e-6f);
    EXPECT_NEAR(treeShapeRatio(2, 1.0f), 1.0f, 1e-6f);
    EXPECT_NEAR(treeShapeRatio(3, 0.3f), 1.0f, 1e-6f);
    EXPECT_NEAR(treeShapeRatio(4, 0.0f), 0.5f, 1e-6f);
    EXPECT_NEAR(treeShapeRatio(5, 0.7f), 1.0f, 1e-6f);
    EXPECT_NEAR(treeShapeRatio(5, 1.0f), 0.0f, 1e-6f);
    EXPECT_NEAR(treeShapeRatio(6, 1.0f), 0.2f, 1e-6f);
    EXPECT_NEAR(treeShapeRatio(7, 0.7f), 1.0f, 1e-6f);
    EXPECT_NEAR(treeShapeRatio(7, 0.0f), 0.5f, 1e-6f);
}

TEST(TreeModel, stemRadiusTapersAndFlares) {
    TreeTypeParams p = trunkOnly();
    StemRecord s{};
    s.posLength = glm::vec4(0, 0, 0, 10.0f);
    s.dims = glm::vec4(0.5f, 0, 0, 0);
    s.ids = glm::uvec4(0u, 1u, 0u, 0u);
    EXPECT_NEAR(treeStemRadius(p, s, 0.0f), 0.5f, 1e-6f);
    EXPECT_NEAR(treeStemRadius(p, s, 0.5f), 0.25f, 1e-6f);
    EXPECT_NEAR(treeStemRadius(p, s, 1.0f), 0.0f, 1e-6f);
    EXPECT_NEAR(treeStemRadiusSlope(p, s, 0.5f), -0.05f, 1e-5f);

    // Flare widens the trunk base only.
    p.ratios.w = 1.0f;
    EXPECT_NEAR(treeStemRadius(p, s, 0.0f), 0.5f * (1.0f + 0.99f), 1e-5f);
    EXPECT_NEAR(treeStemRadius(p, s, 0.5f), 0.25f, 1e-6f);
    s.ids.x = 1u << 16u; // level 1
    EXPECT_NEAR(treeStemRadius(p, s, 0.0f), 0.5f, 1e-6f);
}

TEST(TreeModel, trunkLobesAndBark) {
    TreeTypeParams p = trunkOnly();
    p.trunk = glm::vec4(5.0f, 0.4f, 20.0f, 0.0f);
    float lowest = 10.0f, highest = 0.0f;
    for (int i = 0; i < 100; ++i) {
        const float phi = 2.0f * TREE_PI * float(i) / 100.0f;
        const float f = treeTrunkLobes(p, 0, 0.0f, phi);
        lowest = std::min(lowest, f);
        highest = std::max(highest, f);
        EXPECT_EQ(treeTrunkLobes(p, 0, 0.2f, phi), 1.0f);
        EXPECT_EQ(treeTrunkLobes(p, 1, 0.0f, phi), 1.0f);
    }
    EXPECT_NEAR(lowest, 1.0f, 1e-3f);
    EXPECT_NEAR(highest, 1.4f, 1e-3f);
    p.trunk.y = 0.0f;
    EXPECT_EQ(treeTrunkLobes(p, 0, 0.0f, 1.0f), 1.0f);

    float minHeight = 1.0f, maxHeight = 0.0f;
    for (int i = 0; i < 400; ++i) {
        const float h = treeBarkHeight(glm::vec2(0.37f * float(i), 0.11f * float(i)));
        minHeight = std::min(minHeight, h);
        maxHeight = std::max(maxHeight, h);
    }
    EXPECT_GE(minHeight, 0.0f);
    EXPECT_LE(maxHeight, 1.0f);
    EXPECT_LT(minHeight, 0.2f);
    EXPECT_GT(maxHeight, 0.8f);
}

TEST(TreeModel, trunkOnlyTree) {
    const std::vector<TreeTypeParams> types = {trunkOnly()};
    const TreeRecords r = generateTreesReference(types, {makeInstance(0, 7)}, defaultTreeFrame());
    ASSERT_EQ(r.stems[0].size(), 1u);
    EXPECT_TRUE(r.stems[1].empty());
    EXPECT_TRUE(r.leaves.empty());
    ASSERT_EQ(r.segments.size(), 4u);
    const float length = r.stems[0][0].posLength.w;
    EXPECT_NEAR(length, 10.0f, 1e-5f); // scale 10, scaleV 0, length 1
    EXPECT_NEAR(r.stems[0][0].dims.x, 10.0f * 0.02f, 1e-6f);
    for (size_t i = 0; i < r.segments.size(); ++i) {
        const SegmentRecord& d = r.segments[i];
        EXPECT_EQ(d.ids.w, i);
        EXPECT_NEAR(distance3(d.fromPosRadius, d.toPosRadius), length / 4.0f, 1e-4f);
        // Straight up along +y.
        EXPECT_NEAR(d.toPosRadius.y, length * float(i + 1) / 4.0f, 1e-4f);
        if (i > 0) {
            EXPECT_NEAR(distance3(r.segments[i - 1].toPosRadius, d.fromPosRadius), 0.0f, 1e-6f);
            EXPECT_NEAR(r.segments[i - 1].toPosRadius.w, d.fromPosRadius.w, 1e-6f);
        }
    }
    EXPECT_NEAR(r.segments.back().toPosRadius.w, 0.0f, 1e-6f);
}

TEST(TreeModel, splitsCreateClones) {
    TreeTypeParams p = trunkOnly();
    p.nSegSplits.x = 1.0f;
    p.nSplitAngle.x = 20.0f;
    {
        const TreeRecords r = generateTreesReference({p}, {makeInstance(0, 3)}, defaultTreeFrame());
        // 4 segments: 1 + 2 + 4 + 8 clones.
        EXPECT_EQ(r.segments.size(), 15u);
        std::set<uint32_t> clones;
        for (const SegmentRecord& d : r.segments) {
            if (d.ids.w == 3u) {
                clones.insert(d.ids.y);
            }
        }
        EXPECT_EQ(clones.size(), 8u);
    }
    // 8 segments would need 128 clones; the 32 threads of a stem cap them.
    p.nCurveRes.x = 8.0f;
    const TreeRecords r = generateTreesReference({p}, {makeInstance(0, 3)}, defaultTreeFrame());
    EXPECT_EQ(r.segments.size(), 1u + 2u + 4u + 8u + 16u + 32u + 32u + 32u);
}

TEST(TreeModel, childCountsFollowWeberPenn) {
    TreeTypeParams p = trunkOnly();
    p.ratios.x = 2.0f;
    p.shapeBase.y = 0.4f; // base size
    p.nBranches.y = 20.0f;
    p.nLength.y = 0.3f;
    TreeRecords r = generateTreesReference({p}, {makeInstance(0, 11)}, defaultTreeFrame());
    ASSERT_EQ(r.stems[1].size(), 20u);
    // The threads of a stem write its children in turns, not in order of offset.
    std::sort(r.stems[1].begin(), r.stems[1].end(),
              [](const StemRecord& a, const StemRecord& b) { return a.dims.z < b.dims.z; });
    const float length = r.stems[0][0].posLength.w;
    for (size_t k = 0; k < r.stems[1].size(); ++k) {
        const float offset = r.stems[1][k].dims.z;
        EXPECT_NEAR(offset, 0.4f * length + (float(k) + 0.5f) / 20.0f * 0.6f * length, 1e-3f) << k;
        // Straight trunk along +y: the child sits on the axis at its offset.
        EXPECT_NEAR(r.stems[1][k].posLength.y, offset, 1e-3f) << k;
        EXPECT_EQ(treeLevel(r.stems[1][k].ids), 1);
        // Child length: length * lengthChildMax * shapeRatio(cylindrical) = 0.3 length.
        EXPECT_NEAR(r.stems[1][k].posLength.w, 0.3f * length, 1e-4f);
    }
}

TEST(TreeModel, leavesOnlyOnLastLevel) {
    TreeTypeParams p = trunkOnly();
    p.ratios.x = 2.0f;
    p.nBranches.y = 10.0f;
    p.nLength.y = 0.3f;
    p.leafCount = glm::vec4(8.0f, 0.2f, 1.0f, 0.0f);
    const TreeRecords r = generateTreesReference({p}, {makeInstance(0, 5, glm::vec3(0.0f), 0.5f)}, defaultTreeFrame());
    ASSERT_EQ(r.stems[1].size(), 10u);
    EXPECT_TRUE(r.stems[2].empty());
    ASSERT_FALSE(r.leaves.empty());
    // Every level 1 stem carries about leaves * shapeRatio(4, offset ratio) leaves.
    EXPECT_GE(r.leaves.size(), 10u * 4u);
    EXPECT_LE(r.leaves.size(), 10u * 9u);
    for (const LeafRecord& l : r.leaves) {
        EXPECT_NEAR(l.posScale.w, 0.2f * 0.5f, 1e-6f);
    }
}

TEST(TreeModel, generationIsDeterministicPerSeed) {
    const std::vector<TreeTypeParams> types = {tree_presets::quakingAspen()};
    const TreeRecords a = generateTreesReference(types, {makeInstance(0, 9)}, defaultTreeFrame());
    const TreeRecords b = generateTreesReference(types, {makeInstance(0, 9)}, defaultTreeFrame());
    const TreeRecords c = generateTreesReference(types, {makeInstance(0, 10)}, defaultTreeFrame());
    ASSERT_EQ(a.segments.size(), b.segments.size());
    ASSERT_EQ(a.leaves.size(), b.leaves.size());
    for (size_t i = 0; i < a.segments.size(); ++i) {
        EXPECT_EQ(std::memcmp(&a.segments[i], &b.segments[i], sizeof(SegmentRecord)), 0);
    }
    const bool sameSize = a.segments.size() == c.segments.size() && a.leaves.size() == c.leaves.size();
    const bool sameTip = sameSize && distance3(a.segments.back().toPosRadius, c.segments.back().toPosRadius) < 1e-6f;
    EXPECT_FALSE(sameTip);
}

TEST(TreeModel, attractionUpRaisesBranchTips) {
    auto meanTipHeight = [](float attraction) {
        TreeTypeParams p = tree_presets::quakingAspen();
        p.leafCount.x = 0.0f;
        p.leafCount.w = attraction;
        const TreeRecords r = generateTreesReference({p}, {makeInstance(0, 21)}, defaultTreeFrame());
        double sum = 0.0;
        size_t count = 0;
        for (const SegmentRecord& d : r.segments) {
            if (treeLevel(d.ids) == 2 && d.shape.z + 1e-4f >= 1.0f - 1.0f / 3.0f) {
                sum += d.toPosRadius.y - d.fromPosRadius.y;
                ++count;
            }
        }
        return sum / double(count);
    };
    EXPECT_GT(meanTipHeight(1.0f), meanTipHeight(-1.0f));
}

TEST(TreeModel, presetsGenerateFullTrees) {
    const std::vector<TreeTypeParams> types = {tree_presets::quakingAspen(), tree_presets::blackTupelo(),
                                               tree_presets::sassafras(), tree_presets::pine()};
    for (uint32_t type = 0; type < types.size(); ++type) {
        const TreeRecords r = generateTreesReference(types, {makeInstance(type, 1)}, defaultTreeFrame());
        const int levels = int(types[type].ratios.x);
        for (int level = 0; level < levels; ++level) {
            EXPECT_FALSE(r.stems[level].empty()) << "type " << type << " level " << level;
        }
        EXPECT_GT(r.leaves.size(), 100u) << "type " << type;
        EXPECT_LT(r.leaves.size(), 200000u) << "type " << type;
        std::cout << "type " << type << ": stems " << r.stems[0].size() << " " << r.stems[1].size() << " "
                  << r.stems[2].size() << " " << r.stems[3].size() << ", segments " << r.segments.size() << ", leaves "
                  << r.leaves.size() << std::endl;
    }
}
