// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - fullyGrownIsUnchanged: age 1 gives exactly the records of an instance without an age, for several types
 * - ageRoundTrips: setTreeAge() stores the age that treeAge() reads; an instance without one is fully grown
 * - seedlingIsATrunkOnly: at age 0.05 a tree is a short trunk: no branches, no leaves
 * - growthIsMonotonic: over the ages 0 to 1, the stem, segment and leaf counts, the total stem length, the leaf area
 *   and the height never decrease
 * - branchesSproutBehindTheTip: every branch starts on the grown part of a stem of the level below
 * - growthIsContinuous: from age a to a + 0.005 no segment end or leaf moves or grows by more than 2 % of the tree's
 *   height, and what is new is small
 * - bonsaiPineGrows: the bonsai pine with its needle pads grows like the presets: monotonic, fully grown at 1
 * - gpuMatchesReferenceWhileGrowing: half-grown trees generated on the GPU match the CPU reference
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <tuple>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/tree/bonsai_scene.hpp"
#include "klartraum/tree/tree_generator.hpp"

using namespace klartraum;

namespace {

TreeInstance makeInstance(uint32_t type, uint32_t seed, glm::vec3 pos = glm::vec3(0.0f), float scale = 0.1f) {
    TreeInstance inst{};
    inst.posScale = glm::vec4(pos, scale);
    inst.rot = treeUpRotation(glm::vec3(0.0f, 1.0f, 0.0f));
    inst.ids = glm::uvec4(type, seed, 0u, 0u);
    return inst;
}

std::vector<TreeTypeParams> growthTypes() {
    return {tree_presets::quakingAspen(), tree_presets::pine(), tree_presets::sassafras()};
}

TreeRecords generateAt(const std::vector<TreeTypeParams>& types, std::vector<TreeInstance> instances, float age) {
    for (TreeInstance& inst : instances)
        setTreeAge(inst, age);
    return generateTreesReference(types, instances, defaultTreeFrame());
}

// What the growth tests compare between ages.
struct Measures {
    size_t stems[TREE_MAX_LEVELS] = {};
    size_t segments = 0, leaves = 0;
    double stemLength = 0.0, leafArea = 0.0, height = -1e9;
};

Measures measure(const TreeRecords& r) {
    Measures m;
    for (int level = 0; level < TREE_MAX_LEVELS; ++level)
        m.stems[level] = r.stems[level].size();
    m.segments = r.segments.size();
    m.leaves = r.leaves.size();
    for (const SegmentRecord& d : r.segments) {
        m.stemLength += glm::length(glm::vec3(d.toPosRadius) - glm::vec3(d.fromPosRadius));
        m.height = std::max(m.height, double(std::max(d.fromPosRadius.y, d.toPosRadius.y)));
    }
    for (const LeafRecord& l : r.leaves) {
        m.leafArea += double(l.posScale.w) * l.posScale.w;
        m.height = std::max(m.height, double(l.posScale.y));
    }
    return m;
}

bool sameBytes(const void* a, const void* b, size_t n) { return std::memcmp(a, b, n) == 0; }

void expectIdentical(const TreeRecords& a, const TreeRecords& b) {
    for (int level = 0; level < TREE_MAX_LEVELS; ++level) {
        ASSERT_EQ(a.stems[level].size(), b.stems[level].size()) << "level " << level;
        for (size_t i = 0; i < a.stems[level].size(); ++i) {
            // The stems' birth (ids.w) is the only field allowed to differ: 0 without an age.
            StemRecord x = a.stems[level][i], y = b.stems[level][i];
            x.ids.w = y.ids.w = 0u;
            EXPECT_TRUE(sameBytes(&x, &y, sizeof(StemRecord))) << "stem " << i << " of level " << level;
        }
    }
    ASSERT_EQ(a.segments.size(), b.segments.size());
    EXPECT_TRUE(sameBytes(a.segments.data(), b.segments.data(), a.segments.size() * sizeof(SegmentRecord)));
    ASSERT_EQ(a.leaves.size(), b.leaves.size());
    EXPECT_TRUE(sameBytes(a.leaves.data(), b.leaves.data(), a.leaves.size() * sizeof(LeafRecord)));
}

void expectMonotonicGrowth(const std::vector<TreeTypeParams>& types, const std::vector<TreeInstance>& instances) {
    Measures previous;
    for (int step = 0; step <= 40; ++step) {
        const float age = float(step) / 40.0f;
        SCOPED_TRACE("age " + std::to_string(age));
        const Measures m = measure(generateAt(types, instances, age));
        for (int level = 0; level < TREE_MAX_LEVELS; ++level)
            EXPECT_GE(m.stems[level], previous.stems[level]) << "level " << level;
        EXPECT_GE(m.segments, previous.segments);
        EXPECT_GE(m.leaves, previous.leaves);
        EXPECT_GE(m.stemLength, previous.stemLength - 1e-5);
        EXPECT_GE(m.leafArea, previous.leafArea - 1e-9);
        if (step > 0)
            EXPECT_GE(m.height, previous.height - 1e-5);
        previous = m;
    }
}

auto segmentKey(const SegmentRecord& d) { return std::make_tuple(d.ids.z, d.ids.x, d.ids.y, d.ids.w); }
auto leafKey(const LeafRecord& l) { return std::make_tuple(l.ids.z, l.ids.y); }

float pointSegmentDistance(glm::vec3 p, glm::vec3 a, glm::vec3 b) {
    const glm::vec3 ab = b - a;
    const float t = glm::clamp(glm::dot(p - a, ab) / std::max(glm::dot(ab, ab), 1e-12f), 0.0f, 1.0f);
    return glm::length(p - (a + t * ab));
}

} // namespace

TEST(TreeGrowth, fullyGrownIsUnchanged) {
    const std::vector<TreeTypeParams> types = growthTypes();
    std::vector<TreeInstance> instances;
    for (uint32_t i = 0; i < 6; ++i)
        instances.push_back(makeInstance(i % 3, 40 + i, glm::vec3(float(i) * 3.0f, 0.0f, 0.0f)));
    const TreeRecords without = generateTreesReference(types, instances, defaultTreeFrame());
    const TreeRecords grown = generateAt(types, instances, 1.0f);
    expectIdentical(grown, without);
}

TEST(TreeGrowth, ageRoundTrips) {
    TreeInstance inst = makeInstance(0, 1);
    EXPECT_FLOAT_EQ(treeAge(inst), 1.0f);
    for (const float age : {0.0f, 0.25f, 0.7f, 1.0f}) {
        setTreeAge(inst, age);
        EXPECT_FLOAT_EQ(treeAge(inst), age);
    }
}

TEST(TreeGrowth, seedlingIsATrunkOnly) {
    const std::vector<TreeTypeParams> types = growthTypes();
    for (uint32_t type = 0; type < types.size(); ++type) {
        SCOPED_TRACE("type " + std::to_string(type));
        const std::vector<TreeInstance> instances = {makeInstance(type, 7)};
        const Measures full = measure(generateAt(types, instances, 1.0f));
        const TreeRecords seedling = generateAt(types, instances, 0.05f);
        const Measures young = measure(seedling);
        EXPECT_EQ(young.stems[0], 1u);
        EXPECT_GT(young.segments, 0u);
        for (int level = 1; level < TREE_MAX_LEVELS; ++level)
            EXPECT_EQ(young.stems[level], 0u) << "level " << level;
        EXPECT_EQ(young.leaves, 0u);
        EXPECT_LT(young.height, 0.15 * full.height);
    }
}

TEST(TreeGrowth, growthIsMonotonic) {
    const std::vector<TreeTypeParams> types = growthTypes();
    std::vector<TreeInstance> instances;
    for (uint32_t i = 0; i < 3; ++i)
        instances.push_back(makeInstance(i, 11 + i, glm::vec3(float(i) * 3.0f, 0.0f, 0.0f)));
    expectMonotonicGrowth(types, instances);
}

TEST(TreeGrowth, branchesSproutBehindTheTip) {
    const std::vector<TreeTypeParams> types = growthTypes();
    for (uint32_t type = 0; type < types.size(); ++type)
        for (const float age : {0.3f, 0.6f, 0.85f}) {
            SCOPED_TRACE("type " + std::to_string(type) + " age " + std::to_string(age));
            const TreeRecords r = generateAt(types, {makeInstance(type, 21)}, age);
            for (int level = 1; level < TREE_MAX_LEVELS; ++level) {
                // Every branch of this level starts on a grown segment of the level below.
                std::vector<const SegmentRecord*> parents;
                for (const SegmentRecord& d : r.segments)
                    if (tree::treeLevel(d.ids) == level - 1)
                        parents.push_back(&d);
                for (const StemRecord& s : r.stems[level]) {
                    float nearest = 1e9f;
                    for (const SegmentRecord* d : parents)
                        nearest =
                            std::min(nearest, pointSegmentDistance(glm::vec3(s.posLength), glm::vec3(d->fromPosRadius),
                                                                   glm::vec3(d->toPosRadius)) -
                                                  std::max(d->fromPosRadius.w, d->toPosRadius.w));
                    EXPECT_LT(nearest, 0.01f) << "a branch of level " << level << " away from the grown stems";
                }
            }
        }
}

TEST(TreeGrowth, growthIsContinuous) {
    const std::vector<TreeTypeParams> types = growthTypes();
    for (uint32_t type = 0; type < types.size(); ++type) {
        const std::vector<TreeInstance> instances = {makeInstance(type, 31)};
        const double height = measure(generateAt(types, instances, 1.0f)).height;
        const float bound = 0.02f * float(height);
        for (const float age : {0.1f, 0.35f, 0.6f, 0.8f, 0.95f}) {
            SCOPED_TRACE("type " + std::to_string(type) + " age " + std::to_string(age));
            const TreeRecords before = generateAt(types, instances, age);
            const TreeRecords after = generateAt(types, instances, age + 0.005f);
            std::map<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>, const SegmentRecord*> earlier;
            for (const SegmentRecord& d : before.segments)
                earlier[segmentKey(d)] = &d;
            float worstMove = 0.0f, worstNew = 0.0f;
            for (const SegmentRecord& d : after.segments) {
                const auto found = earlier.find(segmentKey(d));
                if (found != earlier.end()) {
                    worstMove = std::max(worstMove,
                                         glm::length(glm::vec3(d.toPosRadius) - glm::vec3(found->second->toPosRadius)));
                    worstMove = std::max(worstMove, std::abs(d.toPosRadius.w - found->second->toPosRadius.w));
                } else {
                    worstNew = std::max(worstNew, glm::length(glm::vec3(d.toPosRadius) - glm::vec3(d.fromPosRadius)));
                }
            }
            std::map<std::tuple<uint32_t, uint32_t>, float> leafSize;
            for (const LeafRecord& l : before.leaves)
                leafSize[leafKey(l)] = l.posScale.w;
            float worstLeaf = 0.0f;
            for (const LeafRecord& l : after.leaves) {
                const auto found = leafSize.find(leafKey(l));
                worstLeaf = std::max(worstLeaf, l.posScale.w - (found != leafSize.end() ? found->second : 0.0f));
            }
            EXPECT_LT(worstMove, bound);
            EXPECT_LT(worstNew, bound);
            EXPECT_LT(worstLeaf, bound);
        }
    }
}

TEST(TreeGrowth, bonsaiPineGrows) {
    const std::vector<TreeTypeParams> types = {
        tree_presets::bonsaiPinePads(tree_presets::bonsaiStyled(tree_presets::bonsaiPine()))};
    const std::vector<TreeInstance> instances = {makeInstance(0, 5, glm::vec3(0.0f), 0.2f)};
    expectMonotonicGrowth(types, instances);
    expectIdentical(generateAt(types, instances, 1.0f), generateTreesReference(types, instances, defaultTreeFrame()));
}

class TreeGrowthGpuTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vulkanContext = &frontend->getKlartraumEngine().getVulkanContext();
    }
    void TearDown() override { frontend.reset(); }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vulkanContext = nullptr;
};

TEST_F(TreeGrowthGpuTest, gpuMatchesReferenceWhileGrowing) {
    const std::vector<TreeTypeParams> types = growthTypes();
    std::vector<TreeInstance> instances;
    for (uint32_t i = 0; i < 3; ++i)
        instances.push_back(makeInstance(i, 61 + i, glm::vec3(float(i) * 3.0f, 0.0f, 0.0f)));
    const TreeFrame frame = defaultTreeFrame();
    auto generator = std::make_shared<TreeGenerator>(*vulkanContext, types, instances,
                                                     TreeCapacity::fromReference(types, instances, frame), frame);
    ComputeGraph graph(*vulkanContext, 1);
    graph.compileFrom(generator);
    for (const float age : {0.3f, 0.7f}) {
        SCOPED_TRACE("age " + std::to_string(age));
        std::vector<TreeInstance> growing = instances;
        for (TreeInstance& inst : growing)
            setTreeAge(inst, age);
        generator->getInstances().set(growing);
        graph.submitAndWait(vulkanContext->getGraphicsQueue(), 0);
        EXPECT_EQ(generator->readCounters()[kTreeCounterOverflow], 0u);
        TreeRecords gpu = generator->readRecords();
        TreeRecords cpu = generateTreesReference(types, growing, frame);
        for (int level = 0; level < TREE_MAX_LEVELS; ++level)
            EXPECT_EQ(gpu.stems[level].size(), cpu.stems[level].size()) << "level " << level;
        ASSERT_EQ(gpu.segments.size(), cpu.segments.size());
        ASSERT_EQ(gpu.leaves.size(), cpu.leaves.size());
        // The same segments, wherever the threads put them.
        auto bySegment = [](const SegmentRecord& a, const SegmentRecord& b) { return segmentKey(a) < segmentKey(b); };
        std::sort(gpu.segments.begin(), gpu.segments.end(), bySegment);
        std::sort(cpu.segments.begin(), cpu.segments.end(), bySegment);
        float worst = 0.0f;
        for (size_t i = 0; i < gpu.segments.size(); ++i) {
            ASSERT_EQ(segmentKey(gpu.segments[i]), segmentKey(cpu.segments[i]));
            worst = std::max(
                worst, glm::length(glm::vec3(gpu.segments[i].toPosRadius) - glm::vec3(cpu.segments[i].toPosRadius)));
        }
        EXPECT_LT(worst, 1e-3f);
    }
}
