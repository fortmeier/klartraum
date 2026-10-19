// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - trunkOnlyMatchesReference: a one-level tree: the GPU writes the CPU reference's trunk and segments
 * - presetTreesMatchReference: several preset trees: equal record counts per kind and level as the CPU
 *   reference, and records equal within float tolerance
 * - countersResetEveryFrame: submitting twice gives the same counts, not twice as many records
 * - instanceEditsApplyNextFrame: moving an instance through its HostValues moves its records in the next frame
 * - overflowIsFlagged: too small capacities drop the extra records and set the overflow bits
 * - twoPathsGenerateIndependently: in a two-path graph, a submission of path 1 alone generates the trees
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <tuple>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/tree/tree_generator.hpp"

using namespace klartraum;

namespace {

TreeInstance makeInstance(uint32_t type, uint32_t seed, glm::vec3 pos = glm::vec3(0.0f), float scale = 1.0f) {
    TreeInstance inst{};
    inst.posScale = glm::vec4(pos, scale);
    inst.rot = treeUpRotation(glm::vec3(0.0f, 1.0f, 0.0f));
    inst.ids = glm::uvec4(type, seed, 0u, 0u);
    return inst;
}

auto stemKey(const StemRecord& s) { return std::make_tuple(s.ids.z, s.ids.y); }
auto segmentKey(const SegmentRecord& d) { return std::make_tuple(d.ids.z, d.ids.y, d.ids.w); }
auto leafKey(const LeafRecord& l) { return std::make_tuple(l.ids.z, l.ids.y); }

void sortRecords(TreeRecords& r) {
    for (auto& level : r.stems) {
        std::sort(level.begin(), level.end(), [](auto& a, auto& b) { return stemKey(a) < stemKey(b); });
    }
    std::sort(r.segments.begin(), r.segments.end(), [](auto& a, auto& b) { return segmentKey(a) < segmentKey(b); });
    std::sort(r.leaves.begin(), r.leaves.end(), [](auto& a, auto& b) { return leafKey(a) < leafKey(b); });
}

float maxDiff(const glm::vec4& a, const glm::vec4& b) {
    const glm::vec4 d = glm::abs(a - b);
    return std::max(std::max(d.x, d.y), std::max(d.z, d.w));
}

// Compares sorted GPU and CPU records. GPU float math (fused multiply-adds,
// approximate sin/cos/pow) differs from the CPU in the last bits, and the
// differences add up along a stem and down the levels. `tolerance` is in world
// units.
void expectRecordsMatch(TreeRecords gpu, TreeRecords cpu, float tolerance) {
    sortRecords(gpu);
    sortRecords(cpu);
    for (int level = 0; level < TREE_MAX_LEVELS; ++level) {
        ASSERT_EQ(gpu.stems[level].size(), cpu.stems[level].size()) << "level " << level;
        for (size_t i = 0; i < gpu.stems[level].size(); ++i) {
            ASSERT_EQ(gpu.stems[level][i].ids, cpu.stems[level][i].ids);
            EXPECT_LT(maxDiff(gpu.stems[level][i].posLength, cpu.stems[level][i].posLength), tolerance);
        }
    }
    ASSERT_EQ(gpu.segments.size(), cpu.segments.size());
    float worstSegment = 0.0f;
    for (size_t i = 0; i < gpu.segments.size(); ++i) {
        ASSERT_EQ(gpu.segments[i].ids, cpu.segments[i].ids);
        worstSegment = std::max(worstSegment, maxDiff(gpu.segments[i].fromPosRadius, cpu.segments[i].fromPosRadius));
        worstSegment = std::max(worstSegment, maxDiff(gpu.segments[i].toPosRadius, cpu.segments[i].toPosRadius));
        worstSegment = std::max(worstSegment, maxDiff(gpu.segments[i].toRot, cpu.segments[i].toRot) * 1e-2f);
    }
    EXPECT_LT(worstSegment, tolerance);
    ASSERT_EQ(gpu.leaves.size(), cpu.leaves.size());
    float worstLeaf = 0.0f;
    for (size_t i = 0; i < gpu.leaves.size(); ++i) {
        ASSERT_EQ(gpu.leaves[i].ids, cpu.leaves[i].ids);
        worstLeaf = std::max(worstLeaf, maxDiff(gpu.leaves[i].posScale, cpu.leaves[i].posScale));
    }
    EXPECT_LT(worstLeaf, tolerance);
    std::cout << "worst segment difference " << worstSegment << ", worst leaf difference " << worstLeaf << std::endl;
}

std::vector<TreeTypeParams> presetTypes() {
    return {tree_presets::quakingAspen(), tree_presets::blackTupelo(), tree_presets::sassafras(), tree_presets::pine()};
}

} // namespace

class TreeGenerationTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vulkanContext = &frontend->getKlartraumEngine().getVulkanContext();
    }
    void TearDown() override { frontend.reset(); }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vulkanContext = nullptr;
};

TEST_F(TreeGenerationTest, trunkOnlyMatchesReference) {
    TreeTypeParams p = tree_presets::quakingAspen();
    p.ratios.x = 1.0f;
    p.leafCount.x = 0.0f;
    const std::vector<TreeTypeParams> types = {p};
    const std::vector<TreeInstance> instances = {makeInstance(0, 4)};
    const TreeFrame frame = defaultTreeFrame();
    auto generator = std::make_shared<TreeGenerator>(*vulkanContext, types, instances,
                                                     TreeCapacity::fromReference(types, instances, frame), frame);
    ComputeGraph graph(*vulkanContext, 1);
    graph.compileFrom(generator);
    graph.submitAndWait(vulkanContext->getGraphicsQueue(), 0);

    const TreeRecords gpu = generator->readRecords();
    const TreeRecords cpu = generateTreesReference(types, instances, frame);
    ASSERT_EQ(gpu.stems[0].size(), 1u);
    ASSERT_EQ(gpu.segments.size(), cpu.segments.size());
    EXPECT_EQ(generator->readCounters()[kTreeCounterOverflow], 0u);
    expectRecordsMatch(gpu, cpu, 1e-4f);
}

TEST_F(TreeGenerationTest, presetTreesMatchReference) {
    const std::vector<TreeTypeParams> types = presetTypes();
    std::vector<TreeInstance> instances;
    for (uint32_t i = 0; i < 6; ++i) {
        instances.push_back(makeInstance(i % 4, 100 + i, glm::vec3(float(i) * 3.0f, 0.0f, 0.0f), 0.1f));
    }
    const TreeFrame frame = defaultTreeFrame();
    auto generator = std::make_shared<TreeGenerator>(*vulkanContext, types, instances,
                                                     TreeCapacity::fromReference(types, instances, frame), frame);
    ComputeGraph graph(*vulkanContext, 1);
    graph.compileFrom(generator);
    graph.submitAndWait(vulkanContext->getGraphicsQueue(), 0);

    EXPECT_EQ(generator->readCounters()[kTreeCounterOverflow], 0u);
    // Trees about 1.5 to 2.5 units tall at scale 0.1.
    expectRecordsMatch(generator->readRecords(), generateTreesReference(types, instances, frame), 1e-3f);
}

TEST_F(TreeGenerationTest, countersResetEveryFrame) {
    const std::vector<TreeTypeParams> types = presetTypes();
    const std::vector<TreeInstance> instances = {makeInstance(2, 8)};
    const TreeFrame frame = defaultTreeFrame();
    auto generator = std::make_shared<TreeGenerator>(*vulkanContext, types, instances,
                                                     TreeCapacity::fromReference(types, instances, frame), frame);
    ComputeGraph graph(*vulkanContext, 1);
    graph.compileFrom(generator);
    graph.submitAndWait(vulkanContext->getGraphicsQueue(), 0);
    const std::vector<uint32_t> first = generator->readCounters();
    graph.submitAndWait(vulkanContext->getGraphicsQueue(), 0);
    const std::vector<uint32_t> second = generator->readCounters();
    EXPECT_EQ(first, second);
    EXPECT_GT(second[kTreeCounterLeaves], 0u);
}

TEST_F(TreeGenerationTest, instanceEditsApplyNextFrame) {
    const std::vector<TreeTypeParams> types = presetTypes();
    std::vector<TreeInstance> instances = {makeInstance(0, 8)};
    const TreeFrame frame = defaultTreeFrame();
    auto generator = std::make_shared<TreeGenerator>(*vulkanContext, types, instances,
                                                     TreeCapacity::fromReference(types, instances, frame), frame);
    ComputeGraph graph(*vulkanContext, 1);
    graph.compileFrom(generator);
    graph.submitAndWait(vulkanContext->getGraphicsQueue(), 0);
    TreeRecords before = generator->readRecords();

    instances[0].posScale = glm::vec4(5.0f, 0.0f, -2.0f, 1.0f);
    generator->getInstances().set(instances);
    graph.submitAndWait(vulkanContext->getGraphicsQueue(), 0);
    TreeRecords after = generator->readRecords();

    sortRecords(before);
    sortRecords(after);
    ASSERT_EQ(before.segments.size(), after.segments.size());
    for (size_t i = 0; i < before.segments.size(); i += 97) {
        const glm::vec3 shift =
            glm::vec3(after.segments[i].fromPosRadius) - glm::vec3(before.segments[i].fromPosRadius);
        EXPECT_NEAR(shift.x, 5.0f, 1e-3f);
        EXPECT_NEAR(shift.y, 0.0f, 1e-3f);
        EXPECT_NEAR(shift.z, -2.0f, 1e-3f);
    }
}

TEST_F(TreeGenerationTest, overflowIsFlagged) {
    const std::vector<TreeTypeParams> types = presetTypes();
    const std::vector<TreeInstance> instances = {makeInstance(0, 8)};
    const TreeFrame frame = defaultTreeFrame();
    TreeCapacity capacity = TreeCapacity::fromReference(types, instances, frame);
    capacity.stems[2] = 10;
    capacity.leaves = 100;
    auto generator = std::make_shared<TreeGenerator>(*vulkanContext, types, instances, capacity, frame);
    ComputeGraph graph(*vulkanContext, 1);
    graph.compileFrom(generator);
    graph.submitAndWait(vulkanContext->getGraphicsQueue(), 0);

    const std::vector<uint32_t> counters = generator->readCounters();
    EXPECT_EQ(counters[kTreeCounterOverflow], (1u << 2u) | 32u);
    const TreeRecords gpu = generator->readRecords();
    EXPECT_EQ(gpu.stems[2].size(), 10u);
    EXPECT_EQ(gpu.leaves.size(), 100u);
}

TEST_F(TreeGenerationTest, twoPathsGenerateIndependently) {
    const std::vector<TreeTypeParams> types = presetTypes();
    const std::vector<TreeInstance> instances = {makeInstance(1, 3, glm::vec3(0.0f), 0.1f)};
    const TreeFrame frame = defaultTreeFrame();
    auto generator = std::make_shared<TreeGenerator>(*vulkanContext, types, instances,
                                                     TreeCapacity::fromReference(types, instances, frame), frame);
    ComputeGraph graph(*vulkanContext, 2);
    graph.compileFrom(generator);
    graph.submitAndWait(vulkanContext->getGraphicsQueue(), 1);

    expectRecordsMatch(generator->readRecords(1), generateTreesReference(types, instances, frame), 1e-3f);
}
