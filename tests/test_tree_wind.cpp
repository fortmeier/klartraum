// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - noWindNoMotion: without wind, the trees are the same at every time
 * - tipsMoveDownwind: in wind, branch tips move in the wind's direction on average, the trunk's base stays
 * - strongerWindBendsMore: the mean tip displacement grows with the wind's strength
 * - windChangesOverTime: gusts and sway make the tips move between two times, more for thin twigs than the trunk
 * - leavesFlutter: in wind the leaves turn about their stalks over time
 * - gpuMatchesReferenceInWind: the GPU generation in wind equals the CPU reference
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <map>
#include <memory>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/tree/tree_generator.hpp"

#include "tree_test_helpers.hpp"

using namespace klartraum;
using namespace klartraum::tree;
using namespace klartraum::tree_test;

namespace {

TreeFrame windFrame(float strength, float time, glm::vec3 direction = glm::vec3(1.0f, 0.0f, 0.0f)) {
    TreeFrame frame = defaultTreeFrame();
    frame.wind = glm::vec4(glm::normalize(direction), strength);
    frame.timeSeason.x = time;
    return frame;
}

TreeRecords generate(const TreeFrame& frame) {
    const std::vector<TreeTypeParams> types = {tree_presets::quakingAspen()};
    return generateTreesReference(types, {makeInstance(0, 12, glm::vec3(0.0f), 0.1f)}, frame);
}

// End points of the last segments of level `level`, keyed by stem clone seed.
std::map<uint32_t, glm::vec3> tips(const TreeRecords& r, int level) {
    std::map<uint32_t, std::pair<uint32_t, glm::vec3>> last;
    for (const SegmentRecord& d : r.segments) {
        if (treeLevel(d.ids) != level) {
            continue;
        }
        auto& entry = last[d.ids.y];
        if (d.ids.w >= entry.first) {
            entry = {d.ids.w, glm::vec3(d.toPosRadius)};
        }
    }
    std::map<uint32_t, glm::vec3> result;
    for (auto& [seed, entry] : last) {
        result[seed] = entry.second;
    }
    return result;
}

// Mean displacement of the tips between two generations.
glm::vec3 meanShift(const TreeRecords& a, const TreeRecords& b, int level) {
    const auto ta = tips(a, level), tb = tips(b, level);
    glm::vec3 sum(0.0f);
    int count = 0;
    for (auto& [seed, p] : ta) {
        auto it = tb.find(seed);
        if (it != tb.end()) {
            sum += it->second - p;
            ++count;
        }
    }
    return count > 0 ? sum / float(count) : glm::vec3(0.0f);
}

float meanMotion(const TreeRecords& a, const TreeRecords& b, int level) {
    const auto ta = tips(a, level), tb = tips(b, level);
    float sum = 0.0f;
    int count = 0;
    for (auto& [seed, p] : ta) {
        auto it = tb.find(seed);
        if (it != tb.end()) {
            sum += glm::length(it->second - p);
            ++count;
        }
    }
    return count > 0 ? sum / float(count) : 0.0f;
}

} // namespace

TEST(TreeWind, noWindNoMotion) {
    const TreeRecords a = generate(windFrame(0.0f, 0.0f));
    const TreeRecords b = generate(windFrame(0.0f, 3.7f));
    ASSERT_EQ(a.segments.size(), b.segments.size());
    for (size_t i = 0; i < a.segments.size(); ++i) {
        ASSERT_EQ(std::memcmp(&a.segments[i], &b.segments[i], sizeof(SegmentRecord)), 0);
    }
}

TEST(TreeWind, tipsMoveDownwind) {
    const TreeRecords calm = generate(windFrame(0.0f, 1.0f));
    const TreeRecords windy = generate(windFrame(0.5f, 1.0f));
    const glm::vec3 shift = meanShift(calm, windy, 2);
    std::cout << "mean tip shift " << shift.x << " " << shift.y << " " << shift.z << std::endl;
    EXPECT_GT(shift.x, 0.01f);
    EXPECT_GT(shift.x, 3.0f * std::abs(shift.z));
    // The trunk starts where the instance stands.
    EXPECT_EQ(calm.stems[0][0].posLength, windy.stems[0][0].posLength);
    // Wind towards -z moves the tips towards -z.
    const glm::vec3 other = meanShift(calm, generate(windFrame(0.5f, 1.0f, glm::vec3(0.0f, 0.0f, -1.0f))), 2);
    EXPECT_LT(other.z, -0.01f);
}

TEST(TreeWind, strongerWindBendsMore) {
    const TreeRecords calm = generate(windFrame(0.0f, 2.0f));
    float previous = 0.0f;
    for (float strength : {0.2f, 0.4f, 0.8f}) {
        const float shift = meanShift(calm, generate(windFrame(strength, 2.0f)), 2).x;
        EXPECT_GT(shift, previous) << strength;
        previous = shift;
    }
}

TEST(TreeWind, windChangesOverTime) {
    const TreeRecords a = generate(windFrame(0.5f, 1.0f));
    const TreeRecords b = generate(windFrame(0.5f, 1.6f));
    const float twigs = meanMotion(a, b, 2);
    const float trunk = meanMotion(a, b, 0);
    std::cout << "motion in 0.6 s: twigs " << twigs << ", trunk top " << trunk << std::endl;
    EXPECT_GT(twigs, 0.002f);
    EXPECT_LT(trunk, twigs);
}

TEST(TreeWind, leavesFlutter) {
    const TreeRecords a = generate(windFrame(0.5f, 1.0f));
    const TreeRecords b = generate(windFrame(0.5f, 1.1f));
    ASSERT_EQ(a.leaves.size(), b.leaves.size());
    std::map<uint32_t, glm::vec4> rotations;
    for (const LeafRecord& l : a.leaves) {
        rotations[l.ids.y] = l.rot;
    }
    float turned = 0.0f;
    for (const LeafRecord& l : b.leaves) {
        turned += 1.0f - std::abs(glm::dot(rotations[l.ids.y], l.rot));
    }
    EXPECT_GT(turned / float(b.leaves.size()), 1e-3f);
}

TEST(TreeWind, gpuMatchesReferenceInWind) {
    HeadlessFrontend frontend;
    VulkanContext& vc = frontend.getKlartraumEngine().getVulkanContext();
    const std::vector<TreeTypeParams> types = {tree_presets::quakingAspen(), tree_presets::pine()};
    const std::vector<TreeInstance> instances = {makeInstance(0, 12, glm::vec3(0.0f), 0.1f),
                                                 makeInstance(1, 13, glm::vec3(2.0f, 0.0f, 0.0f), 0.1f)};
    const TreeFrame frame = windFrame(0.6f, 2.3f, glm::vec3(1.0f, 0.0f, 0.5f));
    auto generator = std::make_shared<TreeGenerator>(vc, types, instances,
                                                     TreeCapacity::fromReference(types, instances, frame), frame);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(generator);
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    TreeRecords gpu = generator->readRecords();
    TreeRecords cpu = generateTreesReference(types, instances, frame);
    ASSERT_EQ(gpu.segments.size(), cpu.segments.size());
    ASSERT_EQ(gpu.leaves.size(), cpu.leaves.size());
    auto key = [](const SegmentRecord& d) { return std::make_tuple(d.ids.z, d.ids.y, d.ids.w); };
    std::sort(gpu.segments.begin(), gpu.segments.end(), [&](auto& a, auto& b) { return key(a) < key(b); });
    std::sort(cpu.segments.begin(), cpu.segments.end(), [&](auto& a, auto& b) { return key(a) < key(b); });
    float worst = 0.0f;
    for (size_t i = 0; i < gpu.segments.size(); ++i) {
        ASSERT_EQ(gpu.segments[i].ids, cpu.segments[i].ids);
        worst = std::max(worst, glm::length(glm::vec3(gpu.segments[i].toPosRadius - cpu.segments[i].toPosRadius)));
    }
    std::cout << "worst segment difference in wind " << worst << std::endl;
    EXPECT_LT(worst, 1e-3f);
}
