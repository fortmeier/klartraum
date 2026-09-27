/**
 * TESTS:
 * - overlappingLifetimesUseDifferentSlots: simultaneously live tensors cannot alias
 * - disjointLifetimesReuseStorage: non-overlapping tensors share a slot
 * - inclusiveLifetimeBoundaryDoesNotAlias: producer/consumer boundary remains live
 * - differentElementTypesDoNotAlias: slots remain type-compatible
 * - reusableSlotGrowsForLargerTensor: a later larger tensor grows an idle slot
 **/

#include <gtest/gtest.h>

#include "klartraum/computegraph/tensor_memory_planner.hpp"

using namespace klartraum;

TEST(TensorMemoryPlannerTest, overlappingLifetimesUseDifferentSlots) {
    const auto plan = TensorMemoryPlanner::plan({
        {"a", 64, 0, 2, typeid(float)},
        {"b", 32, 1, 3, typeid(float)},
    });
    ASSERT_EQ(plan.assignments.size(), 2);
    EXPECT_NE(plan.assignments[0].slot, plan.assignments[1].slot);
    EXPECT_EQ(plan.allocatedBytes, 96);
    EXPECT_EQ(plan.peakLiveBytes, 96);
}

TEST(TensorMemoryPlannerTest, disjointLifetimesReuseStorage) {
    const auto plan = TensorMemoryPlanner::plan({
        {"a", 64, 0, 1, typeid(float)},
        {"b", 32, 2, 3, typeid(float)},
    });
    EXPECT_EQ(plan.assignments[0].slot, plan.assignments[1].slot);
    EXPECT_EQ(plan.slotCapacities, std::vector<size_t>({64}));
    EXPECT_EQ(plan.logicalBytes, 96);
    EXPECT_EQ(plan.allocatedBytes, 64);
    EXPECT_EQ(plan.peakLiveBytes, 64);
}

TEST(TensorMemoryPlannerTest, inclusiveLifetimeBoundaryDoesNotAlias) {
    const auto plan = TensorMemoryPlanner::plan({
        {"a", 64, 0, 2, typeid(float)},
        {"b", 64, 2, 3, typeid(float)},
    });
    EXPECT_NE(plan.assignments[0].slot, plan.assignments[1].slot);
}

TEST(TensorMemoryPlannerTest, differentElementTypesDoNotAlias) {
    const auto plan = TensorMemoryPlanner::plan({
        {"float", 64, 0, 0, typeid(float)},
        {"int", 64, 1, 1, typeid(int)},
    });
    EXPECT_NE(plan.assignments[0].slot, plan.assignments[1].slot);
}

TEST(TensorMemoryPlannerTest, reusableSlotGrowsForLargerTensor) {
    const auto plan = TensorMemoryPlanner::plan({
        {"small", 32, 0, 0, typeid(float)},
        {"large", 128, 1, 2, typeid(float)},
    });
    EXPECT_EQ(plan.assignments[0].slot, plan.assignments[1].slot);
    EXPECT_EQ(plan.slotCapacities, std::vector<size_t>({128}));
    EXPECT_EQ(plan.allocatedBytes, 128);
}
