// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - ScheduleMatchesDiffusers: The shifted flow schedule and truncated timesteps match
 *   diffusers for four steps
 * - SingleStepReturnsDataPrediction: A single step lands on the data prediction x -
 *   sigma * v
 * - FourStepsMatchDiffusers: Four steps with a synthetic velocity reproduce diffusers'
 *   UniPC samples, warm-up and corrector included
 * - RejectsInvalidUse: Stepping past the schedule and mismatched sizes are rejected
 **/

#include <cstddef>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>

#include "klartraum/cosmos3/unipc_flow_scheduler.hpp"

using klartraum::UniPCFlowScheduler;

namespace {

// Synthetic model used to generate the references with diffusers'
// UniPCMultistepScheduler (flow_shift 12, bh2, solver_order 2, predict_x0,
// lower_order_final, final_sigmas_type "zero", Cosmos3's native sigmas).
std::vector<float> syntheticVelocity(const std::vector<float>& sample, size_t step) {
    std::vector<float> velocity(sample.size());
    for (size_t i = 0; i < sample.size(); ++i)
        velocity[i] = 0.5f * sample[i] + 0.1f * float(step + 1);
    return velocity;
}

} // namespace

TEST(UniPCFlowSchedulerTest, ScheduleMatchesDiffusers) {
    UniPCFlowScheduler scheduler(4, 12.0);
    const std::vector<float> sigmas{0.9999166131019592f, 0.9728676676750183f, 0.9229347705841064f, 0.7997865080833435f,
                                    0.0f};
    const std::vector<float> timesteps{999.0f, 972.0f, 922.0f, 799.0f};
    ASSERT_EQ(scheduler.sigmas().size(), sigmas.size());
    for (size_t i = 0; i < sigmas.size(); ++i)
        EXPECT_NEAR(scheduler.sigmas()[i], sigmas[i], 1e-7f) << i;
    EXPECT_EQ(scheduler.timesteps(), timesteps);
}

TEST(UniPCFlowSchedulerTest, SingleStepReturnsDataPrediction) {
    UniPCFlowScheduler scheduler(1, 12.0);
    std::vector<float> sample{0.5f, -1.25f, 2.0f};
    scheduler.step(syntheticVelocity(sample, 0), sample);
    const std::vector<float> expected{0.15002918243408203f, -0.725043773651123f, 0.9000916481018066f};
    for (size_t i = 0; i < expected.size(); ++i)
        EXPECT_NEAR(sample[i], expected[i], 1e-6f) << i;
}

TEST(UniPCFlowSchedulerTest, FourStepsMatchDiffusers) {
    UniPCFlowScheduler scheduler(4, 12.0);
    std::vector<float> sample{0.5f, -1.25f, 2.0f};
    const std::vector<std::vector<float>> expected{
        {0.49053287506103516f, -1.2357993125915527f, 1.9702461957931519f},
        {0.4666614532470703f, -1.2167737483978271f, 1.9096060991287231f},
        {0.39373424649238586f, -1.1878362894058228f, 1.749366283416748f},
        {-0.08363205194473267f, -1.032743215560913f, 0.7298918962478638f},
    };
    for (size_t step = 0; step < expected.size(); ++step) {
        EXPECT_EQ(scheduler.stepIndex(), step);
        scheduler.step(syntheticVelocity(sample, step), sample);
        for (size_t i = 0; i < sample.size(); ++i) {
            EXPECT_NEAR(sample[i], expected[step][i], 2e-6f) << "step " << step << ", element " << i;
        }
    }
}

TEST(UniPCFlowSchedulerTest, RejectsInvalidUse) {
    UniPCFlowScheduler scheduler(1, 12.0);
    std::vector<float> sample{1.0f, 2.0f};
    EXPECT_THROW(scheduler.step({1.0f}, sample), std::runtime_error);
    scheduler.step({0.0f, 0.0f}, sample);
    EXPECT_THROW(scheduler.step({0.0f, 0.0f}, sample), std::runtime_error);
}
