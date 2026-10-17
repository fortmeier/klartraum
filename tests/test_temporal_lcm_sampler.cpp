// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - rejectsInvalidSettings: the sampler needs one equally sized noise latent per step, and a start step that skips
 *   at least one step and leaves at least one
 * - keyframeRunsAllStepsFromNoise: the first frame first samples the keyframe from noise latent 0 through every
 *   timestep, exactly as stepping the scheduler by hand
 * - framesAreImg2ImgOfTheKeyframe: with anchorWeight 1, every frame (the first included) starts at the start step
 *   from the noised keyframe and visits only the remaining timesteps
 * - anchorWeightMixesPreviousFrameAndKeyframe: with a smaller anchor weight, a frame starts from the noised mix of
 *   the previous frame and the keyframe
 * - independentFramesRepeatTheFullChain: with `independent` set, every frame runs all steps from the shared noise,
 *   so equal predictions give equal frames, and there is no keyframe
 **/

#include <gtest/gtest.h>

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <vector>

#include "klartraum/sd15/temporal_lcm_sampler.hpp"

using namespace klartraum;

namespace {

LcmScheduler threeSteps() {
    return LcmScheduler(
        {{999, 0.1f, 0.4f, 0.2f, 0.8f}, {666, 0.4f, 0.7f, 0.05f, 0.95f}, {333, 0.7f, 0.99f, 0.0f, 1.0f}});
}

std::vector<std::vector<float>> threeNoises() { return {{1.0f, -1.0f}, {0.5f, 0.25f}, {-0.75f, 2.0f}}; }

// A predictor that depends on latents and timestep, and records what it sees.
struct RecordingPredictor {
    std::vector<int64_t> timesteps;
    std::vector<std::vector<float>> inputs;
    float offset = 0.0f;
    std::vector<float> operator()(const std::vector<float>& latents, int64_t timestep) {
        timesteps.push_back(timestep);
        inputs.push_back(latents);
        std::vector<float> result(latents.size());
        for (size_t i = 0; i < latents.size(); ++i) {
            result[i] = 0.3f * latents[i] + 0.001f * static_cast<float>(timestep) + offset;
        }
        return result;
    }
};

// The scheduler run by hand from step `first` with the same predictor.
std::vector<float> runByHand(size_t first, std::vector<float> latents, float offset) {
    const auto scheduler = threeSteps();
    RecordingPredictor predictor;
    predictor.offset = offset;
    std::vector<float> denoised;
    for (size_t step = first; step < 3; ++step) {
        const auto prediction = predictor(latents, scheduler.getSteps()[step].timestep);
        scheduler.step(step, latents, prediction, step + 1 < 3 ? threeNoises()[step + 1] : std::vector<float>{},
                       denoised);
    }
    return latents;
}

void expectNear(const std::vector<float>& actual, const std::vector<float>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-5f) << "element " << i;
    }
}

} // namespace

TEST(TemporalLcmSampler, rejectsInvalidSettings) {
    EXPECT_THROW(TemporalLcmSampler(threeSteps(), {{1.0f}, {1.0f}}, {}), std::runtime_error);
    EXPECT_THROW(TemporalLcmSampler(threeSteps(), {{1.0f}, {1.0f, 2.0f}, {1.0f}}, {}), std::runtime_error);
    EXPECT_THROW(TemporalLcmSampler(threeSteps(), threeNoises(), {0, 1.0f, false}), std::runtime_error);
    EXPECT_THROW(TemporalLcmSampler(threeSteps(), threeNoises(), {3, 1.0f, false}), std::runtime_error);
    EXPECT_NO_THROW(TemporalLcmSampler(threeSteps(), threeNoises(), {2, 1.0f, false}));
}

TEST(TemporalLcmSampler, keyframeRunsAllStepsFromNoise) {
    TemporalLcmSampler sampler(threeSteps(), threeNoises(), {1, 1.0f, false});
    EXPECT_EQ(sampler.evaluationsOfNextFrame(), 5u);
    RecordingPredictor predictor;
    sampler.sampleFrame(std::ref(predictor));

    ASSERT_EQ(predictor.timesteps, (std::vector<int64_t>{999, 666, 333, 666, 333}));
    expectNear(predictor.inputs[0], threeNoises()[0]);
    expectNear(sampler.getKeyframe(), runByHand(0, threeNoises()[0], 0.0f));
    EXPECT_EQ(sampler.framesSampled(), 1u);
}

TEST(TemporalLcmSampler, framesAreImg2ImgOfTheKeyframe) {
    TemporalLcmSampler sampler(threeSteps(), threeNoises(), {1, 1.0f, false});
    RecordingPredictor first;
    const auto frame0 = sampler.sampleFrame(std::ref(first));
    const auto keyframe = sampler.getKeyframe();
    const auto start = threeSteps().addNoise(1, keyframe, threeNoises()[1]);
    expectNear(first.inputs[3], start);
    expectNear(frame0, runByHand(1, start, 0.0f));

    EXPECT_EQ(sampler.evaluationsOfNextFrame(), 2u);
    RecordingPredictor second;
    second.offset = 0.1f;
    const auto frame1 = sampler.sampleFrame(std::ref(second));
    EXPECT_EQ(second.timesteps, (std::vector<int64_t>{666, 333}));
    expectNear(second.inputs[0], start);
    expectNear(frame1, runByHand(1, start, 0.1f));
}

TEST(TemporalLcmSampler, anchorWeightMixesPreviousFrameAndKeyframe) {
    const float anchor = 0.25f;
    TemporalLcmSampler sampler(threeSteps(), threeNoises(), {1, anchor, false});
    RecordingPredictor first;
    const auto frame0 = sampler.sampleFrame(std::ref(first));
    const auto keyframe = sampler.getKeyframe();
    RecordingPredictor second;
    sampler.sampleFrame(std::ref(second));
    std::vector<float> mix(keyframe.size());
    for (size_t i = 0; i < mix.size(); ++i) {
        mix[i] = (1.0f - anchor) * frame0[i] + anchor * keyframe[i];
    }
    expectNear(second.inputs[0], threeSteps().addNoise(1, mix, threeNoises()[1]));
}

TEST(TemporalLcmSampler, independentFramesRepeatTheFullChain) {
    TemporalLcmSampler sampler(threeSteps(), threeNoises(), {1, 1.0f, true});
    EXPECT_EQ(sampler.evaluationsOfNextFrame(), 3u);
    RecordingPredictor first, second;
    const auto frame0 = sampler.sampleFrame(std::ref(first));
    const auto frame1 = sampler.sampleFrame(std::ref(second));
    EXPECT_EQ(second.timesteps, first.timesteps);
    expectNear(frame0, runByHand(0, threeNoises()[0], 0.0f));
    expectNear(frame1, frame0);
    EXPECT_TRUE(sampler.getKeyframe().empty());
}
