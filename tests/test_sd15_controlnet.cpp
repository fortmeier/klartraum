// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - lcmStepFollowsTheConsistencyUpdate: one LcmScheduler step predicts the clean latent, applies c_skip and c_out,
 *   and noises the result to the next timestep; the last step returns the denoised latent
 * - lcmAddNoiseMixesCleanLatentAndNoise: addNoise() scales clean latent and noise by sqrt(alpha) and
 *   sqrt(1 - alpha) of the step's timestep
 * - exportedLcmScheduleIsValid: the exported schedule has decreasing timesteps, alphas in (0, 1) that grow from
 *   step to step, and ends with c_skip close to 0 and c_out close to 1
 * - controlNetMatchesOnnxRuntime: the ControlNet residuals of the first step match ONNX Runtime
 * - controlledUnetMatchesOnnxRuntime: ControlNet and UNet in one graph (the residuals passed directly) give the
 *   UNet output of the first step that ONNX Runtime computes
 * - lcmPipelineMatchesReference: all LCM steps with ControlNet, run in Klartraum with the exported noise, end at the
 *   final latents of the Python reference
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/sd15/controlnet_denoiser.hpp"
#include "klartraum/sd15/lcm_scheduler.hpp"

using namespace klartraum;

namespace {

const std::filesystem::path kModelDirectory = "data/onnx/sd15_controlnet_256";
constexpr uint32_t kLatentSize = 32;
constexpr size_t kLatentElements = 4 * kLatentSize * kLatentSize;

template <typename T>
std::vector<T> readValues(const std::string& name) {
    const auto path = kModelDirectory / name;
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("Could not open " + path.string());
    }
    std::vector<T> values(static_cast<size_t>(input.tellg()) / sizeof(T));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(T)));
    return values;
}

// Largest absolute difference relative to the largest reference magnitude.
float relativeError(const float* actual, const float* expected, size_t count) {
    float difference = 0.0f, magnitude = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(actual[i])) {
            return std::numeric_limits<float>::infinity();
        }
        difference = std::max(difference, std::abs(actual[i] - expected[i]));
        magnitude = std::max(magnitude, std::abs(expected[i]));
    }
    return difference / std::max(magnitude, 1e-6f);
}

std::vector<float> slice(const std::vector<float>& values, size_t index, size_t count) {
    return std::vector<float>(values.begin() + index * count, values.begin() + (index + 1) * count);
}

// One HeadlessFrontend and ControlNetDenoiser for all tests that need the exported models: loading them takes time.
class ControlNetTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        if (!std::filesystem::exists(kModelDirectory / "sd15_unet_control.onnx")) {
            return;
        }
        frontend = new HeadlessFrontend();
        auto& vc = frontend->getKlartraumEngine().getVulkanContext();
        denoiser = new std::shared_ptr<ControlNetDenoiser>(
            std::make_shared<ControlNetDenoiser>(vc, kModelDirectory, kLatentSize));
        graph = new ComputeGraph(vc, 1);
        graph->compileFrom(*denoiser);
    }

    static void TearDownTestSuite() {
        delete graph;
        delete denoiser;
        delete frontend;
        graph = nullptr;
        denoiser = nullptr;
        frontend = nullptr;
    }

    void SetUp() override {
        if (!frontend) {
            GTEST_SKIP() << "Missing " << (kModelDirectory / "sd15_unet_control.onnx").string()
                         << "; run scripts/sd15_onnx/export_controlnet.py (see its README)";
        }
    }

    // Evaluates ControlNet and UNet for `latents` at `timestep` with the exported prompt and pose.
    std::vector<float> evaluate(const std::vector<float>& latents, int64_t timestep) {
        auto& vc = frontend->getKlartraumEngine().getVulkanContext();
        (*denoiser)->getSample()->setData(0, latents);
        (*denoiser)->getTimestep()->setData(0, std::vector<int64_t>{timestep});
        (*denoiser)->getEmbeddings()->setData(0, readValues<float>("prompt_embeddings_f32.bin"));
        (*denoiser)->getPose()->setData(0, readValues<float>("pose_reference_f32.bin"));
        graph->submitAndWait(vc.getGraphicsQueue(), 0);
        std::vector<float> result(kLatentElements);
        (*denoiser)->getNoisePrediction()->getDataBuffer(0).memcopyTo(result);
        return result;
    }

    static HeadlessFrontend* frontend;
    static std::shared_ptr<ControlNetDenoiser>* denoiser;
    static ComputeGraph* graph;
};

HeadlessFrontend* ControlNetTest::frontend = nullptr;
std::shared_ptr<ControlNetDenoiser>* ControlNetTest::denoiser = nullptr;
ComputeGraph* ControlNetTest::graph = nullptr;

} // namespace

TEST(LcmScheduler, lcmStepFollowsTheConsistencyUpdate) {
    const LcmScheduler scheduler({{999, 0.25f, 0.81f, 0.1f, 0.9f}, {499, 0.81f, 0.9991f, 0.0f, 1.0f}});
    std::vector<float> latents{2.0f, -1.0f};
    const std::vector<float> prediction{0.5f, 1.0f};
    const std::vector<float> noise{1.0f, -2.0f};
    std::vector<float> denoised;
    scheduler.step(0, latents, prediction, noise, denoised);
    for (size_t i = 0; i < 2; ++i) {
        const float x = i == 0 ? 2.0f : -1.0f;
        const float original = (x - std::sqrt(0.75f) * prediction[i]) / 0.5f;
        const float expectedDenoised = 0.9f * original + 0.1f * x;
        EXPECT_NEAR(denoised[i], expectedDenoised, 1e-6f);
        EXPECT_NEAR(latents[i], 0.9f * expectedDenoised + std::sqrt(0.19f) * noise[i], 1e-6f);
    }
    scheduler.step(1, latents, prediction, {}, denoised);
    EXPECT_EQ(latents, denoised);
}

TEST(LcmScheduler, lcmAddNoiseMixesCleanLatentAndNoise) {
    const LcmScheduler scheduler({{999, 0.36f, 0.5f, 0.0f, 1.0f}});
    const auto noised = scheduler.addNoise(0, {1.0f, 2.0f}, {-1.0f, 0.5f});
    EXPECT_NEAR(noised[0], 0.6f * 1.0f + 0.8f * -1.0f, 1e-6f);
    EXPECT_NEAR(noised[1], 0.6f * 2.0f + 0.8f * 0.5f, 1e-6f);
}

TEST_F(ControlNetTest, exportedLcmScheduleIsValid) {
    const auto scheduler = LcmScheduler::load(kModelDirectory / "lcm_schedule_f32.bin");
    const auto& steps = scheduler.getSteps();
    ASSERT_GE(steps.size(), 2u);
    for (size_t i = 0; i < steps.size(); ++i) {
        EXPECT_GT(steps[i].alpha, 0.0f);
        EXPECT_LT(steps[i].alpha, 1.0f);
        EXPECT_GT(steps[i].alphaNext, steps[i].alpha);
        if (i > 0) {
            EXPECT_LT(steps[i].timestep, steps[i - 1].timestep);
        }
    }
    EXPECT_LT(steps.back().cSkip, 0.01f);
    EXPECT_GT(steps.back().cOut, 0.99f);
}

TEST_F(ControlNetTest, controlNetMatchesOnnxRuntime) {
    const auto noise = readValues<float>("lcm_noise_f32.bin");
    const auto timesteps = readValues<int64_t>("lcm_timesteps_i64.bin");
    evaluate(slice(noise, 0, kLatentElements), timesteps[0]);
    const auto reference = readValues<float>("controlnet_residuals_f32.bin");
    size_t offset = 0;
    for (uint32_t index = 0; index < ControlNetDenoiser::kResidualCount; ++index) {
        auto residual = std::dynamic_pointer_cast<TensorElement<float>>((*denoiser)->getControlNet()->getOutputElement(
            index + 1 < ControlNetDenoiser::kResidualCount ? "down_residual_" + std::to_string(index)
                                                           : "mid_residual"));
        ASSERT_TRUE(residual) << index;
        std::vector<float> values(residual->getDataElementCount());
        residual->getDataBuffer(0).memcopyTo(values);
        ASSERT_LE(offset + values.size(), reference.size());
        EXPECT_LT(relativeError(values.data(), reference.data() + offset, values.size()), 2e-3f)
            << "residual " << index;
        offset += values.size();
    }
    EXPECT_EQ(offset, reference.size());
}

TEST_F(ControlNetTest, controlledUnetMatchesOnnxRuntime) {
    const auto noise = readValues<float>("lcm_noise_f32.bin");
    const auto timesteps = readValues<int64_t>("lcm_timesteps_i64.bin");
    const auto result = evaluate(slice(noise, 0, kLatentElements), timesteps[0]);
    const auto reference = readValues<float>("unet_reference_f32.bin");
    ASSERT_EQ(reference.size(), kLatentElements);
    EXPECT_LT(relativeError(result.data(), reference.data(), kLatentElements), 5e-3f);
}

TEST_F(ControlNetTest, lcmPipelineMatchesReference) {
    const auto scheduler = LcmScheduler::load(kModelDirectory / "lcm_schedule_f32.bin");
    const auto noise = readValues<float>("lcm_noise_f32.bin");
    std::vector<float> latents = slice(noise, 0, kLatentElements);
    std::vector<float> denoised;
    for (size_t step = 0; step < scheduler.size(); ++step) {
        const auto prediction = evaluate(latents, scheduler.getSteps()[step].timestep);
        const bool last = step + 1 == scheduler.size();
        scheduler.step(step, latents, prediction, last ? std::vector<float>{} : slice(noise, step + 1, kLatentElements),
                       denoised);
    }
    const auto reference = readValues<float>("final_latents_f32.bin");
    ASSERT_EQ(reference.size(), kLatentElements);
    EXPECT_LT(relativeError(latents.data(), reference.data(), kLatentElements), 2e-2f);
}
