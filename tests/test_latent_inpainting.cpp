// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - latentMaskCoversMaskedBlocks: a latent pixel is masked when any pixel of its 8 x 8 block is, and the dilation
 *   adds its neighbours
 * - emptyMaskKeepsTheKnownLatent: with nothing masked, inpainting returns the known latent and never matters what
 *   the predictor says
 * - fullMaskIsPlainSampling: with everything masked, inpainting equals stepping the scheduler by hand from the noise
 * - maskedRegionIsGeneratedRestIsKept: with half the latent masked, the unmasked half is the known latent and the
 *   masked half the generated one; the predictor sees the known latent noised to each step outside the mask
 * - noiseFieldIsDeterministicAndShifts: a noise field gives the same windows for the same seed and offset, a window
 *   shifted by (dx, dy) holds the same values shifted, windows wrap around the field, and the values are standard
 *   normal
 **/

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#include "klartraum/sd15/latent_inpainting.hpp"

using namespace klartraum;

namespace {

LcmScheduler threeSteps() {
    return LcmScheduler(
        {{999, 0.1f, 0.4f, 0.2f, 0.8f}, {666, 0.4f, 0.7f, 0.05f, 0.95f}, {333, 0.7f, 0.99f, 0.0f, 1.0f}});
}

// Noise for 3 steps of a 1-channel 2 x 2 latent.
std::vector<std::vector<float>> threeNoises() {
    return {{1.0f, -1.0f, 0.5f, 0.25f}, {0.5f, 0.25f, -0.75f, 2.0f}, {-0.75f, 2.0f, 1.5f, -0.5f}};
}

struct RecordingPredictor {
    std::vector<std::vector<float>> inputs;
    std::vector<float> operator()(const std::vector<float>& latents, int64_t timestep) {
        inputs.push_back(latents);
        std::vector<float> result(latents.size());
        for (size_t i = 0; i < latents.size(); ++i) {
            result[i] = 0.3f * latents[i] + 0.001f * float(timestep) + 0.2f;
        }
        return result;
    }
};

void expectNear(const std::vector<float>& actual, const std::vector<float>& expected) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], 1e-5f) << "element " << i;
    }
}

} // namespace

TEST(LatentInpainting, latentMaskCoversMaskedBlocks) {
    // A 32 x 16 pixel mask with one set pixel at (19, 2): block (2, 0) of a 4 x 2 latent.
    std::vector<uint8_t> pixels(32 * 16, 0);
    pixels[2 * 32 + 19] = 255;
    const auto undilated = latentMask(pixels, 32, 16, 8, 0);
    ASSERT_EQ(undilated.size(), 8u);
    EXPECT_EQ(undilated, (std::vector<float>{0, 0, 1, 0, 0, 0, 0, 0}));
    const auto dilated = latentMask(pixels, 32, 16, 8, 1);
    EXPECT_EQ(dilated, (std::vector<float>{0, 1, 1, 1, 0, 1, 1, 1}));
}

TEST(LatentInpainting, emptyMaskKeepsTheKnownLatent) {
    const std::vector<float> known{0.3f, -0.2f, 0.7f, 0.1f};
    RecordingPredictor predictor;
    const auto result =
        sampleInpainted(threeSteps(), threeNoises(), {std::vector<float>(4, 0.0f), known}, std::ref(predictor), 1);
    expectNear(result, known);
    EXPECT_EQ(predictor.inputs.size(), 3u);
}

TEST(LatentInpainting, fullMaskIsPlainSampling) {
    RecordingPredictor predictor, reference;
    const auto result =
        sampleInpainted(threeSteps(), threeNoises(), {std::vector<float>(4, 1.0f), std::vector<float>(4, 5.0f)},
                        std::ref(predictor), 1);
    const auto scheduler = threeSteps();
    std::vector<float> latents = threeNoises()[0], denoised;
    for (size_t step = 0; step < 3; ++step) {
        const auto prediction = reference(latents, scheduler.getSteps()[step].timestep);
        scheduler.step(step, latents, prediction, step + 1 < 3 ? threeNoises()[step + 1] : std::vector<float>{},
                       denoised);
    }
    expectNear(result, latents);
}

TEST(LatentInpainting, maskedRegionIsGeneratedRestIsKept) {
    // Two channels of 2 x 1 pixels; the mask covers pixel 0 of both channels.
    const std::vector<float> mask{1.0f, 0.0f};
    const std::vector<float> known{0.3f, -0.2f, 0.7f, 0.1f};
    RecordingPredictor predictor;
    const auto noise = threeNoises();
    const auto result = sampleInpainted(threeSteps(), noise, {mask, known}, std::ref(predictor), 2);
    // Pixel 1 of each channel (elements 1 and 3) is kept.
    EXPECT_NEAR(result[1], known[1], 1e-6f);
    EXPECT_NEAR(result[3], known[3], 1e-6f);
    EXPECT_GT(std::abs(result[0] - known[0]), 1e-3f);
    // Before each step, pixel 1 is the known latent noised to that step.
    const auto scheduler = threeSteps();
    for (size_t step = 0; step < 3; ++step) {
        const auto noised = scheduler.addNoise(step, known, noise[step]);
        EXPECT_NEAR(predictor.inputs[step][1], noised[1], 1e-5f) << "step " << step;
        EXPECT_NEAR(predictor.inputs[step][3], noised[3], 1e-5f) << "step " << step;
    }
    // The masked pixel starts from the noise.
    EXPECT_NEAR(predictor.inputs[0][0], noise[0][0], 1e-5f);
}

TEST(LatentInpainting, noiseFieldIsDeterministicAndShifts) {
    const NoiseField field(2, 4, 64, 48, 7);
    const auto a = field.window(5, 3, 16, 8);
    ASSERT_EQ(a.size(), 2u);
    ASSERT_EQ(a[0].size(), size_t(4 * 16 * 8));
    EXPECT_EQ(a, NoiseField(2, 4, 64, 48, 7).window(5, 3, 16, 8));
    EXPECT_NE(a, NoiseField(2, 4, 64, 48, 8).window(5, 3, 16, 8));

    // Shifted by (2, 1): window value (x, y) equals the original's (x + 2, y + 1).
    const auto shifted = field.window(7, 4, 16, 8);
    for (size_t step = 0; step < 2; ++step) {
        for (size_t channel = 0; channel < 4; ++channel) {
            for (size_t y = 0; y + 1 < 8; ++y) {
                for (size_t x = 0; x + 2 < 16; ++x) {
                    ASSERT_EQ(shifted[step][(channel * 8 + y) * 16 + x], a[step][(channel * 8 + y + 1) * 16 + x + 2]);
                }
            }
        }
    }
    // Windows wrap around the field.
    EXPECT_EQ(field.window(-64, -48, 16, 8), field.window(0, 0, 16, 8));

    const auto whole = field.window(0, 0, 64, 48);
    double sum = 0.0, squares = 0.0;
    for (float value : whole[0]) {
        sum += value;
        squares += double(value) * value;
    }
    const double mean = sum / double(whole[0].size());
    EXPECT_NEAR(mean, 0.0, 0.05);
    EXPECT_NEAR(squares / double(whole[0].size()) - mean * mean, 1.0, 0.05);
}
