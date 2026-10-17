// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - promptEncodesLikeTheExport: setPrompt() with the exported prompt gives the prompt embedding the ControlNet
 *   export computed in Python
 * - characterStepsDrawPoseMaps: each controller step draws an OpenPose pose map into the denoiser's pose tensor:
 *   it holds the joint colours, the background stays black, and walking changes it from step to step
 * - keyframeSamplingReducesFlicker: six frames of a walking character sampled as img2img of a keyframe flicker less
 *   than half as much as frames sampled independently from the shared noise, change their overall colour less, and
 *   need fewer ControlNet and UNet evaluations
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "klartraum/headless_frontend.hpp"
#include "klartraum/sd15/pose_video_pipeline.hpp"

using namespace klartraum;

namespace {

constexpr uint32_t kSize = 256;
const std::filesystem::path kControlDirectory = "data/onnx/sd15_controlnet_256";
const std::filesystem::path kSdDirectory = "data/onnx/sd15_denoiser_256";

std::vector<float> readFloats(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    std::vector<float> values(static_cast<size_t>(input.tellg()) / sizeof(float));
    input.seekg(0);
    input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    return values;
}

std::string exportedPrompt() {
    std::ifstream input(kControlDirectory / "prompt.txt");
    std::string prompt;
    std::getline(input, prompt);
    return prompt;
}

double meanAbsoluteDifference(const std::vector<float>& a, const std::vector<float>& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += std::abs(a[i] - b[i]);
    }
    return sum / double(a.size());
}

// Largest change of the mean of a colour channel relative to the first frame.
double colourDrift(const std::vector<std::vector<float>>& frames) {
    const size_t plane = size_t(kSize) * kSize;
    auto mean = [&](const std::vector<float>& frame, size_t channel) {
        double sum = 0.0;
        for (size_t i = channel * plane; i < (channel + 1) * plane; ++i) {
            sum += frame[i];
        }
        return sum / double(plane);
    };
    double drift = 0.0;
    for (const auto& frame : frames) {
        for (size_t channel = 0; channel < 3; ++channel) {
            drift = std::max(drift, std::abs(mean(frame, channel) - mean(frames.front(), channel)));
        }
    }
    return drift;
}

CharacterControl walkForward() {
    CharacterControl control;
    control.velocity = {0.0f, 1.3f};
    control.facing = {0.0f, 1.0f};
    return control;
}

class PoseVideoTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        for (const auto& path : {kControlDirectory / "sd15_unet_control.onnx", kSdDirectory / "sd15_vae_decoder.onnx",
                                 std::filesystem::path("data/character/controller.onnx")}) {
            if (!std::filesystem::exists(path)) {
                missing = path.string();
                return;
            }
        }
        frontend = new HeadlessFrontend();
        pipeline = new PoseVideoPipeline(frontend->getKlartraumEngine().getVulkanContext(),
                                         {"data/character", kSdDirectory, kControlDirectory}, kSize);
    }

    static void TearDownTestSuite() {
        delete pipeline;
        delete frontend;
        pipeline = nullptr;
        frontend = nullptr;
    }

    void SetUp() override {
        if (!pipeline) {
            GTEST_SKIP() << "Missing " << missing
                         << "; run scripts/sd15_onnx (export_denoiser.py and export_controlnet.py, --size 256)";
        }
    }

    // Decoded frames of a character walking forward, two controller steps per frame.
    std::vector<std::vector<float>> walk(const TemporalLcmSampler::Settings& settings, uint32_t frames,
                                         uint64_t& evaluations) {
        pipeline->getController().reset();
        pipeline->setPromptEmbedding(readFloats(kControlDirectory / "prompt_embeddings_f32.bin"));
        auto sampler = pipeline->makeSampler(settings);
        const uint64_t before = pipeline->getEvaluations();
        std::vector<std::vector<float>> images;
        for (uint32_t frame = 0; frame < frames; ++frame) {
            for (int step = 0; step < 2; ++step) {
                const glm::vec3 hips(pipeline->getController().readJoints()[0]);
                pipeline->stepCharacter(walkForward(), PoseVideoPipeline::lookAt({hips.x, 0.9f, hips.z}));
            }
            images.push_back(pipeline->decode(pipeline->sampleFrame(sampler)));
        }
        evaluations = pipeline->getEvaluations() - before;
        return images;
    }

    static inline HeadlessFrontend* frontend = nullptr;
    static inline PoseVideoPipeline* pipeline = nullptr;
    static inline std::string missing;
};

} // namespace

TEST_F(PoseVideoTest, promptEncodesLikeTheExport) {
    pipeline->setPrompt(exportedPrompt());
    const auto embedding = pipeline->getPromptEmbedding();
    const auto reference = readFloats(kControlDirectory / "prompt_embeddings_f32.bin");
    ASSERT_EQ(embedding.size(), reference.size());
    float error = 0.0f;
    for (size_t i = 0; i < embedding.size(); ++i) {
        error = std::max(error, std::abs(embedding[i] - reference[i]));
    }
    EXPECT_LT(error, 2e-2f);
}

TEST_F(PoseVideoTest, characterStepsDrawPoseMaps) {
    pipeline->getController().reset();
    std::vector<float> previous;
    for (int step = 0; step < 4; ++step) {
        pipeline->stepCharacter(walkForward(), PoseVideoPipeline::lookAt({0.0f, 0.9f, 0.0f}));
        const auto pose = pipeline->readPoseMap();
        const size_t plane = size_t(kSize) * kSize;
        size_t drawn = 0, red = 0;
        for (size_t i = 0; i < plane; ++i) {
            const float r = pose[i], g = pose[plane + i], b = pose[2 * plane + i];
            drawn += (r + g + b) > 0.0f ? 1 : 0;
            // The nose joint is pure red (255, 0, 0).
            red += (r == 1.0f && g == 0.0f && b == 0.0f) ? 1 : 0;
        }
        EXPECT_GT(drawn, 300u) << "step " << step;
        EXPECT_LT(drawn, plane / 4) << "step " << step;
        EXPECT_GT(red, 0u) << "step " << step;
        if (!previous.empty()) {
            EXPECT_GT(meanAbsoluteDifference(pose, previous), 0.0) << "step " << step;
        }
        previous = pose;
    }
}

TEST_F(PoseVideoTest, keyframeSamplingReducesFlicker) {
    constexpr uint32_t kFrames = 6;
    auto flicker = [](const std::vector<std::vector<float>>& frames) {
        double sum = 0.0;
        for (size_t i = 1; i < frames.size(); ++i) {
            sum += meanAbsoluteDifference(frames[i], frames[i - 1]);
        }
        return sum / double(frames.size() - 1);
    };
    uint64_t independentEvaluations = 0, keyframeEvaluations = 0;
    const auto independent = walk({1, 1.0f, true}, kFrames, independentEvaluations);
    const auto keyframe = walk({1, 1.0f, false}, kFrames, keyframeEvaluations);

    const double independentFlicker = flicker(independent), keyframeFlicker = flicker(keyframe);
    std::cout << "flicker (mean |difference| of consecutive frames, of 2): independent " << independentFlicker
              << ", keyframe " << keyframeFlicker << "; colour drift: independent " << colourDrift(independent)
              << ", keyframe " << colourDrift(keyframe) << std::endl;
    EXPECT_LT(keyframeFlicker, 0.5 * independentFlicker);
    EXPECT_LT(colourDrift(keyframe), colourDrift(independent));
    EXPECT_EQ(independentEvaluations, 4u * kFrames);
    EXPECT_EQ(keyframeEvaluations, 4u + 3u * kFrames);
    for (const auto& frame : keyframe) {
        EXPECT_GT(*std::max_element(frame.begin(), frame.end()), *std::min_element(frame.begin(), frame.end()) + 0.5f);
    }
}
