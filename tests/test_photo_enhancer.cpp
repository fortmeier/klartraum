// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - generatorMatchesPyTorch: the photo enhancer's generator exported by scripts/neural/train_photo_enhancer.py runs in
 *   Klartraum and reproduces the PyTorch image of its stored reference input (11 G-buffer channels, and with a
 *   surface texture its features after them)
 * - seasonGeneratorMatchesPyTorch: a generator conditioned on the season (--season-film, a second input "season" of
 *   the season's code) reproduces the PyTorch image of its reference input and season
 *
 * The graphs and their references are written by the training into build/TestingOutput/gan_runs/v6/ and v7/ (not in
 * the repository); the tests skip without them.
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/onnx/onnx_network.hpp"

using namespace klartraum;

namespace {

std::vector<float> readFloats(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    std::vector<float> data(static_cast<size_t>(file.tellg()) / sizeof(float));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
    return data;
}

// Runs the generator `stem`.onnx on its reference input (and season, where it has that input) and returns the largest
// difference to the PyTorch image.
float runAgainstReference(const std::string& stem) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    auto network = vc.create<OnnxNetwork>(stem + ".onnx");
    network->retainTensor("image");
    const auto reference = readFloats(stem + "_input_f32.bin");
    const auto channels = uint32_t(reference.size() / (464 * 832));
    auto input = vc.create<TensorElement<float>>(std::vector<uint32_t>{1, channels, 464, 832});
    network->setInputTensor("input", input, -1);
    std::shared_ptr<TensorElement<float>> season;
    const bool seasonal = std::filesystem::exists(stem + "_season_f32.bin");
    if (seasonal) {
        season = vc.create<TensorElement<float>>(std::vector<uint32_t>{1, 2});
        network->setInputTensor("season", season, -1);
    }
    ComputeGraph graph(vc, 1);
    graph.compileFrom(network);
    input->getDataBuffer(0).memcopyFrom(reference);
    if (seasonal)
        season->getDataBuffer(0).memcopyFrom(readFloats(stem + "_season_f32.bin"));
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    auto output = std::dynamic_pointer_cast<TensorElement<float>>(network->getOutputElement("image"));
    EXPECT_NE(output, nullptr);
    if (!output)
        return INFINITY;
    std::vector<float> actual(output->getDataElementCount());
    output->getDataBuffer(0).memcopyTo(actual);
    const auto expected = readFloats(stem + "_output_f32.bin");
    EXPECT_EQ(actual.size(), expected.size());
    float worst = 0.0f;
    for (size_t i = 0; i < std::min(actual.size(), expected.size()); ++i)
        worst = std::max(worst, std::abs(actual[i] - expected[i]));
    std::cout << "largest difference " << worst << " (image values in [0, 1])" << std::endl;
    return worst;
}

} // namespace

TEST(PhotoEnhancerTest, generatorMatchesPyTorch) {
    const std::string stem = "./build/TestingOutput/gan_runs/v6/photo_enhancer";
    if (!std::filesystem::exists(stem + ".onnx") || !std::filesystem::exists(stem + "_input_f32.bin"))
        GTEST_SKIP() << "train scripts/neural/train_photo_enhancer.py first";
    EXPECT_LT(runAgainstReference(stem), 1e-3f);
}

TEST(PhotoEnhancerTest, seasonGeneratorMatchesPyTorch) {
    const std::string stem = "./build/TestingOutput/gan_runs/v7/photo_enhancer";
    if (!std::filesystem::exists(stem + ".onnx") || !std::filesystem::exists(stem + "_season_f32.bin"))
        GTEST_SKIP() << "train scripts/neural/train_photo_enhancer.py --season-film first";
    EXPECT_LT(runAgainstReference(stem), 1e-3f);
}
