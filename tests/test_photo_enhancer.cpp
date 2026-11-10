// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - generatorMatchesPyTorch: the photo enhancer's generator exported by scripts/neural/train_photo_enhancer.py runs in
 *   Klartraum and reproduces the PyTorch image of its stored reference input (11 G-buffer channels, and with a
 *   surface texture its features after them)
 *
 * The graph and its reference are written by the training into build/TestingOutput/gan_runs/v6/ (not in the
 * repository); the test skips without them.
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

const std::string kModel = "./build/TestingOutput/gan_runs/v6/photo_enhancer";

std::vector<float> readFloats(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    std::vector<float> data(static_cast<size_t>(file.tellg()) / sizeof(float));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
    return data;
}

} // namespace

TEST(PhotoEnhancerTest, generatorMatchesPyTorch) {
    if (!std::filesystem::exists(kModel + ".onnx") || !std::filesystem::exists(kModel + "_input_f32.bin"))
        GTEST_SKIP() << "train scripts/neural/train_photo_enhancer.py first";
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    auto network = vc.create<OnnxNetwork>(kModel + ".onnx");
    network->retainTensor("image");
    const auto reference = readFloats(kModel + "_input_f32.bin");
    const auto channels = uint32_t(reference.size() / (464 * 832));
    auto input = vc.create<TensorElement<float>>(std::vector<uint32_t>{1, channels, 464, 832});
    network->setInputTensor("input", input, -1);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(network);
    input->getDataBuffer(0).memcopyFrom(reference);
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    auto output = std::dynamic_pointer_cast<TensorElement<float>>(network->getOutputElement("image"));
    ASSERT_NE(output, nullptr);
    std::vector<float> actual(output->getDataElementCount());
    output->getDataBuffer(0).memcopyTo(actual);
    const auto expected = readFloats(kModel + "_output_f32.bin");
    ASSERT_EQ(actual.size(), expected.size());
    float worst = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i)
        worst = std::max(worst, std::abs(actual[i] - expected[i]));
    std::cout << "largest difference " << worst << " (image values in [0, 1])" << std::endl;
    EXPECT_LT(worst, 1e-3f);
}
