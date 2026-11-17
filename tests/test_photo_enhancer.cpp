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
 * - inputsKeepAConstantColour: neural::photoEnhancerInputs() of a frame of one colour gives that colour in the RGB
 *   channels of every sample pixel, at a sample of another size and aspect (scaled, then cropped)
 * - inputsDecodeClassesAndCoordinates: classes become one-hot channels and class indices (the nearest class colour),
 *   u, v and the object id are decoded from their 16-bit encodings, each from the frame pixel nearest-neighbour
 *   sampling takes
 * - inputsStretchTheInverseDepth: the covered pixels' inverse depth runs from 0 (2nd percentile) to 1 (depthHigh 1:
 *   the nearest pixel), the background stays 0
 * - spearmanRanksCorrelation: the rank correlation of a series with a monotonic function of it is 1, with a falling
 *   one -1
 * - photoAlignmentMatchesPython: neural::PhotoAlignment scores the first crate photographs against their rendered
 *   depth like scripts/neural/check_photo_alignment.py (alignment.json) within 0.05 (skips without the crate data or
 *   the depth network)
 * - appearanceGeneratorMatchesPyTorch: a generator with per-image appearance codes (--appearance, an input
 *   "appearance" of the code) reproduces the PyTorch image of its reference input and code
 *
 * The graphs and their references are written by the training into build/TestingOutput/gan_runs/v6/ and v7/ and
 * build/TestingOutput/crate/runs/c2/ (not in the repository); the tests skip without them.
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/neural/photo_alignment.hpp"
#include "klartraum/neural/photo_enhancer.hpp"
#include "klartraum/tree/tree_model.hpp"
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

// Runs the generator `stem`.onnx on its reference input (and season and appearance code, where it has those inputs)
// and returns the largest difference to the PyTorch image.
float runAgainstReference(const std::string& stem, uint32_t height = 464, uint32_t width = 832) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    auto network = vc.create<OnnxNetwork>(stem + ".onnx");
    network->retainTensor("image");
    const auto reference = readFloats(stem + "_input_f32.bin");
    const auto channels = uint32_t(reference.size() / (height * width));
    auto input = vc.create<TensorElement<float>>(std::vector<uint32_t>{1, channels, height, width});
    network->setInputTensor("input", input, -1);
    std::shared_ptr<TensorElement<float>> season;
    const bool seasonal = std::filesystem::exists(stem + "_season_f32.bin");
    if (seasonal) {
        season = vc.create<TensorElement<float>>(std::vector<uint32_t>{1, 2});
        network->setInputTensor("season", season, -1);
    }
    std::shared_ptr<TensorElement<float>> appearance;
    const bool coded = std::filesystem::exists(stem + "_appearance_input_f32.bin");
    const auto code = coded ? readFloats(stem + "_appearance_input_f32.bin") : std::vector<float>{};
    if (coded) {
        appearance = vc.create<TensorElement<float>>(std::vector<uint32_t>{1, uint32_t(code.size())});
        network->setInputTensor("appearance", appearance, -1);
    }
    ComputeGraph graph(vc, 1);
    graph.compileFrom(network);
    input->getDataBuffer(0).memcopyFrom(reference);
    if (seasonal)
        season->getDataBuffer(0).memcopyFrom(readFloats(stem + "_season_f32.bin"));
    if (coded)
        appearance->getDataBuffer(0).memcopyFrom(code);
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

namespace {

// A frame of w x h pixels: every G-buffer black, the depth window 0..10.
neural::GBufferFrame blankFrame(uint32_t w, uint32_t h) {
    neural::GBufferFrame frame;
    frame.width = w;
    frame.height = h;
    const std::vector<uint8_t> black(size_t(w) * h * 4, 0);
    frame.colour = frame.classes = frame.depth16 = frame.u = frame.v = black;
    frame.depthFar = 10.0f;
    return frame;
}

void setPixel(std::vector<uint8_t>& bgra, size_t p, glm::vec3 rgb) {
    bgra[4 * p + 2] = uint8_t(std::lround(rgb.r * 255.0f));
    bgra[4 * p + 1] = uint8_t(std::lround(rgb.g * 255.0f));
    bgra[4 * p] = uint8_t(std::lround(rgb.b * 255.0f));
}

void set16(std::vector<uint8_t>& bgra, size_t p, float value, uint8_t blue) {
    const uint32_t q = uint32_t(std::lround(value * 65535.0f));
    bgra[4 * p + 2] = uint8_t(q >> 8u);
    bgra[4 * p + 1] = uint8_t(q & 255u);
    bgra[4 * p] = blue;
}

} // namespace

TEST(PhotoEnhancerTest, inputsKeepAConstantColour) {
    auto frame = blankFrame(64, 36);
    for (size_t p = 0; p < 64 * 36; ++p)
        setPixel(frame.colour, p, glm::vec3(0.2f, 0.4f, 0.8f));
    const auto inputs = neural::photoEnhancerInputs(frame, neural::PhotoEnhancerSample{32, 16});
    ASSERT_EQ(inputs.channels.size(), 11u * 32 * 16);
    for (size_t p = 0; p < 32 * 16; ++p) {
        EXPECT_NEAR(inputs.channels[p], 51.0f / 255.0f, 1e-6f);
        EXPECT_NEAR(inputs.channels[32 * 16 + p], 102.0f / 255.0f, 1e-6f);
        EXPECT_NEAR(inputs.channels[2 * 32 * 16 + p], 204.0f / 255.0f, 1e-6f);
    }
}

TEST(PhotoEnhancerTest, inputsDecodeClassesAndCoordinates) {
    // The left half stem, the right half soil; u = x / w, v = y / h, the object id 0x1234.
    auto frame = blankFrame(64, 64);
    for (uint32_t y = 0; y < 64; ++y)
        for (uint32_t x = 0; x < 64; ++x) {
            const size_t p = size_t(y) * 64 + x;
            setPixel(frame.classes, p, tree::treeClassColor(x < 32 ? tree::TREE_CLASS_STEM : tree::TREE_CLASS_SOIL));
            set16(frame.u, p, float(x) / 64.0f, 0x12);
            set16(frame.v, p, float(y) / 64.0f, 0x34);
        }
    const auto inputs = neural::photoEnhancerInputs(frame, neural::PhotoEnhancerSample{32, 32});
    const size_t pixels = 32 * 32;
    for (uint32_t y = 0; y < 32; ++y)
        for (uint32_t x = 0; x < 32; ++x) {
            const size_t p = size_t(y) * 32 + x;
            const uint32_t expected = x < 16 ? tree::TREE_CLASS_STEM : tree::TREE_CLASS_SOIL;
            EXPECT_EQ(inputs.classes[p], float(expected));
            for (uint32_t c = 0; c < 7; ++c)
                EXPECT_EQ(inputs.channels[(4 + c) * pixels + p], c == expected ? 1.0f : 0.0f);
            // Nearest sampling takes frame pixel 2 x + 1.
            EXPECT_NEAR(inputs.coordinates[p], float(2 * x + 1) / 64.0f, 1e-4f);
            EXPECT_NEAR(inputs.coordinates[pixels + p], float(2 * y + 1) / 64.0f, 1e-4f);
            EXPECT_NEAR(inputs.coordinates[2 * pixels + p], float(0x1234) / 65535.0f, 1e-7f);
        }
}

TEST(PhotoEnhancerTest, inputsStretchTheInverseDepth) {
    // A covered square in the middle at distances 2 (top rows) to 6 (bottom rows); the rest background.
    auto frame = blankFrame(32, 32);
    for (uint32_t y = 8; y < 24; ++y)
        for (uint32_t x = 8; x < 24; ++x) {
            const size_t p = size_t(y) * 32 + x;
            setPixel(frame.classes, p, tree::treeClassColor(tree::TREE_CLASS_STEM));
            set16(frame.depth16, p, (2.0f + 4.0f * float(y - 8) / 15.0f) / frame.depthFar, 0);
        }
    neural::PhotoEnhancerSample sample{32, 32, 1.0f};
    const auto inputs = neural::photoEnhancerInputs(frame, sample);
    const float* depth = inputs.channels.data() + 3 * 32 * 32;
    EXPECT_NEAR(depth[8 * 32 + 16], 1.0f, 1e-3f); // the nearest row
    EXPECT_NEAR(depth[23 * 32 + 16], 0.0f, 1e-3f);
    EXPECT_GT(depth[12 * 32 + 16], depth[16 * 32 + 16]);
    EXPECT_EQ(depth[2 * 32 + 2], 0.0f);
}

TEST(PhotoEnhancerTest, appearanceGeneratorMatchesPyTorch) {
    const std::string stem = "./build/TestingOutput/crate/runs/c2/photo_enhancer";
    if (!std::filesystem::exists(stem + ".onnx") || !std::filesystem::exists(stem + "_appearance_input_f32.bin"))
        GTEST_SKIP() << "train scripts/neural/train_photo_enhancer.py --appearance first";
    EXPECT_LT(runAgainstReference(stem, 256, 256), 1e-3f);
}

TEST(PhotoEnhancerTest, spearmanRanksCorrelation) {
    const std::vector<float> a{0.3f, 1.5f, -2.0f, 0.9f, 4.0f};
    std::vector<float> rising, falling;
    for (float v : a) {
        rising.push_back(std::exp(v));
        falling.push_back(-v * v * v);
    }
    EXPECT_NEAR(neural::spearman(a, rising), 1.0f, 1e-6f);
    EXPECT_NEAR(neural::spearman(a, falling), -1.0f, 1e-6f);
}

TEST(PhotoEnhancerTest, photoAlignmentMatchesPython) {
    const std::filesystem::path data = "build/TestingOutput/crate/data/train";
    const std::string model = "build/models/depth/onnx/depth_anything_v2_metric_outdoor_small_518x924.onnx";
    if (!std::filesystem::exists(data / "alignment.json") || !std::filesystem::exists(model))
        GTEST_SKIP() << "needs the crate data (crate_gbuffer_example, Qwen photos, check_photo_alignment.py)";
    // The scores of alignment.json: "id": score.
    std::map<std::string, float> expected;
    {
        std::ifstream file(data / "alignment.json");
        for (std::string line; std::getline(file, line);) {
            const size_t a = line.find('"'), b = line.find('"', a + 1), colon = line.find(':', b);
            if (a != std::string::npos && b != std::string::npos && colon != std::string::npos)
                expected[line.substr(a + 1, b - a - 1)] = std::stof(line.substr(colon + 1));
        }
    }
    // A binary PPM as planar [3][h][w] (or its first channel as [h][w]) in [0, 1].
    const auto readPpm = [](const std::filesystem::path& path, bool grey) {
        std::ifstream file(path, std::ios::binary);
        std::string magic;
        uint32_t w = 0, h = 0, maximum = 0;
        file >> magic >> w >> h >> maximum;
        file.get();
        std::vector<uint8_t> bytes(size_t(w) * h * 3);
        file.read(reinterpret_cast<char*>(bytes.data()), std::streamsize(bytes.size()));
        const size_t pixels = size_t(w) * h;
        std::vector<float> planar(grey ? pixels : 3 * pixels);
        for (size_t p = 0; p < pixels; ++p)
            for (size_t c = 0; c < (grey ? 1u : 3u); ++c)
                planar[c * pixels + p] = bytes[3 * p + c] / 255.0f;
        return planar;
    };
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    neural::PhotoAlignment alignment(vc, model);
    int checked = 0;
    float worst = 0.0f;
    for (const auto& [id, score] : expected) {
        if (checked == 10)
            break;
        const auto photo = readPpm(data / "photo" / (id + ".ppm"), false);
        const auto depth = readPpm(data / "depth" / (id + ".ppm"), true);
        const float actual = alignment.score(photo, depth, 256, 256);
        std::cout << id << ": " << actual << " (Python " << score << ")" << std::endl;
        worst = std::max(worst, std::abs(actual - score));
        ++checked;
    }
    EXPECT_LT(worst, 0.05f);
}
