// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - lookupMatchesReference: the hash texture layer's features of an image of random surface coordinates and classes
 *   equal a CPU implementation of the same multi-resolution hash encoding (direct index on coarse levels, Instant
 *   NGP's spatial hash on fine ones, trilinear blending), each pixel from its class's grid, zeros for classes without
 *   one, for an image size that is not a multiple of the workgroup
 * - lookupMatchesTrainedTexture: with a texture exported by scripts/neural/train_photo_enhancer.py --texture, the
 *   layer reproduces the features PyTorch computed for its reference coordinates (skips without the export)
 **/

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/neural/hash_texture.hpp"

using namespace klartraum;

namespace {

// The features of one point from one grid's table [levels][tableSize][features].
std::vector<float> referenceFeatures(const neural::HashTextureShape& shape, const float* table, const float point[3]) {
    std::vector<float> out(size_t(shape.levels) * shape.features, 0.0f);
    for (uint32_t level = 0; level < shape.levels; ++level) {
        const uint32_t resolution = shape.resolutions[level];
        int cell[3];
        float weight[3];
        for (int a = 0; a < 3; ++a) {
            const float scaled = point[a] * float(resolution);
            cell[a] = std::clamp(int(std::floor(scaled)), 0, int(resolution) - 1);
            weight[a] = scaled - float(cell[a]);
        }
        for (int bits = 0; bits < 8; ++bits) {
            uint32_t corner[3];
            float w = 1.0f;
            for (int a = 0; a < 3; ++a) {
                const uint32_t offset = (bits >> a) & 1;
                corner[a] = uint32_t(cell[a]) + offset;
                w *= offset ? weight[a] : 1.0f - weight[a];
            }
            const uint64_t r = resolution + 1;
            uint64_t index;
            if (r * r * r <= shape.tableSize)
                index = corner[0] + r * (corner[1] + r * corner[2]);
            else
                index = uint64_t(corner[0] * 1u ^ corner[1] * 2654435761u ^ corner[2] * 805459861u) % shape.tableSize;
            for (uint32_t f = 0; f < shape.features; ++f)
                out[level * shape.features + f] +=
                    w * table[(size_t(level) * shape.tableSize + index) * shape.features + f];
        }
    }
    return out;
}

std::vector<float> readFloats(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    std::vector<float> data(size_t(file.tellg()) / sizeof(float));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(data.data()), std::streamsize(data.size() * sizeof(float)));
    return data;
}

// Runs the layer once and returns its features [levels * features][height][width].
std::vector<float> runLayer(const neural::HashTextureShape& shape, const std::vector<float>& coordinates,
                            const std::vector<float>& classes, const std::vector<float>& table) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    auto coordinateTensor = vc.create<TensorElement<float>>(std::vector<uint32_t>{uint32_t(coordinates.size())});
    auto classTensor = vc.create<TensorElement<float>>(std::vector<uint32_t>{uint32_t(classes.size())});
    auto tableTensor = vc.create<TensorElement<float>>(std::vector<uint32_t>{uint32_t(table.size())});
    auto output = vc.create<TensorElement<float>>(
        std::vector<uint32_t>{shape.levels * shape.features * shape.height * shape.width});
    auto layer = neural::hashTexture(vc, shape);
    layer->setInput(coordinateTensor, 0);
    layer->setInput(classTensor, 1);
    layer->setInput(tableTensor, 2);
    layer->setInput(output, 3);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(layer);
    coordinateTensor->getDataBuffer(0).memcopyFrom(coordinates);
    classTensor->getDataBuffer(0).memcopyFrom(classes);
    tableTensor->getDataBuffer(0).memcopyFrom(table);
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    std::vector<float> features(output->getDataElementCount());
    output->getDataBuffer(0).memcopyTo(features);
    return features;
}

} // namespace

TEST(HashTextureTest, lookupMatchesReference) {
    neural::HashTextureShape shape;
    shape.width = 37;
    shape.height = 21;
    shape.levels = 6;
    shape.features = 2;
    shape.tableSize = 1u << 12;
    shape.resolutions = {4, 8, 16, 32, 64, 128};
    shape.classes = {2, 3, 4};
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    const size_t pixels = size_t(shape.width) * shape.height;
    std::vector<float> table(shape.classes.size() * shape.levels * shape.tableSize * shape.features);
    for (auto& v : table)
        v = uniform(rng) * 2.0f - 1.0f;
    std::vector<float> coordinates(3 * pixels), classes(pixels);
    const float choices[] = {0.0f, 2.0f, 3.0f, 4.0f, 6.0f};
    for (size_t p = 0; p < pixels; ++p) {
        for (int a = 0; a < 3; ++a)
            coordinates[a * pixels + p] = uniform(rng);
        classes[p] = choices[rng() % 5];
    }
    const auto features = runLayer(shape, coordinates, classes, table);
    const size_t channels = size_t(shape.levels) * shape.features;
    ASSERT_EQ(features.size(), channels * pixels);
    float worst = 0.0f;
    for (size_t p = 0; p < pixels; ++p) {
        std::vector<float> expected(channels, 0.0f);
        for (size_t g = 0; g < shape.classes.size(); ++g)
            if (uint32_t(classes[p]) == shape.classes[g]) {
                const float point[3] = {coordinates[p], coordinates[pixels + p], coordinates[2 * pixels + p]};
                expected =
                    referenceFeatures(shape, table.data() + g * shape.levels * shape.tableSize * shape.features, point);
            }
        for (size_t c = 0; c < channels; ++c)
            worst = std::max(worst, std::abs(features[c * pixels + p] - expected[c]));
    }
    std::cout << "largest difference " << worst << std::endl;
    EXPECT_LT(worst, 1e-5f);
}

TEST(HashTextureTest, lookupMatchesTrainedTexture) {
    const std::filesystem::path stem = "build/TestingOutput/gan_runs/v6/photo_enhancer";
    const auto shapePath = stem.string() + "_texture.txt";
    if (!std::filesystem::exists(shapePath))
        GTEST_SKIP() << "train scripts/neural/train_photo_enhancer.py --texture first";
    const auto shape = neural::HashTextureShape::fromFile(shapePath);
    const auto table = readFloats(stem.string() + "_texture_f32.bin");
    const auto coordinates = readFloats(stem.string() + "_texture_coordinates_f32.bin");
    const auto classes = readFloats(stem.string() + "_texture_classes_f32.bin");
    const auto expected = readFloats(stem.string() + "_texture_features_f32.bin");
    const auto features = runLayer(shape, coordinates, classes, table);
    ASSERT_EQ(features.size(), expected.size());
    float worst = 0.0f, largest = 0.0f;
    for (size_t i = 0; i < features.size(); ++i) {
        worst = std::max(worst, std::abs(features[i] - expected[i]));
        largest = std::max(largest, std::abs(expected[i]));
    }
    std::cout << "largest difference " << worst << " of features up to " << largest << std::endl;
    EXPECT_LT(worst, 1e-4f * std::max(largest, 1.0f));
}
