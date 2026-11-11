// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - convBackwardMatchesPyTorch: the input, weight and bias gradients of 2D convolutions (3x3 with stride 1 and 2,
 *   1x1, 4x4 with stride 2) equal PyTorch autograd's for the same inputs and upstream gradient
 *
 * The references come from scripts/neural/make_training_fixtures.py (build/TestingOutput/training_fixtures/); the
 * tests skip without them.
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/layers/training.hpp"

using namespace klartraum;

namespace {

const std::string kFixtures = "build/TestingOutput/training_fixtures/";

// A fixture case: its tensors' shapes and values.
struct Fixture {
    std::map<std::string, std::vector<uint32_t>> shapes;

    explicit Fixture(const std::string& name)
        : name(name) {
        std::ifstream file(kFixtures + name + ".txt");
        for (std::string line; std::getline(file, line);) {
            std::istringstream words(line);
            std::string tensor;
            words >> tensor;
            std::vector<uint32_t> shape;
            for (uint32_t d; words >> d;)
                shape.push_back(d);
            shapes[tensor] = shape;
        }
    }

    bool exists() const { return !shapes.empty(); }

    std::vector<float> values(const std::string& tensor) const {
        std::ifstream file(kFixtures + name + "_" + tensor + ".bin", std::ios::binary | std::ios::ate);
        std::vector<float> data(size_t(file.tellg()) / sizeof(float));
        file.seekg(0);
        file.read(reinterpret_cast<char*>(data.data()), std::streamsize(data.size() * sizeof(float)));
        return data;
    }

    std::string name;
};

std::shared_ptr<TensorElement<float>> tensor(VulkanContext& vc, const std::vector<uint32_t>& shape) {
    return vc.create<TensorElement<float>>(shape);
}

std::vector<float> read(const std::shared_ptr<TensorElement<float>>& t) {
    std::vector<float> values(t->getDataElementCount());
    t->getDataBuffer(0).memcopyTo(values);
    return values;
}

// The largest difference relative to the reference's largest magnitude.
float relativeError(const std::vector<float>& actual, const std::vector<float>& expected) {
    EXPECT_EQ(actual.size(), expected.size());
    float worst = 0.0f, largest = 1e-12f;
    for (size_t i = 0; i < std::min(actual.size(), expected.size()); ++i) {
        worst = std::max(worst, std::abs(actual[i] - expected[i]));
        largest = std::max(largest, std::abs(expected[i]));
    }
    return worst / largest;
}

} // namespace

TEST(PhotoEnhancerTraining, convBackwardMatchesPyTorch) {
    const struct {
        const char* name;
        uint32_t kernel, stride, pad;
    } cases[] = {
        {"conv3x3", 3, 1, 1}, {"conv3x3_stride2", 3, 2, 1}, {"conv1x1", 1, 1, 0}, {"conv4x4_stride2", 4, 2, 1}};
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    for (const auto& c : cases) {
        const Fixture fixture(c.name);
        if (!fixture.exists())
            GTEST_SKIP() << "run scripts/neural/make_training_fixtures.py first";
        layers::ConvAttributes attributes;
        attributes.kernelShape = {c.kernel, c.kernel};
        attributes.strides = {c.stride, c.stride};
        attributes.pads = {c.pad, c.pad, c.pad, c.pad};
        const auto& xShape = fixture.shapes.at("x");
        const auto& wShape = fixture.shapes.at("w");
        const auto& yShape = fixture.shapes.at("y");
        auto x = tensor(vc, xShape), w = tensor(vc, wShape), dy = tensor(vc, yShape);
        auto dx = tensor(vc, xShape), dw = tensor(vc, wShape), db = tensor(vc, {wShape[0]});
        auto inputGradient = training::convBackwardInput(vc, attributes, xShape, wShape, yShape);
        inputGradient->setInput(dy, 0);
        inputGradient->setInput(w, 1);
        inputGradient->setInput(dx, 2);
        auto weightGradient = training::convBackwardWeights(vc, attributes, xShape, wShape, yShape);
        weightGradient->setInput(x, 0);
        weightGradient->setInput(dy, 1);
        weightGradient->setInput(dw, 2);
        weightGradient->setInput(db, 3);
        weightGradient->addDependency(inputGradient);
        ComputeGraph graph(vc, 1);
        graph.compileFrom(weightGradient);
        x->getDataBuffer(0).memcopyFrom(fixture.values("x"));
        w->getDataBuffer(0).memcopyFrom(fixture.values("w"));
        dy->getDataBuffer(0).memcopyFrom(fixture.values("dy"));
        graph.submitAndWait(vc.getGraphicsQueue(), 0);
        EXPECT_LT(relativeError(read(dx), fixture.values("dx")), 1e-5f) << c.name;
        EXPECT_LT(relativeError(read(dw), fixture.values("dw")), 1e-5f) << c.name;
        EXPECT_LT(relativeError(read(db), fixture.values("db")), 1e-5f) << c.name;
    }
}
