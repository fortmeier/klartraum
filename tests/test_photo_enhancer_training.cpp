// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - convBackwardMatchesPyTorch: the input, weight and bias gradients of 2D convolutions (3x3 with stride 1 and 2,
 *   1x1, 4x4 with stride 2) equal PyTorch autograd's for the same inputs and upstream gradient
 * - activationBackwardMatchesPyTorch: the ELU and leaky ReLU (0.2) input gradients, and the leaky ReLU itself, equal
 *   PyTorch's
 * - channelNormMatchesPyTorch: ChannelNorm's output and its input, scale and shift gradients equal PyTorch's
 *   (train_photo_enhancer.ChannelNorm)
 * - upsampleBackwardMatchesPyTorch: the input gradient of a bilinear upsampling by two equals PyTorch's
 *   (F.interpolate, align_corners=False)
 * - channelCopyConcatsAndSplits: copying two tensors' channels to their offsets concatenates them, copying them back
 *   out recovers them, and accumulating adds
 * - filmMatchesPyTorch: FiLM's output and its input, scale and shift gradients equal PyTorch's
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

namespace {

// Runs `element` on tensors of the given shapes (bound to its slots in order, the first `inputs` filled from the
// fixture's tensors of the same names) and returns the fixture comparisons of the rest.
void checkElement(VulkanContext& vc, const Fixture& fixture, ComputeGraphElementPtr element,
                  const std::vector<std::pair<std::string, std::vector<uint32_t>>>& slots, size_t inputs,
                  float tolerance = 1e-5f) {
    std::vector<std::shared_ptr<TensorElement<float>>> tensors;
    for (size_t k = 0; k < slots.size(); ++k) {
        tensors.push_back(tensor(vc, slots[k].second));
        element->setInput(tensors.back(), int(k));
    }
    ComputeGraph graph(vc, 1);
    graph.compileFrom(element);
    for (size_t k = 0; k < inputs; ++k)
        tensors[k]->getDataBuffer(0).memcopyFrom(fixture.values(slots[k].first));
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    for (size_t k = inputs; k < slots.size(); ++k)
        if (fixture.shapes.count(slots[k].first))
            EXPECT_LT(relativeError(read(tensors[k]), fixture.values(slots[k].first)), tolerance)
                << fixture.name << " " << slots[k].first;
}

} // namespace

TEST(PhotoEnhancerTraining, activationBackwardMatchesPyTorch) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    for (const auto& [name, activation] :
         {std::pair{"elu", training::Activation::Elu}, std::pair{"leaky_relu", training::Activation::LeakyRelu}}) {
        const Fixture fixture(name);
        if (!fixture.exists())
            GTEST_SKIP() << "run scripts/neural/make_training_fixtures.py first";
        const auto shape = fixture.shapes.at("x");
        checkElement(vc, fixture, training::activationBackward(vc, activation, shape),
                     {{"x", shape}, {"dy", shape}, {"dx", shape}}, 2);
        if (activation == training::Activation::LeakyRelu)
            checkElement(vc, fixture, training::leakyRelu(vc, shape), {{"x", shape}, {"y", shape}}, 1);
    }
}

TEST(PhotoEnhancerTraining, channelNormMatchesPyTorch) {
    const Fixture fixture("channel_norm");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_training_fixtures.py first";
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const auto shape = fixture.shapes.at("x");
    const uint32_t c = shape[1];
    const std::vector<uint32_t> statsShape{shape[0], 2, shape[2], shape[3]};
    auto x = tensor(vc, shape), weight = tensor(vc, {c}), bias = tensor(vc, {c}), y = tensor(vc, shape);
    auto stats = tensor(vc, statsShape), dy = tensor(vc, shape), dx = tensor(vc, shape);
    auto dweight = tensor(vc, {c}), dbias = tensor(vc, {c});
    auto forward = training::channelNorm(vc, shape);
    forward->setInput(x, 0);
    forward->setInput(weight, 1);
    forward->setInput(bias, 2);
    forward->setInput(y, 3);
    forward->setInput(stats, 4);
    auto input = training::channelNormBackwardInput(vc, shape);
    input->setInput(x, 0);
    input->setInput(weight, 1);
    input->setInput(stats, 2);
    input->setInput(dy, 3);
    input->setInput(dx, 4);
    input->addDependency(forward);
    auto parameters = training::channelNormBackwardParameters(vc, shape);
    parameters->setInput(x, 0);
    parameters->setInput(stats, 1);
    parameters->setInput(dy, 2);
    parameters->setInput(dweight, 3);
    parameters->setInput(dbias, 4);
    parameters->addDependency(input);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(parameters);
    x->getDataBuffer(0).memcopyFrom(fixture.values("x"));
    weight->getDataBuffer(0).memcopyFrom(fixture.values("weight"));
    bias->getDataBuffer(0).memcopyFrom(fixture.values("bias"));
    dy->getDataBuffer(0).memcopyFrom(fixture.values("dy"));
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    EXPECT_LT(relativeError(read(y), fixture.values("y")), 1e-5f);
    EXPECT_LT(relativeError(read(dx), fixture.values("dx")), 1e-5f);
    EXPECT_LT(relativeError(read(dweight), fixture.values("dweight")), 1e-5f);
    EXPECT_LT(relativeError(read(dbias), fixture.values("dbias")), 1e-5f);
}

TEST(PhotoEnhancerTraining, upsampleBackwardMatchesPyTorch) {
    const Fixture fixture("upsample2x");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_training_fixtures.py first";
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const auto shape = fixture.shapes.at("x");
    checkElement(vc, fixture, training::upsample2xBackward(vc, shape), {{"dy", fixture.shapes.at("y")}, {"dx", shape}},
                 1);
}

TEST(PhotoEnhancerTraining, channelCopyConcatsAndSplits) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const uint32_t n = 2, pixels = 6;
    auto a = tensor(vc, {n, 2, 2, 3}), b = tensor(vc, {n, 3, 2, 3}), joined = tensor(vc, {n, 5, 2, 3});
    auto back = tensor(vc, {n, 3, 2, 3});
    auto copyA = training::channelCopy(vc, n, pixels, {2, 0}, {5, 0}, 2);
    copyA->setInput(a, 0);
    copyA->setInput(joined, 1);
    auto copyB = training::channelCopy(vc, n, pixels, {3, 0}, {5, 2}, 3);
    copyB->setInput(b, 0);
    copyB->setInput(joined, 1);
    copyB->addDependency(copyA);
    // b's channels back out, then added once more.
    auto split = training::channelCopy(vc, n, pixels, {5, 2}, {3, 0}, 3);
    split->setInput(joined, 0);
    split->setInput(back, 1);
    split->addDependency(copyB);
    auto add = training::channelCopy(vc, n, pixels, {5, 2}, {3, 0}, 3, true);
    add->setInput(joined, 0);
    add->setInput(back, 1);
    add->addDependency(split);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(add);
    std::vector<float> va(n * 2 * pixels), vb(n * 3 * pixels);
    for (size_t i = 0; i < va.size(); ++i)
        va[i] = float(i);
    for (size_t i = 0; i < vb.size(); ++i)
        vb[i] = 100.0f + float(i);
    a->getDataBuffer(0).memcopyFrom(va);
    b->getDataBuffer(0).memcopyFrom(vb);
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    const auto j = read(joined), r = read(back);
    for (uint32_t s = 0; s < n; ++s)
        for (uint32_t p = 0; p < pixels; ++p) {
            for (uint32_t c = 0; c < 2; ++c)
                EXPECT_EQ(j[(s * 5 + c) * pixels + p], va[(s * 2 + c) * pixels + p]);
            for (uint32_t c = 0; c < 3; ++c) {
                EXPECT_EQ(j[(s * 5 + 2 + c) * pixels + p], vb[(s * 3 + c) * pixels + p]);
                EXPECT_EQ(r[(s * 3 + c) * pixels + p], 2.0f * vb[(s * 3 + c) * pixels + p]);
            }
        }
}

TEST(PhotoEnhancerTraining, filmMatchesPyTorch) {
    const Fixture fixture("film");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_training_fixtures.py first";
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const auto shape = fixture.shapes.at("x");
    const std::vector<uint32_t> planes{shape[0], shape[1]};
    checkElement(vc, fixture, training::film(vc, shape),
                 {{"x", shape}, {"scale", planes}, {"shift", planes}, {"y", shape}}, 3);
    checkElement(
        vc, fixture, training::filmBackward(vc, shape),
        {{"x", shape}, {"scale", planes}, {"dy", shape}, {"dx", shape}, {"dscale", planes}, {"dshift", planes}}, 3);
}
