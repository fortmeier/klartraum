// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - activationBackwardMatchesPyTorch: the ELU and leaky ReLU (0.2) input gradients, and the leaky ReLU itself, equal
 *   PyTorch's
 * - channelCopyConcatsAndSplits: copying two tensors' channels to their offsets concatenates them, copying them back
 *   out recovers them, and accumulating adds
 * - linearMatchesPyTorch: a linear layer's output (with an offset) and its input (also accumulating), weight and
 *   bias gradients equal nn.Linear's
 * - convBackwardMatchesPyTorch: the input, weight and bias gradients of 2D convolutions (3x3 with stride 1 and 2,
 *   1x1, 4x4 with stride 2) equal PyTorch autograd's for the same inputs and upstream gradient
 * - channelNormMatchesPyTorch: ChannelNorm's output and its input, scale and shift gradients equal PyTorch's
 *   (train_photo_enhancer.ChannelNorm)
 * - upsampleBackwardMatchesPyTorch: the input gradient of a bilinear upsampling by two equals PyTorch's
 *   (F.interpolate, align_corners=False)
 * - maxPoolMatchesPyTorch: a 2 x 2 max-pool and its input gradient equal F.max_pool2d's
 * - averagePoolMatchesMeans: a 2 x 2 average pool gives the mean of every 2 x 2 block, and its backward pass spreads a
 *   quarter of each output gradient onto the block's pixels, written or added to what is there
 * - embeddingPicksAndScattersRows: rows picked by index (one twice) come out in batch order; the table gradient adds
 *   the picked rows' gradients plus the L2 penalty's, and is zero for rows nobody picked
 *
 * The references come from scripts/neural/make_element_fixtures.py (build/TestingOutput/training_fixtures/); the
 * tests skip without them.
 **/

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/layers/training.hpp"
#include "training_fixtures.hpp"

using namespace klartraum;
using namespace klartraum::test;

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

TEST(TrainingElements, activationBackwardMatchesPyTorch) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    for (const auto& [name, activation] :
         {std::pair{"elu", training::Activation::Elu}, std::pair{"leaky_relu", training::Activation::LeakyRelu}}) {
        const Fixture fixture(name);
        if (!fixture.exists())
            GTEST_SKIP() << "run scripts/neural/make_element_fixtures.py first";
        const auto shape = fixture.shapes.at("x");
        checkElement(vc, fixture, training::activationBackward(vc, activation, shape),
                     {{"x", shape}, {"dy", shape}, {"dx", shape}}, 2);
        if (activation == training::Activation::LeakyRelu)
            checkElement(vc, fixture, training::leakyRelu(vc, shape), {{"x", shape}, {"y", shape}}, 1);
    }
}

TEST(TrainingElements, channelCopyConcatsAndSplits) {
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

TEST(TrainingElements, linearMatchesPyTorch) {
    const Fixture fixture("linear");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_element_fixtures.py first";
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const training::LinearShape shape{3, 5, 7};
    checkElement(vc, fixture, training::linear(vc, shape), {{"x", {3, 5}}, {"w", {7, 5}}, {"b", {7}}, {"y", {3, 7}}},
                 3);
    checkElement(vc, fixture, training::linearBackwardInput(vc, shape), {{"dy", {3, 7}}, {"w", {7, 5}}, {"dx", {3, 5}}},
                 2);
    checkElement(vc, fixture, training::linearBackwardWeights(vc, shape),
                 {{"x", {3, 5}}, {"dy", {3, 7}}, {"dw", {7, 5}}, {"db", {7}}}, 2);
    // The offset adds to every output, and accumulate adds to the input gradient.
    auto x = tensor(vc, {3, 5}), w = tensor(vc, {7, 5}), b = tensor(vc, {7}), y = tensor(vc, {3, 7});
    auto dy = tensor(vc, {3, 7}), dx = tensor(vc, {3, 5});
    auto forward = training::linear(vc, shape, 1.0f);
    forward->setInput(x, 0);
    forward->setInput(w, 1);
    forward->setInput(b, 2);
    forward->setInput(y, 3);
    auto first = training::linearBackwardInput(vc, shape);
    first->setInput(dy, 0);
    first->setInput(w, 1);
    first->setInput(dx, 2);
    first->addDependency(forward);
    auto second = training::linearBackwardInput(vc, shape, true);
    second->setInput(dy, 0);
    second->setInput(w, 1);
    second->setInput(dx, 2);
    second->addDependency(first);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(second);
    x->getDataBuffer(0).memcopyFrom(fixture.values("x"));
    w->getDataBuffer(0).memcopyFrom(fixture.values("w"));
    b->getDataBuffer(0).memcopyFrom(fixture.values("b"));
    dy->getDataBuffer(0).memcopyFrom(fixture.values("dy"));
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    auto expectedY = fixture.values("y"), expectedDx = fixture.values("dx");
    for (auto& v : expectedY)
        v += 1.0f;
    for (auto& v : expectedDx)
        v *= 2.0f;
    EXPECT_LT(relativeError(read(y), expectedY), 1e-5f);
    EXPECT_LT(relativeError(read(dx), expectedDx), 1e-5f);
}

TEST(TrainingElements, convBackwardMatchesPyTorch) {
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
            GTEST_SKIP() << "run scripts/neural/make_element_fixtures.py first";
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

TEST(TrainingElements, channelNormMatchesPyTorch) {
    const Fixture fixture("channel_norm");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_element_fixtures.py first";
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

TEST(TrainingElements, upsampleBackwardMatchesPyTorch) {
    const Fixture fixture("upsample2x");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_element_fixtures.py first";
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const auto shape = fixture.shapes.at("x");
    checkElement(vc, fixture, training::upsample2xBackward(vc, shape), {{"dy", fixture.shapes.at("y")}, {"dx", shape}},
                 1);
}

TEST(TrainingElements, maxPoolMatchesPyTorch) {
    const Fixture fixture("max_pool");
    if (!fixture.exists())
        GTEST_SKIP() << "run scripts/neural/make_element_fixtures.py first";
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const auto shape = fixture.shapes.at("x"), pooled = fixture.shapes.at("y");
    auto x = tensor(vc, shape), y = tensor(vc, pooled), taken = tensor(vc, pooled), dy = tensor(vc, pooled);
    auto dx = tensor(vc, shape);
    auto forward = training::maxPool(vc, shape);
    forward->setInput(x, 0);
    forward->setInput(y, 1);
    forward->setInput(taken, 2);
    auto backward = training::maxPoolBackward(vc, shape);
    backward->setInput(dy, 0);
    backward->setInput(taken, 1);
    backward->setInput(dx, 2);
    backward->addDependency(forward);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(backward);
    x->getDataBuffer(0).memcopyFrom(fixture.values("x"));
    dy->getDataBuffer(0).memcopyFrom(fixture.values("dy"));
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    EXPECT_EQ(read(y), fixture.values("y"));
    EXPECT_EQ(read(dx), fixture.values("dx"));
}

TEST(TrainingElements, averagePoolMatchesMeans) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const std::vector<uint32_t> shape{2, 3, 4, 6}, pooled{2, 3, 2, 3};
    auto x = tensor(vc, shape), y = tensor(vc, pooled), dy = tensor(vc, pooled), dx = tensor(vc, shape);
    auto dxAdded = tensor(vc, shape);
    auto forward = training::averagePool(vc, shape);
    forward->setInput(x, 0);
    forward->setInput(y, 1);
    auto backward = training::averagePoolBackward(vc, shape);
    backward->setInput(dy, 0);
    backward->setInput(dx, 1);
    backward->addDependency(forward);
    auto added = training::averagePoolBackward(vc, shape, true);
    added->setInput(dy, 0);
    added->setInput(dxAdded, 1);
    added->addDependency(backward);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(added);
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    std::vector<float> xs(2 * 3 * 4 * 6), dys(2 * 3 * 2 * 3);
    for (auto& v : xs)
        v = uniform(rng);
    for (auto& v : dys)
        v = uniform(rng);
    x->getDataBuffer(0).memcopyFrom(xs);
    dy->getDataBuffer(0).memcopyFrom(dys);
    dxAdded->getDataBuffer(0).memcopyFrom(std::vector<float>(xs.size(), 1.0f));
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    const auto ys = read(y), dxs = read(dx), dxsAdded = read(dxAdded);
    for (uint32_t plane = 0; plane < 6; ++plane)
        for (uint32_t oy = 0; oy < 2; ++oy)
            for (uint32_t ox = 0; ox < 3; ++ox) {
                float mean = 0.0f;
                for (uint32_t k = 0; k < 4; ++k)
                    mean += 0.25f * xs[(plane * 4 + 2 * oy + k / 2) * 6 + 2 * ox + k % 2];
                const size_t o = (plane * 2 + oy) * 3 + ox;
                EXPECT_NEAR(ys[o], mean, 1e-6f);
                for (uint32_t k = 0; k < 4; ++k) {
                    const size_t i = (plane * 4 + 2 * oy + k / 2) * 6 + 2 * ox + k % 2;
                    EXPECT_NEAR(dxs[i], 0.25f * dys[o], 1e-6f);
                    EXPECT_NEAR(dxsAdded[i], 1.0f + 0.25f * dys[o], 1e-6f);
                }
            }
}

TEST(TrainingElements, embeddingPicksAndScattersRows) {
    HeadlessFrontend frontend;
    auto& vc = frontend.getKlartraumEngine().getVulkanContext();
    const uint32_t batch = 3, rows = 4, size = 2;
    const float decay = 0.1f;
    auto table = tensor(vc, {rows, size}), indices = tensor(vc, {batch}), picked = tensor(vc, {batch, size});
    auto pickedGradient = tensor(vc, {batch, size}), tableGradient = tensor(vc, {rows, size});
    auto pick = training::embedding(vc, batch, rows, size);
    pick->setInput(table, 0);
    pick->setInput(indices, 1);
    pick->setInput(picked, 2);
    auto scatter = training::embeddingBackward(vc, batch, rows, size, decay);
    scatter->setInput(table, 0);
    scatter->setInput(indices, 1);
    scatter->setInput(pickedGradient, 2);
    scatter->setInput(tableGradient, 3);
    scatter->addDependency(pick);
    ComputeGraph graph(vc, 1);
    graph.compileFrom(scatter);
    const std::vector<float> values{1, 2, 3, 4, 5, 6, 7, 8}, rowsPicked{2, 0, 2}, gradients{0.5f, 1, 2, 3, 4, 5};
    table->getDataBuffer(0).memcopyFrom(values);
    indices->getDataBuffer(0).memcopyFrom(rowsPicked);
    pickedGradient->getDataBuffer(0).memcopyFrom(gradients);
    graph.submitAndWait(vc.getGraphicsQueue(), 0);
    EXPECT_EQ(read(picked), (std::vector<float>{5, 6, 1, 2, 5, 6}));
    const auto g = read(tableGradient);
    const float p = 2.0f * decay / float(batch);
    EXPECT_NEAR(g[0], 2.0f + p * 1.0f, 1e-6f);
    EXPECT_NEAR(g[1], 3.0f + p * 2.0f, 1e-6f);
    EXPECT_EQ(g[2], 0.0f);
    EXPECT_EQ(g[3], 0.0f);
    EXPECT_NEAR(g[4], 0.5f + 4.0f + 2.0f * p * 5.0f, 1e-6f);
    EXPECT_NEAR(g[5], 1.0f + 5.0f + 2.0f * p * 6.0f, 1e-6f);
    EXPECT_EQ(g[6], 0.0f);
}
