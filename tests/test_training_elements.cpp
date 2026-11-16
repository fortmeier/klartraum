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
