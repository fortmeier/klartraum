// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - activationBackwardMatchesDerivatives: the ReLU and sigmoid gradients from Slang's automatic differentiation
 *   equal the analytic derivatives times the output gradient
 * - sigmoidMseLossBackwardMatchesFiniteDifferences: the logit gradients of the mean squared error of sigmoid
 *   outputs match central differences of the loss, and the squared errors are each element's
 * - columnSumAddsRows: the column sums of a matrix add its rows
 * - adamStepMatchesReference: three Adam steps with the step number from the GPU counter match a CPU Adam
 * - mlpGradientsMatchFiniteDifferences: the weight and bias gradients of a three-layer ReLU network, from one
 *   training step with learning rate zero, match central differences of a CPU forward pass, and the loss is
 *   the CPU loss
 * - mlpTrainingReducesLoss: training a 2-16-16-1 network on the four XOR samples drives the loss below 1 % of
 *   its first value and classifies all four samples
 **/

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/layers/mlp_training.hpp"
#include "klartraum/layers/training.hpp"

using namespace klartraum;
using layers::Shape;

namespace {

using Tensor = TensorElement<float>;

class TrainingTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
    }

    std::shared_ptr<Tensor> tensor(const Shape& shape) { return vc->create<Tensor>(shape); }

    // Runs `element` with `inputs` in its first slots and `outputs` after them; returns the outputs.
    std::vector<std::vector<float>> run(const ComputeGraphElementPtr& element,
                                        const std::vector<std::pair<Shape, std::vector<float>>>& inputs,
                                        const std::vector<Shape>& outputs) {
        std::vector<std::shared_ptr<Tensor>> tensors;
        int slot = 0;
        for (const auto& [shape, values] : inputs) {
            tensors.push_back(tensor(shape));
            element->setInput(tensors.back(), slot++);
        }
        for (const auto& shape : outputs) {
            tensors.push_back(tensor(shape));
            element->setInput(tensors.back(), slot++);
        }
        ComputeGraph graph(*vc, 1);
        graph.compileFrom(element);
        for (size_t i = 0; i < inputs.size(); ++i)
            tensors[i]->getDataBuffer(0).memcopyFrom(inputs[i].second);
        graph.submitAndWait(vc->getGraphicsQueue(), 0);
        std::vector<std::vector<float>> results;
        for (size_t i = inputs.size(); i < tensors.size(); ++i) {
            results.emplace_back(tensors[i]->getDataElementCount());
            tensors[i]->getDataBuffer(0).memcopyTo(results.back());
        }
        return results;
    }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
};

std::vector<float> ramp(size_t count, float scale, float offset) {
    std::vector<float> values(count);
    for (size_t i = 0; i < count; ++i)
        values[i] = offset + scale * float((i * 7) % 23);
    return values;
}

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

} // namespace

TEST_F(TrainingTest, activationBackwardMatchesDerivatives) {
    const Shape shape{3, 17};
    const auto x = ramp(51, 0.37f, -4.0f), dy = ramp(51, 0.11f, -1.2f);
    const auto relu = run(training::activationBackward(*vc, training::Activation::Relu, shape),
                          {{shape, x}, {shape, dy}}, {shape})[0];
    const auto sig = run(training::activationBackward(*vc, training::Activation::Sigmoid, shape),
                         {{shape, x}, {shape, dy}}, {shape})[0];
    for (size_t i = 0; i < x.size(); ++i) {
        EXPECT_FLOAT_EQ(relu[i], x[i] > 0.0f ? dy[i] : 0.0f) << i;
        const double s = sigmoid(x[i]);
        EXPECT_NEAR(sig[i], s * (1.0 - s) * dy[i], 1e-6) << i;
    }
}

TEST_F(TrainingTest, sigmoidMseLossBackwardMatchesFiniteDifferences) {
    const Shape shape{4, 5};
    const auto logits = ramp(20, 0.31f, -3.0f), targets = ramp(20, 0.04f, 0.05f);
    const auto results =
        run(training::sigmoidMseLossBackward(*vc, shape), {{shape, logits}, {shape, targets}}, {shape, shape});
    auto meanLoss = [&](size_t changed, double delta) {
        double sum = 0.0;
        for (size_t i = 0; i < logits.size(); ++i) {
            const double e = sigmoid(logits[i] + (i == changed ? delta : 0.0)) - targets[i];
            sum += e * e;
        }
        return sum / double(logits.size());
    };
    for (size_t i = 0; i < logits.size(); ++i) {
        const double h = 1e-4;
        EXPECT_NEAR(results[0][i], (meanLoss(i, h) - meanLoss(i, -h)) / (2 * h), 1e-6) << i;
        const double e = sigmoid(logits[i]) - targets[i];
        EXPECT_NEAR(results[1][i], e * e, 1e-6) << i;
    }
}

TEST_F(TrainingTest, columnSumAddsRows) {
    const uint32_t rows = 37, columns = 70;
    const auto x = ramp(size_t(rows) * columns, 0.5f, -5.0f);
    const auto sums = run(training::columnSum(*vc, rows, columns), {{{rows, columns}, x}}, {{columns}})[0];
    for (uint32_t c = 0; c < columns; ++c) {
        double expected = 0.0;
        for (uint32_t r = 0; r < rows; ++r)
            expected += x[size_t(r) * columns + c];
        EXPECT_NEAR(sums[c], expected, 1e-4) << c;
    }
}

TEST_F(TrainingTest, adamStepMatchesReference) {
    const uint32_t count = 300;
    training::AdamParameters adam{0.01f, 0.9f, 0.999f, 1e-8f};
    auto parameters = tensor({count}), gradients = tensor({count}), m = tensor({count}), v = tensor({count});
    auto step = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(*vc, 1);
    auto counter = training::stepCounter(*vc);
    counter->setInput(step, 0);
    auto update = training::adamStep(*vc, count, adam);
    update->setInput(parameters, 0);
    update->setInput(gradients, 1);
    update->setInput(m, 2);
    update->setInput(v, 3);
    update->setInput(step, 4);
    update->addDependency(counter);
    ComputeGraph graph(*vc, 1);
    graph.compileFrom(update);
    auto p = ramp(count, 0.2f, -2.0f);
    const std::vector<float> zeros(count, 0.0f);
    parameters->getDataBuffer(0).memcopyFrom(p);
    m->getDataBuffer(0).memcopyFrom(zeros);
    v->getDataBuffer(0).memcopyFrom(zeros);
    step->getBuffer(0).memcopyFrom(std::vector<uint32_t>{0});
    std::vector<double> reference(p.begin(), p.end()), mr(count, 0.0), vr(count, 0.0);
    for (int t = 1; t <= 3; ++t) {
        const auto g = ramp(count, 0.05f * float(t), -0.3f);
        gradients->getDataBuffer(0).memcopyFrom(g);
        graph.submitAndWait(vc->getGraphicsQueue(), 0);
        for (uint32_t i = 0; i < count; ++i) {
            mr[i] = adam.beta1 * mr[i] + (1 - adam.beta1) * g[i];
            vr[i] = adam.beta2 * vr[i] + (1 - adam.beta2) * double(g[i]) * g[i];
            const double mHat = mr[i] / (1 - std::pow(double(adam.beta1), t));
            const double vHat = vr[i] / (1 - std::pow(double(adam.beta2), t));
            reference[i] -= adam.learningRate * mHat / (std::sqrt(vHat) + adam.epsilon);
        }
    }
    parameters->getDataBuffer(0).memcopyTo(p);
    for (uint32_t i = 0; i < count; ++i)
        EXPECT_NEAR(p[i], reference[i], 1e-5) << i;
}

TEST_F(TrainingTest, mlpGradientsMatchFiniteDifferences) {
    training::MlpOptions options;
    options.layerSizes = {3, 5, 4, 2};
    options.batch = 6;
    options.adam.learningRate = 0.0f;
    training::MlpTraining mlp(*vc, options);
    const auto inputs = ramp(18, 0.21f, -1.5f), targets = ramp(12, 0.04f, 0.1f);
    mlp.setInputs(inputs);
    mlp.setTargets(targets);
    mlp.step();

    std::vector<std::vector<double>> weights, biases;
    for (size_t l = 0; l < 3; ++l) {
        const auto w = mlp.parameters(l, false), b = mlp.parameters(l, true);
        weights.emplace_back(w.begin(), w.end());
        biases.emplace_back(b.begin(), b.end());
    }
    // The CPU forward pass and mean squared error in double precision.
    auto loss = [&]() {
        double sum = 0.0;
        for (uint32_t n = 0; n < options.batch; ++n) {
            std::vector<double> h(inputs.begin() + n * 3, inputs.begin() + n * 3 + 3);
            for (size_t l = 0; l < 3; ++l) {
                const uint32_t in = options.layerSizes[l], out = options.layerSizes[l + 1];
                std::vector<double> z(out);
                for (uint32_t o = 0; o < out; ++o) {
                    z[o] = biases[l][o];
                    for (uint32_t i = 0; i < in; ++i)
                        z[o] += h[i] * weights[l][size_t(i) * out + o];
                    if (l < 2)
                        z[o] = std::max(z[o], 0.0);
                }
                h = z;
            }
            for (uint32_t o = 0; o < 2; ++o) {
                const double e = sigmoid(h[o]) - targets[n * 2 + o];
                sum += e * e;
            }
        }
        return sum / double(options.batch * 2);
    };
    EXPECT_NEAR(mlp.loss(), loss(), 1e-6);
    for (size_t l = 0; l < 3; ++l) {
        for (const bool bias : {false, true}) {
            const auto gradients = mlp.gradients(l, bias);
            auto& values = bias ? biases[l] : weights[l];
            ASSERT_EQ(gradients.size(), values.size());
            for (size_t i = 0; i < values.size(); ++i) {
                const double original = values[i], h = 1e-5;
                values[i] = original + h;
                const double up = loss();
                values[i] = original - h;
                const double down = loss();
                values[i] = original;
                EXPECT_NEAR(gradients[i], (up - down) / (2 * h), 2e-6)
                    << "layer " << l << (bias ? " bias " : " weight ") << i;
            }
        }
    }
}

TEST_F(TrainingTest, mlpTrainingReducesLoss) {
    training::MlpOptions options;
    options.layerSizes = {2, 16, 16, 1};
    options.batch = 4;
    options.adam.learningRate = 0.05f;
    training::MlpTraining mlp(*vc, options);
    mlp.setInputs({0, 0, 0, 1, 1, 0, 1, 1});
    const std::vector<float> targets{0, 1, 1, 0};
    mlp.setTargets(targets);
    mlp.step();
    const float first = mlp.loss();
    for (int i = 0; i < 300; ++i)
        mlp.step();
    EXPECT_LT(mlp.loss(), 0.01f * first);
    const auto outputs = mlp.outputs();
    for (size_t i = 0; i < 4; ++i)
        EXPECT_EQ(outputs[i] > 0.5f, targets[i] > 0.5f) << i;
}
