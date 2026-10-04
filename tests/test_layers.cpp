// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - broadcastShapeRules: shapes broadcast from the right; ones stretch, other mismatches throw
 * - scaleAndOffsetChain: Mul and Add with scalar tensors chained in one graph compute a*x+b
 *   (the VAE decoder's latent scaling) without an ONNX model
 * - softmaxRowsSumToOne: last-axis softmax matches a CPU softmax
 * - batchedMatMul: a batch of matrix products matches the CPU, with the right side broadcast
 * - matMulOddSizes: sizes that are not multiples of any tile, both sides batched, match the CPU
 * - matMulCosmos3Projection: a [1152, 2048] x [2048, 2048] weight product (Cosmos3 denoiser at
 *   256x256) matches the CPU on sampled outputs; prints its GPU throughput
 * - conv3x3WithBias: the specialized 3x3 convolution with padding and bias matches a CPU convolution
 * - layerNormalization: last-axis normalization with scale and bias matches the CPU
 **/

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/layers/layers.hpp"

using namespace klartraum;
using layers::Shape;

namespace {

using Tensor = TensorElement<float>;

class LayersTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
    }

    std::shared_ptr<Tensor> tensor(const Shape& shape) { return vc->create<Tensor>(shape); }

    // Connects `inputs`, then `output`, to `layer` (the layers' slot
    // convention), runs it, and returns the output.
    std::vector<float> run(const ComputeGraphElementPtr& layer,
                           const std::vector<std::pair<std::shared_ptr<Tensor>, std::vector<float>>>& inputs,
                           const std::shared_ptr<Tensor>& output) {
        int slot = 0;
        for (const auto& [input, values] : inputs) {
            layer->setInput(input, slot++);
        }
        layer->setInput(output, slot);
        ComputeGraph graph(*vc, 1);
        graph.compileFrom(layer);
        for (const auto& [input, values] : inputs) {
            input->getDataBuffer(0).memcopyFrom(values);
        }
        graph.submitAndWait(vc->getGraphicsQueue(), 0);
        std::vector<float> result(output->getDataElementCount());
        output->getDataBuffer(0).memcopyTo(result);
        return result;
    }

    static std::vector<float> ramp(size_t count, float scale = 0.1f, float offset = -1.0f) {
        std::vector<float> values(count);
        for (size_t i = 0; i < count; ++i) {
            values[i] = offset + scale * static_cast<float>((i * 7) % 23);
        }
        return values;
    }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
};

void expectNear(const std::vector<float>& actual, const std::vector<float>& expected, float tolerance) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_NEAR(actual[i], expected[i], tolerance) << "element " << i;
    }
}

} // namespace

TEST(Layers, broadcastShapeRules) {
    EXPECT_EQ(layers::broadcastShape({2, 3, 4}, {4}), (Shape{2, 3, 4}));
    EXPECT_EQ(layers::broadcastShape({2, 1}, {1, 3}), (Shape{2, 3}));
    EXPECT_EQ(layers::broadcastShape({1, 4, 8, 8}, {1}), (Shape{1, 4, 8, 8}));
    EXPECT_THROW(layers::broadcastShape({2, 3}, {4, 3}), std::runtime_error);
    EXPECT_EQ(layers::elementCount({2, 3, 4}), 24u);
}

TEST_F(LayersTest, scaleAndOffsetChain) {
    const Shape shape{1, 4, 2, 2};
    const Shape scalar{1};
    auto x = tensor(shape);
    auto a = tensor(scalar);
    auto b = tensor(scalar);
    auto scaled = tensor(shape);
    auto result = tensor(shape);

    auto mul = layers::binary(*vc, layers::BinaryOp::Mul, shape, scalar, shape);
    mul->setInput(x, 0);
    mul->setInput(a, 1);
    mul->setInput(scaled, 2);
    auto add = layers::binary(*vc, layers::BinaryOp::Add, shape, scalar, shape);
    add->setInput(mul, 0, 2); // Mul's output tensor
    add->setInput(b, 1);
    add->setInput(result, 2);

    ComputeGraph graph(*vc, 1);
    graph.compileFrom(add);
    const auto values = ramp(16);
    x->getDataBuffer(0).memcopyFrom(values);
    a->getDataBuffer(0).memcopyFrom(std::vector<float>{0.5f});
    b->getDataBuffer(0).memcopyFrom(std::vector<float>{0.5f});
    graph.submitAndWait(vc->getGraphicsQueue(), 0);

    std::vector<float> actual(16);
    result->getDataBuffer(0).memcopyTo(actual);
    std::vector<float> expected(16);
    for (size_t i = 0; i < 16; ++i) {
        expected[i] = 0.5f * values[i] + 0.5f;
    }
    expectNear(actual, expected, 1e-6f);
}

TEST_F(LayersTest, softmaxRowsSumToOne) {
    const Shape shape{3, 7};
    const auto values = ramp(21, 0.4f, -2.0f);
    const auto actual = run(layers::softmax(*vc, shape), {{tensor(shape), values}}, tensor(shape));
    std::vector<float> expected(21);
    for (size_t row = 0; row < 3; ++row) {
        float maximum = -INFINITY;
        for (size_t i = 0; i < 7; ++i)
            maximum = std::max(maximum, values[row * 7 + i]);
        float sum = 0.0f;
        for (size_t i = 0; i < 7; ++i)
            sum += std::exp(values[row * 7 + i] - maximum);
        for (size_t i = 0; i < 7; ++i)
            expected[row * 7 + i] = std::exp(values[row * 7 + i] - maximum) / sum;
    }
    expectNear(actual, expected, 1e-5f);
}

TEST_F(LayersTest, batchedMatMul) {
    const Shape lhs{2, 5, 3};
    const Shape rhs{3, 4}; // one right side for both batches
    const Shape output{2, 5, 4};
    const auto l = ramp(30);
    const auto r = ramp(12, 0.2f, 0.5f);
    const auto actual =
        run(layers::matMul(*vc, lhs, rhs, output), {{tensor(lhs), l}, {tensor(rhs), r}}, tensor(output));
    std::vector<float> expected(40, 0.0f);
    for (size_t batch = 0; batch < 2; ++batch) {
        for (size_t row = 0; row < 5; ++row) {
            for (size_t column = 0; column < 4; ++column) {
                for (size_t k = 0; k < 3; ++k) {
                    expected[(batch * 5 + row) * 4 + column] += l[(batch * 5 + row) * 3 + k] * r[k * 4 + column];
                }
            }
        }
    }
    expectNear(actual, expected, 1e-5f);
}

TEST_F(LayersTest, matMulOddSizes) {
    const Shape lhs{3, 67, 45};
    const Shape rhs{3, 45, 70};
    const Shape output{3, 67, 70};
    const auto l = ramp(3 * 67 * 45, 0.05f, -0.5f);
    const auto r = ramp(3 * 45 * 70, 0.03f, -0.3f);
    const auto actual =
        run(layers::matMul(*vc, lhs, rhs, output), {{tensor(lhs), l}, {tensor(rhs), r}}, tensor(output));
    std::vector<float> expected(3 * 67 * 70, 0.0f);
    for (size_t batch = 0; batch < 3; ++batch) {
        for (size_t row = 0; row < 67; ++row) {
            for (size_t column = 0; column < 70; ++column) {
                double sum = 0.0;
                for (size_t k = 0; k < 45; ++k) {
                    sum += double(l[(batch * 67 + row) * 45 + k]) * r[(batch * 45 + k) * 70 + column];
                }
                expected[(batch * 67 + row) * 70 + column] = float(sum);
            }
        }
    }
    expectNear(actual, expected, 1e-4f);
}

TEST_F(LayersTest, matMulCosmos3Projection) {
    const uint32_t rows = 1152, reduction = 2048, columns = 2048;
    const Shape lhs{rows, reduction};
    const Shape rhs{reduction, columns};
    const Shape output{rows, columns};
    std::vector<float> l(size_t(rows) * reduction), r(size_t(reduction) * columns);
    for (size_t i = 0; i < l.size(); ++i)
        l[i] = float(int((i * 2654435761u) % 2001) - 1000) * 1e-3f;
    for (size_t i = 0; i < r.size(); ++i)
        r[i] = float(int((i * 40503u) % 2001) - 1000) * 2e-5f;

    auto layer = layers::matMul(*vc, lhs, rhs, output);
    auto lhsTensor = tensor(lhs), rhsTensor = tensor(rhs), outputTensor = tensor(output);
    layer->setInput(lhsTensor, 0);
    layer->setInput(rhsTensor, 1);
    layer->setInput(outputTensor, 2);
    ComputeGraph graph(*vc, 1);
    graph.compileFrom(layer);
    lhsTensor->getDataBuffer(0).memcopyFrom(l);
    rhsTensor->getDataBuffer(0).memcopyFrom(r);
    graph.submitAndWait(vc->getGraphicsQueue(), 0); // warm-up
    const int repeats = 20;
    const auto started = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i)
        graph.submitAndWait(vc->getGraphicsQueue(), 0);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() / repeats;
    std::cout << "matMulCosmos3Projection: " << seconds * 1e3 << " ms, "
              << 2.0 * rows * reduction * columns / seconds * 1e-9 << " GFLOP/s" << std::endl;

    std::vector<float> actual(size_t(rows) * columns);
    outputTensor->getDataBuffer(0).memcopyTo(actual);
    for (uint32_t sample = 0; sample < 256; ++sample) {
        const uint32_t row = (sample * 977) % rows, column = (sample * 1531 + 7) % columns;
        double expected = 0.0;
        for (uint32_t k = 0; k < reduction; ++k)
            expected += double(l[size_t(row) * reduction + k]) * r[size_t(k) * columns + column];
        EXPECT_NEAR(actual[size_t(row) * columns + column], expected, 1e-4) << "row " << row << " column " << column;
    }
}

TEST_F(LayersTest, conv3x3WithBias) {
    const Shape input{1, 2, 5, 6};
    const Shape weights{3, 2, 3, 3};
    const Shape bias{3};
    const Shape output{1, 3, 5, 6};
    layers::ConvAttributes attributes;
    attributes.kernelShape = {3, 3};
    attributes.pads = {1, 1, 1, 1};
    const auto x = ramp(60);
    const auto w = ramp(54, 0.05f, -0.5f);
    const std::vector<float> b{0.1f, -0.2f, 0.3f};
    const auto actual = run(layers::conv(*vc, attributes, input, weights, bias, output),
                            {{tensor(input), x}, {tensor(weights), w}, {tensor(bias), b}}, tensor(output));
    std::vector<float> expected(90);
    for (int o = 0; o < 3; ++o) {
        for (int y = 0; y < 5; ++y) {
            for (int xx = 0; xx < 6; ++xx) {
                float sum = b[o];
                for (int c = 0; c < 2; ++c) {
                    for (int ky = 0; ky < 3; ++ky) {
                        for (int kx = 0; kx < 3; ++kx) {
                            const int iy = y + ky - 1;
                            const int ix = xx + kx - 1;
                            if (iy < 0 || iy >= 5 || ix < 0 || ix >= 6)
                                continue;
                            sum += x[(c * 5 + iy) * 6 + ix] * w[((o * 2 + c) * 3 + ky) * 3 + kx];
                        }
                    }
                }
                expected[(o * 5 + y) * 6 + xx] = sum;
            }
        }
    }
    expectNear(actual, expected, 1e-4f);
}

TEST_F(LayersTest, layerNormalization) {
    const Shape shape{4, 6};
    const Shape axis{6};
    const auto values = ramp(24, 0.3f, -1.5f);
    const auto scale = ramp(6, 0.1f, 0.5f);
    const auto bias = ramp(6, 0.05f, -0.1f);
    const auto actual = run(layers::layerNormalization(*vc, shape, 1e-5f),
                            {{tensor(shape), values}, {tensor(axis), scale}, {tensor(axis), bias}}, tensor(shape));
    std::vector<float> expected(24);
    for (size_t row = 0; row < 4; ++row) {
        const float* v = &values[row * 6];
        const float mean = std::accumulate(v, v + 6, 0.0f) / 6.0f;
        float variance = 0.0f;
        for (size_t i = 0; i < 6; ++i)
            variance += (v[i] - mean) * (v[i] - mean);
        variance /= 6.0f;
        for (size_t i = 0; i < 6; ++i) {
            expected[row * 6 + i] = (v[i] - mean) / std::sqrt(variance + 1e-5f) * scale[i] + bias[i];
        }
    }
    expectNear(actual, expected, 1e-4f);
}
