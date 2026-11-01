// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - conv: 2D and 3D convolutions at SD1.5 UNet, SD1.5 VAE and Cosmos3 VAE shapes match a CPU convolution on
 *   sampled outputs; prints the GPU time and throughput of each
 * - matMul: batched products at CLIP, SD1.5 UNet and Cosmos3 shapes match the CPU on sampled outputs; prints
 *   the GPU time and throughput of each
 * - fusedAttention: the fused attention kernels at SD1.5 UNet and VAE shapes (head widths 40, 80, 160 and 512)
 *   and the Cosmos3 128-wide kernel with a key-padding bias match a CPU softmax(q kᵀ + bias) v on sampled
 *   queries; prints the GPU time of each
 * - elementwise: Mul and Add with equal shapes, a per-channel and a scalar operand, and Sigmoid at SD1.5 VAE
 *   and UNet shapes match the CPU on sampled elements; prints the GPU time and bandwidth of each
 * - normalization: instance normalization at SD1.5 VAE group-norm shapes and layer normalization and softmax at
 *   CLIP and UNet shapes match the CPU on sampled rows; prints the GPU time and bandwidth of each
 *
 * Run with KLARTRAUM_BENCHMARK=1; the tests are skipped otherwise. GPU times are the means of the kernel's
 * timestamps over several submissions after a warm-up.
 **/

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/headless_frontend.hpp"
#include "klartraum/layers/layers.hpp"

using namespace klartraum;
using layers::Shape;

namespace {

using Tensor = TensorElement<float>;

size_t count(const Shape& shape) {
    size_t n = 1;
    for (uint32_t d : shape)
        n *= d;
    return n;
}

// Deterministic values in [-scale, scale].
std::vector<float> pattern(size_t n, uint32_t seed, float scale) {
    std::vector<float> values(n);
    uint32_t state = seed * 747796405u + 2891336453u;
    for (size_t i = 0; i < n; ++i) {
        state = state * 1664525u + 1013904223u;
        values[i] = (float((state >> 8) & 0xffff) / 32767.5f - 1.0f) * scale;
    }
    return values;
}

// Indices of `samples` elements spread over [0, n), first and last included.
std::vector<size_t> sampleIndices(size_t n, size_t samples) {
    std::vector<size_t> indices;
    for (size_t s = 0; s < samples; ++s)
        indices.push_back(samples == 1 ? 0 : (n - 1) * s / (samples - 1));
    return indices;
}

class KernelBenchmark : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::getenv("KLARTRAUM_BENCHMARK"))
            GTEST_SKIP() << "Set KLARTRAUM_BENCHMARK=1 to run";
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
    }

    // Runs `layer` with `inputs` in its slots (then the output), and returns the output and the layer's mean GPU
    // time in milliseconds over `repeats` submissions after a warm-up.
    std::vector<float> run(const ComputeGraphElementPtr& layer, const std::vector<Shape>& inputShapes,
                           const std::vector<std::vector<float>>& inputs, const Shape& outputShape, double& ms,
                           int repeats = 10) {
        std::vector<std::shared_ptr<Tensor>> tensors;
        int slot = 0;
        for (const auto& shape : inputShapes) {
            tensors.push_back(vc->create<Tensor>(shape));
            layer->setInput(tensors.back(), slot++);
        }
        auto output = vc->create<Tensor>(outputShape);
        layer->setInput(output, slot);
        ComputeGraph graph(*vc, 1);
        graph.enableProfiling();
        graph.compileFrom(layer);
        for (size_t i = 0; i < inputs.size(); ++i)
            tensors[i]->getDataBuffer(0).memcopyFrom(inputs[i]);
        graph.submitAndWait(vc->getGraphicsQueue(), 0);
        std::vector<float> result(count(outputShape));
        output->getDataBuffer(0).memcopyTo(result);
        graph.resetProfiling();
        for (int i = 0; i < repeats; ++i)
            graph.submitAndWait(vc->getGraphicsQueue(), 0);
        const auto times = graph.getElementTimesMs();
        const auto& elements = graph.getElements();
        ms = 0.0;
        for (size_t i = 0; i < elements.size(); ++i)
            if (elements[i] == layer)
                ms = times[i];
        return result;
    }

    static void report(const std::string& name, double ms, double flops, double bytes) {
        std::cout << "  " << std::left << std::setw(48) << name << std::right << std::fixed << std::setprecision(3)
                  << std::setw(10) << ms << " ms";
        if (flops > 0)
            std::cout << std::setw(10) << std::setprecision(0) << flops / (ms * 1e6) << " GFLOP/s";
        if (bytes > 0)
            std::cout << std::setw(10) << std::setprecision(1) << bytes / (ms * 1e6) << " GB/s";
        std::cout << std::endl;
    }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
};

struct ConvCase {
    std::string name;
    uint32_t batch, inChannels, outChannels, depth, height, width;
    uint32_t kd, kh, kw, stride;
    bool causal; // temporal pads kd - 1 at the front only, else symmetric spatial pads of (k - 1) / 2
};

} // namespace

TEST_F(KernelBenchmark, conv) {
    const std::vector<ConvCase> cases = {
        {"unet 2560->1280 3x3 @4x4", 2, 2560, 1280, 1, 4, 4, 1, 3, 3, 1, false},
        {"unet 2560->1280 3x3 @8x8", 2, 2560, 1280, 1, 8, 8, 1, 3, 3, 1, false},
        {"unet 1920->640 3x3 @16x16", 2, 1920, 640, 1, 16, 16, 1, 3, 3, 1, false},
        {"unet 640->640 3x3 @16x16", 2, 640, 640, 1, 16, 16, 1, 3, 3, 1, false},
        {"unet 320->320 3x3 @32x32", 2, 320, 320, 1, 32, 32, 1, 3, 3, 1, false},
        {"unet 640->320 1x1 @32x32", 2, 640, 320, 1, 32, 32, 1, 1, 1, 1, false},
        {"unet 320->320 3x3 /2 @32x32", 2, 320, 320, 1, 32, 32, 1, 3, 3, 2, false},
        {"vae 512->512 3x3 @64x64", 1, 512, 512, 1, 64, 64, 1, 3, 3, 1, false},
        {"vae 256->256 3x3 @128x128", 1, 256, 256, 1, 128, 128, 1, 3, 3, 1, false},
        {"vae 128->128 3x3 @256x256", 1, 128, 128, 1, 256, 256, 1, 3, 3, 1, false},
        {"vae 128->3 3x3 @256x256", 1, 128, 3, 1, 256, 256, 1, 3, 3, 1, false},
        {"cosmos3 vae 384->384 3x3x3 causal T3 @32x32", 1, 384, 384, 3, 32, 32, 3, 3, 3, 1, true},
        {"cosmos3 vae 96->96 3x3x3 causal T3 @128x128", 1, 96, 96, 3, 128, 128, 3, 3, 3, 1, true},
    };
    std::cout << "conv:" << std::endl;
    for (const auto& c : cases) {
        const uint32_t ph = (c.kh - 1) / 2, pw = (c.kw - 1) / 2, pd = c.causal ? c.kd - 1 : (c.kd - 1) / 2;
        const uint32_t od = c.causal ? c.depth : (c.depth + 2 * pd - c.kd) / c.stride + 1;
        const uint32_t oh = (c.height + 2 * ph - c.kh) / c.stride + 1, ow = (c.width + 2 * pw - c.kw) / c.stride + 1;
        const Shape input{c.batch, c.inChannels, c.depth, c.height, c.width};
        const Shape weights{c.outChannels, c.inChannels, c.kd, c.kh, c.kw};
        const Shape bias{c.outChannels};
        const Shape output{c.batch, c.outChannels, od, oh, ow};
        ComputeGraphElementPtr layer;
        if (c.depth == 1 && c.kd == 1) {
            layers::ConvAttributes attributes;
            attributes.kernelShape = {c.kh, c.kw};
            attributes.strides = {c.stride, c.stride};
            attributes.pads = {ph, pw, ph, pw};
            layer = layers::conv(*vc, attributes, {c.batch, c.inChannels, c.height, c.width},
                                 {c.outChannels, c.inChannels, c.kh, c.kw}, bias, {c.batch, c.outChannels, oh, ow});
        } else {
            layers::Conv3dAttributes attributes;
            attributes.kernelShape = {c.kd, c.kh, c.kw};
            attributes.strides = {c.stride, c.stride, c.stride};
            attributes.pads = {pd, ph, pw, c.causal ? 0 : pd, ph, pw};
            layer = layers::conv3d(*vc, attributes, input, weights, output);
        }
        const auto x = pattern(count(input), 1, 1.0f);
        const auto w = pattern(count(weights), 2, 1.0f / std::sqrt(float(c.inChannels * c.kd * c.kh * c.kw)));
        const auto b = pattern(count(bias), 3, 0.1f);
        double ms = 0.0;
        const auto y = run(layer, {input, weights, bias}, {x, w, b}, output, ms, 5);
        const double flops = 2.0 * c.batch * c.outChannels * od * oh * ow * c.inChannels * c.kd * c.kh * c.kw;
        report(c.name, ms, flops, 0);
        for (size_t index : sampleIndices(y.size(), 97)) {
            size_t rest = index;
            const uint32_t ox = rest % ow;
            rest /= ow;
            const uint32_t oy = rest % oh;
            rest /= oh;
            const uint32_t oz = rest % od;
            rest /= od;
            const uint32_t o = rest % c.outChannels, n = uint32_t(rest / c.outChannels);
            double expected = b[o];
            for (uint32_t i = 0; i < c.inChannels; ++i)
                for (uint32_t dz = 0; dz < c.kd; ++dz)
                    for (uint32_t dy = 0; dy < c.kh; ++dy)
                        for (uint32_t dx = 0; dx < c.kw; ++dx) {
                            const int z = int(oz * c.stride + dz) - int(pd), yy = int(oy * c.stride + dy) - int(ph),
                                      xx = int(ox * c.stride + dx) - int(pw);
                            if (z < 0 || yy < 0 || xx < 0 || z >= int(c.depth) || yy >= int(c.height) ||
                                xx >= int(c.width))
                                continue;
                            expected +=
                                double(x[(((size_t(n) * c.inChannels + i) * c.depth + z) * c.height + yy) * c.width +
                                         xx]) *
                                w[(((size_t(o) * c.inChannels + i) * c.kd + dz) * c.kh + dy) * c.kw + dx];
                        }
            ASSERT_NEAR(y[index], expected, 2e-4 + 2e-4 * std::abs(expected)) << c.name << " element " << index;
        }
    }
}

TEST_F(KernelBenchmark, matMul) {
    struct Case {
        std::string name;
        uint32_t batch, rows, reduction, columns;
    };
    const std::vector<Case> cases = {
        {"clip fc1 [154,768]x[768,3072]", 1, 154, 768, 3072},
        {"clip fc2 [154,3072]x[3072,768]", 1, 154, 3072, 768},
        {"unet proj [2048,320]x[320,320]", 1, 2048, 320, 320},
        {"unet ff [2048,320]x[320,2560]", 1, 2048, 320, 2560},
        {"unet ff [128,1280]x[1280,10240]", 1, 128, 1280, 10240},
        {"unet attn scores 16x[1024,40]x[40,1024]", 16, 1024, 40, 1024},
        {"cosmos3 [1152,2048]x[2048,2048]", 1, 1152, 2048, 2048},
        {"cosmos3 mlp [1152,2048]x[2048,8192]", 1, 1152, 2048, 8192},
    };
    std::cout << "matMul:" << std::endl;
    for (const auto& c : cases) {
        const Shape lhs{c.batch, c.rows, c.reduction}, rhs{c.batch, c.reduction, c.columns},
            output{c.batch, c.rows, c.columns};
        const auto l = pattern(count(lhs), 4, 1.0f), r = pattern(count(rhs), 5, 1.0f / std::sqrt(float(c.reduction)));
        double ms = 0.0;
        const auto y = run(layers::matMul(*vc, lhs, rhs, output), {lhs, rhs}, {l, r}, output, ms, 5);
        report(c.name, ms, 2.0 * c.batch * c.rows * c.reduction * c.columns, 0);
        for (size_t index : sampleIndices(y.size(), 97)) {
            const size_t b = index / (size_t(c.rows) * c.columns), i = index / c.columns % c.rows,
                         j = index % c.columns;
            double expected = 0.0;
            for (uint32_t k = 0; k < c.reduction; ++k)
                expected += double(l[(b * c.rows + i) * c.reduction + k]) * r[(b * c.reduction + k) * c.columns + j];
            ASSERT_NEAR(y[index], expected, 2e-4 + 2e-4 * std::abs(expected)) << c.name << " element " << index;
        }
    }
}

TEST_F(KernelBenchmark, fusedAttention) {
    struct Case {
        std::string name;
        uint32_t batch, heads, queries, keys, depth;
        bool bias;
    };
    const std::vector<Case> cases = {
        {"unet self d40 [2x8, 1024 q, 1024 k]", 2, 8, 1024, 1024, 40, false},
        {"unet cross d40 [2x8, 1024 q, 77 k]", 2, 8, 1024, 77, 40, false},
        {"unet self d80 [2x8, 256 q, 256 k]", 2, 8, 256, 256, 80, false},
        {"unet self d160 [2x8, 64 q, 64 k]", 2, 8, 64, 64, 160, false},
        {"vae mid d512 [1x1, 1024 q, 1024 k]", 1, 1, 1024, 1024, 512, false},
        {"cosmos3 d128 bias [2x16, 1152 q, 1152 k]", 2, 16, 1152, 1152, 128, true},
        {"cosmos3 cross d128 bias [2x16, 1152 q, 512 k]", 2, 16, 1152, 512, 128, true},
    };
    std::cout << "fusedAttention:" << std::endl;
    for (const auto& c : cases) {
        const Shape q{c.batch, c.heads, c.queries, c.depth}, k{c.batch, c.heads, c.depth, c.keys},
            v{c.batch, c.heads, c.keys, c.depth}, biasShape{c.batch, 1, 1, c.keys};
        const float scale = 1.0f / std::sqrt(float(c.depth));
        auto qv = pattern(count(q), 6, 2.0f * scale), kv = pattern(count(k), 7, 2.0f), vv = pattern(count(v), 8, 1.0f);
        std::vector<float> bv(count(biasShape), 0.0f);
        if (c.bias)
            for (uint32_t key = c.keys - c.keys / 5; key < c.keys; ++key)
                bv[key] = -1e9f; // the first batch's text padding
        double ms = 0.0;
        const auto y = c.bias ? run(layers::fusedAttentionWithBias(*vc, q, k, v, biasShape, q), {q, k, v, biasShape},
                                    {qv, kv, vv, bv}, q, ms)
                              : run(layers::fusedAttention(*vc, q, k, v, q), {q, k, v}, {qv, kv, vv}, q, ms);
        const double flops = 4.0 * c.batch * c.heads * c.queries * c.keys * c.depth;
        report(c.name, ms, flops, 0);
        for (size_t row : sampleIndices(size_t(c.batch) * c.heads * c.queries, 31)) {
            const size_t bh = row / c.queries, b = bh / c.heads, query = row % c.queries;
            std::vector<double> scores(c.keys);
            double maximum = -1e300;
            for (uint32_t key = 0; key < c.keys; ++key) {
                double s = c.bias ? bv[b * c.keys + key] : 0.0;
                for (uint32_t d = 0; d < c.depth; ++d)
                    s += double(qv[(bh * c.queries + query) * c.depth + d]) * kv[(bh * c.depth + d) * c.keys + key];
                scores[key] = s;
                maximum = std::max(maximum, s);
            }
            double sum = 0.0;
            for (auto& s : scores)
                sum += (s = std::exp(s - maximum));
            for (uint32_t d = 0; d < c.depth; d += 7) {
                double expected = 0.0;
                for (uint32_t key = 0; key < c.keys; ++key)
                    expected += scores[key] / sum * vv[(bh * c.keys + key) * c.depth + d];
                ASSERT_NEAR(y[(bh * c.queries + query) * c.depth + d], expected, 2e-4) << c.name << " row " << row;
            }
        }
    }
}

TEST_F(KernelBenchmark, normalization) {
    std::cout << "normalization:" << std::endl;
    struct Case {
        std::string name;
        int kind; // 0 instance norm, 1 layer norm, 2 softmax
        Shape shape;
    };
    const std::vector<Case> cases = {
        {"vae group norm 512ch @64x64 [1,32,65536]", 0, {1, 32, 65536}},
        {"vae group norm 256ch @128x128 [1,32,131072]", 0, {1, 32, 131072}},
        {"vae group norm 128ch @256x256 [1,32,262144]", 0, {1, 32, 262144}},
        {"unet group norm 320ch @32x32 [2,32,10240]", 0, {2, 32, 10240}},
        {"clip layer norm [154,768]", 1, {154, 768}},
        {"unet layer norm [2048,320]", 1, {2048, 320}},
        {"unet layer norm [128,1280]", 1, {128, 1280}},
        {"softmax [16384,1024]", 2, {16384, 1024}},
        {"softmax [16384,77]", 2, {16384, 77}},
    };
    for (const auto& c : cases) {
        const uint32_t rowLength = c.shape.back();
        const size_t rows = count(c.shape) / rowLength;
        const auto x = pattern(count(c.shape), 9, 2.0f);
        const Shape affine{c.kind == 0 ? c.shape[1] : rowLength};
        const auto scale = pattern(affine[0], 10, 1.0f), shift = pattern(affine[0], 11, 0.5f);
        double ms = 0.0;
        std::vector<float> y;
        if (c.kind == 0)
            y = run(layers::instanceNormalization(*vc, c.shape), {c.shape, affine, affine}, {x, scale, shift}, c.shape,
                    ms);
        else if (c.kind == 1)
            y = run(layers::layerNormalization(*vc, c.shape), {c.shape, affine, affine}, {x, scale, shift}, c.shape,
                    ms);
        else
            y = run(layers::softmax(*vc, c.shape), {c.shape}, {x}, c.shape, ms);
        report(c.name, ms, 0, 2.0 * count(c.shape) * sizeof(float));
        for (size_t row : sampleIndices(rows, 17)) {
            const float* in = &x[row * rowLength];
            double mean = 0.0, squares = 0.0, maximum = -1e300;
            for (uint32_t i = 0; i < rowLength; ++i) {
                mean += in[i];
                maximum = std::max(maximum, double(in[i]));
            }
            mean /= rowLength;
            for (uint32_t i = 0; i < rowLength; ++i)
                squares += (in[i] - mean) * (in[i] - mean);
            const double inverse = 1.0 / std::sqrt(squares / rowLength + 1e-5);
            double sum = 0.0;
            for (uint32_t i = 0; i < rowLength; ++i)
                sum += std::exp(in[i] - maximum);
            for (uint32_t i = 0; i < rowLength; i += 13) {
                const uint32_t a = c.kind == 0 ? uint32_t(row % c.shape[1]) : i;
                const double expected =
                    c.kind == 2 ? std::exp(in[i] - maximum) / sum : (in[i] - mean) * inverse * scale[a] + shift[a];
                ASSERT_NEAR(y[row * rowLength + i], expected, 2e-4 + 2e-4 * std::abs(expected))
                    << c.name << " row " << row;
            }
        }
    }
}

TEST_F(KernelBenchmark, elementwise) {
    std::cout << "elementwise:" << std::endl;
    struct Case {
        std::string name;
        int kind; // 0 Mul, 1 Add, 2 Sigmoid
        Shape lhs, rhs;
    };
    const std::vector<Case> cases = {
        {"vae mul equal [1,128,256,256]", 0, {1, 128, 256, 256}, {1, 128, 256, 256}},
        {"vae mul per-channel [1,128,256,256]x[128,1,1]", 0, {1, 128, 256, 256}, {128, 1, 1}},
        {"vae add equal [1,256,128,128]", 1, {1, 256, 128, 128}, {1, 256, 128, 128}},
        {"unet add scalar [2,320,32,32]+[]", 1, {2, 320, 32, 32}, {1}},
        {"vae sigmoid [1,128,256,256]", 2, {1, 128, 256, 256}, {}},
    };
    for (const auto& c : cases) {
        const auto x = pattern(count(c.lhs), 12, 2.0f);
        const auto y = c.rhs.empty() ? std::vector<float>{} : pattern(count(c.rhs), 13, 2.0f);
        double ms = 0.0;
        std::vector<float> result;
        if (c.kind == 2) {
            result = run(layers::unary(*vc, layers::UnaryOp::Sigmoid, c.lhs), {c.lhs}, {x}, c.lhs, ms);
        } else {
            const auto op = c.kind == 0 ? layers::BinaryOp::Mul : layers::BinaryOp::Add;
            result = run(layers::binary(*vc, op, c.lhs, c.rhs, c.lhs), {c.lhs, c.rhs}, {x, y}, c.lhs, ms);
        }
        report(c.name, ms, 0, (c.kind == 2 ? 2.0 : 3.0) * count(c.lhs) * sizeof(float));
        const size_t inner = c.rhs.size() == 3 ? size_t(c.lhs[2]) * c.lhs[3] : 1;
        for (size_t i : sampleIndices(result.size(), 101)) {
            float expected;
            if (c.kind == 2) {
                expected = 1.0f / (1.0f + std::exp(-x[i]));
            } else {
                const float other = y.size() == x.size() ? y[i] : y.size() == 1 ? y[0] : y[i / inner % y.size()];
                expected = c.kind == 0 ? x[i] * other : x[i] + other;
            }
            ASSERT_NEAR(result[i], expected, 1e-6f + 1e-6f * std::abs(expected)) << c.name << " element " << i;
        }
    }
}
