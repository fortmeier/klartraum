// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - layerNormModulateMatchesCpu: every token's row is normalized (mean 0, variance 1 up to epsilon) and modulated
 *   by the modulation vector's scale and shift at their offsets, as a CPU evaluation
 * - gatedResidualAddsGatedValues: x += gate * y with the gate at its offset in the modulation vector, for a count
 *   that is not a multiple of the workgroup
 * - activationsMatchCpu: GELU with the tanh approximation and SiLU, in place, as a CPU evaluation
 * - rotaryTurnsPairs: every pair of consecutive values of every head is turned by its token's angle (complex
 *   multiplication), and the pairs' lengths stay
 **/

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/diffusion/dit_layers.hpp"
#include "klartraum/headless_frontend.hpp"

using namespace klartraum;
namespace dl = klartraum::dit_layers;

namespace {

class DitLayersTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
    }

    std::shared_ptr<TensorElement<float>> tensor(const std::vector<float>& values) {
        auto t = vc->create<TensorElement<float>>(std::vector<uint32_t>{uint32_t(values.size())});
        pending.push_back({t, values});
        return t;
    }

    // Compiles a graph of `layer` with `slots` in order, uploads the tensors' values and runs it once.
    void run(const ComputeGraphElementPtr& layer, const std::vector<std::shared_ptr<TensorElement<float>>>& slots) {
        for (size_t i = 0; i < slots.size(); ++i)
            layer->setInput(slots[i], int(i));
        ComputeGraph graph(*vc, 1);
        graph.compileFrom(layer);
        for (auto& [t, values] : pending)
            t->getDataBuffer(0).memcopyFrom(values);
        pending.clear();
        graph.submitAndWait(vc->getGraphicsQueue(), 0);
    }

    static std::vector<float> read(const std::shared_ptr<TensorElement<float>>& t) {
        std::vector<float> values(t->getDataElementCount());
        t->getDataBuffer(0).memcopyTo(values);
        return values;
    }

    std::vector<float> random(size_t count, float scale = 1.0f) {
        std::uniform_real_distribution<float> distribution(-scale, scale);
        std::vector<float> values(count);
        for (auto& v : values)
            v = distribution(rng);
        return values;
    }

    std::mt19937 rng{7};
    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
    std::vector<std::pair<std::shared_ptr<TensorElement<float>>, std::vector<float>>> pending;
};

} // namespace

TEST_F(DitLayersTest, layerNormModulateMatchesCpu) {
    constexpr uint32_t width = 3072, tokens = 5, shift = 100, scale = 100 + width;
    const auto x = random(width * tokens, 3.0f), modulation = random(2 * width + 300);
    auto input = tensor(x), mod = tensor(modulation), output = tensor(std::vector<float>(x.size(), 0.0f));
    run(dl::layerNormModulate(*vc, width, tokens, shift, scale), {input, mod, output});
    const auto y = read(output);
    for (uint32_t t = 0; t < tokens; ++t) {
        double mean = 0.0, variance = 0.0;
        for (uint32_t i = 0; i < width; ++i)
            mean += x[t * width + i];
        mean /= width;
        for (uint32_t i = 0; i < width; ++i)
            variance += (x[t * width + i] - mean) * (x[t * width + i] - mean);
        variance /= width;
        for (uint32_t i = 0; i < width; ++i) {
            const double normed = (x[t * width + i] - mean) / std::sqrt(variance + 1e-6);
            const double expected = normed * (1.0 + modulation[scale + i]) + modulation[shift + i];
            ASSERT_NEAR(y[t * width + i], expected, 1e-4) << "token " << t << " value " << i;
        }
    }
}

TEST_F(DitLayersTest, gatedResidualAddsGatedValues) {
    constexpr uint32_t width = 100, tokens = 7, gate = 33;
    const auto x = random(width * tokens), y = random(width * tokens), modulation = random(200);
    auto target = tensor(x), values = tensor(y), mod = tensor(modulation);
    run(dl::gatedResidual(*vc, width, tokens, gate), {values, mod, target});
    const auto result = read(target);
    for (uint32_t i = 0; i < width * tokens; ++i)
        ASSERT_NEAR(result[i], x[i] + modulation[gate + i % width] * y[i], 1e-6) << i;
}

TEST_F(DitLayersTest, activationsMatchCpu) {
    const auto x = random(1000, 5.0f);
    for (const auto kind : {dl::Activation::GeluTanh, dl::Activation::Silu}) {
        auto values = tensor(x);
        run(dl::activation(*vc, kind, uint32_t(x.size())), {values});
        const auto y = read(values);
        for (size_t i = 0; i < x.size(); ++i) {
            const double v = x[i];
            const double expected =
                kind == dl::Activation::GeluTanh
                    ? 0.5 * v * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (v + 0.044715 * v * v * v)))
                    : v / (1.0 + std::exp(-v));
            ASSERT_NEAR(y[i], expected, 1e-5) << i;
        }
    }
}

TEST_F(DitLayersTest, rotaryTurnsPairs) {
    constexpr uint32_t tokens = 6, heads = 3;
    const auto x = random(tokens * heads * 128);
    std::vector<float> cosines(tokens * 64), sines(tokens * 64), angles(tokens * 64);
    for (uint32_t i = 0; i < tokens * 64; ++i) {
        angles[i] = 0.37f * float(i % 64) * float(i / 64 + 1);
        cosines[i] = std::cos(angles[i]);
        sines[i] = std::sin(angles[i]);
    }
    auto values = tensor(x), c = tensor(cosines), s = tensor(sines);
    run(dl::rotary(*vc, tokens, heads), {c, s, values});
    const auto y = read(values);
    for (uint32_t t = 0; t < tokens; ++t)
        for (uint32_t h = 0; h < heads; ++h)
            for (uint32_t p = 0; p < 64; ++p) {
                const size_t i = (size_t(t) * heads + h) * 128 + 2 * p;
                const double a = angles[t * 64 + p];
                ASSERT_NEAR(y[i], x[i] * std::cos(a) - x[i + 1] * std::sin(a), 1e-5);
                ASSERT_NEAR(y[i + 1], x[i] * std::sin(a) + x[i + 1] * std::cos(a), 1e-5);
            }
}
