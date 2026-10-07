// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - MatVecAllWeightTypes: matVec of every supported weight type matches a CPU product of
 *   the dequantized rows, for 3 active tokens and a row count that is not a multiple of
 *   the workgroup's rows
 * - MatVecAccumulateAndLastToken: matVec with accumulate adds to the output in place;
 *   with lastTokenOnly it computes the last token only
 * - RmsNormRowsPerToken: rmsNorm normalizes several rows per token and leaves inactive
 *   tokens untouched
 * - SwiGlu: swiGlu computes silu(gate) * up for the active tokens
 * - AttentionWithCacheAcrossSubmissions: attentionPrep + attention over two submissions
 *   (a 3-token chunk, then 1 token) match a CPU gated causal GQA attention with per-head
 *   RMS norms and NEOX RoPE
 * - LinearConvStateAcrossSubmissions: linearConv carries its state across submissions: 2
 *   + 3 tokens equal 5 tokens at once
 * - GatedDeltaNetAcrossSubmissions: gatedDeltaNet over two submissions matches a CPU
 *   gated delta rule with the gated RMS norm
 * - MatVecThroughput: matVecThroughput (only with KLARTRAUM_BENCHMARK set): prints the
 *   weight bandwidth of each type at the Qwen3.6-27B FFN shape, for 1 and 16 tokens
 **/

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/gguf/gguf_layers.hpp"
#include "klartraum/gguf/gguf_quants.hpp"
#include "klartraum/headless_frontend.hpp"

using namespace klartraum;
namespace gl = klartraum::gguf_layers;

namespace {

using FloatTensor = TensorElement<float>;
using UintTensor = TensorElement<uint32_t>;

float silu(float x) { return x / (1.0f + std::exp(-x)); }
float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

class GgufLayersTest : public ::testing::Test {
protected:
    void SetUp() override {
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
        params = vc->create<UintTensor>(std::vector<uint32_t>{4});
    }

    std::shared_ptr<FloatTensor> tensor(size_t count) {
        return vc->create<FloatTensor>(std::vector<uint32_t>{uint32_t(count)});
    }

    static void connect(const ComputeGraphElementPtr& layer, const std::vector<ComputeGraphElementPtr>& slots) {
        for (size_t i = 0; i < slots.size(); ++i)
            layer->setInput(slots[i], int(i));
    }

    // Connects `slots` in order to `layer` and compiles a one-layer graph.
    std::unique_ptr<ComputeGraph> compile(const ComputeGraphElementPtr& layer,
                                          const std::vector<ComputeGraphElementPtr>& slots) {
        connect(layer, slots);
        auto graph = std::make_unique<ComputeGraph>(*vc, 1);
        graph->compileFrom(layer);
        return graph;
    }

    void run(ComputeGraph& graph, uint32_t tokens, uint32_t position) {
        params->setData(0, {tokens, position, 0, 0});
        graph.submitAndWait(vc->getGraphicsQueue(), 0);
    }

    static std::vector<float> read(const std::shared_ptr<FloatTensor>& tensor) {
        std::vector<float> values(tensor->getDataElementCount());
        tensor->getDataBuffer(0).memcopyTo(values);
        return values;
    }

    std::vector<float> random(size_t count, float scale = 1.0f) {
        std::uniform_real_distribution<float> distribution(-scale, scale);
        std::vector<float> values(count);
        for (auto& value : values)
            value = distribution(rng);
        return values;
    }

    // `rows` rows of random blocks of `type`; half-float scales are finite.
    std::vector<uint8_t> randomWeights(GgmlType type, uint32_t rows, uint32_t columns) {
        const uint64_t rowBytes = ggmlRowBytes(type, columns);
        std::vector<uint8_t> bytes(rows * rowBytes);
        std::uniform_int_distribution<int> byte(0, 255);
        for (auto& b : bytes)
            b = uint8_t(byte(rng));
        auto setHalf = [&](size_t offset) {
            const uint16_t bits = floatToHalf(random(1, 0.05f)[0]);
            std::memcpy(&bytes[offset], &bits, 2);
        };
        const uint32_t blockBytes = ggmlBlockBytes(type);
        if (type == GgmlType::F16) {
            for (size_t i = 0; i < bytes.size(); i += 2)
                setHalf(i);
        } else if (type == GgmlType::F32) {
            const auto values = random(bytes.size() / 4);
            std::memcpy(bytes.data(), values.data(), bytes.size());
        } else if (type == GgmlType::BF16) {
            for (size_t i = 0; i < bytes.size(); i += 2)
                bytes[i + 1] = uint8_t(0x3c + byte(rng) % 4);
        } else {
            for (size_t block = 0; block < bytes.size(); block += blockBytes) {
                switch (type) {
                case GgmlType::Q8_0:
                    setHalf(block);
                    break;
                case GgmlType::Q3_K:
                    setHalf(block + 108);
                    break;
                case GgmlType::Q4_K:
                case GgmlType::Q5_K:
                    setHalf(block);
                    setHalf(block + 2);
                    break;
                case GgmlType::Q6_K:
                    setHalf(block + 208);
                    break;
                default:
                    break;
                }
            }
        }
        return bytes;
    }

    // A weight tensor holding `bytes` (GGUF rows) in the GPU block layout, uploaded by uploadWeights().
    std::shared_ptr<UintTensor> weightTensor(GgmlType type, const std::vector<uint8_t>& bytes, uint32_t rows,
                                             uint32_t columns) {
        std::vector<uint8_t> packed(gl::gpuRowBytes(type, columns) * rows);
        gl::packWeights(type, bytes.data(), rows, columns, packed.data());
        auto weights =
            vc->create<UintTensor>(std::vector<uint32_t>{uint32_t((packed.size() + gl::kWeightPaddingBytes + 3) / 4)});
        pendingWeights.push_back({weights, packed});
        return weights;
    }

    void uploadWeights() {
        for (auto& [weights, bytes] : pendingWeights) {
            std::vector<uint32_t> words(weights->getDataElementCount(), 0);
            std::memcpy(words.data(), bytes.data(), bytes.size());
            weights->setData(0, words);
        }
        pendingWeights.clear();
    }

    std::mt19937 rng{42};
    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
    std::shared_ptr<UintTensor> params;
    std::vector<std::pair<std::shared_ptr<UintTensor>, std::vector<uint8_t>>> pendingWeights;
};

// y[t][r] = sum_c W[r][c] x[t][c] with W decoded on the CPU.
std::vector<float> cpuMatVec(GgmlType type, const std::vector<uint8_t>& bytes, uint32_t rows, uint32_t columns,
                             const std::vector<float>& x, uint32_t tokens) {
    const uint64_t rowBytes = ggmlRowBytes(type, columns);
    std::vector<float> row(columns), y(size_t(tokens) * rows);
    for (uint32_t r = 0; r < rows; ++r) {
        dequantizeRow(type, bytes.data() + r * rowBytes, row.data(), columns);
        for (uint32_t t = 0; t < tokens; ++t) {
            double sum = 0.0;
            for (uint32_t c = 0; c < columns; ++c)
                sum += double(row[c]) * x[size_t(t) * columns + c];
            y[size_t(t) * rows + r] = float(sum);
        }
    }
    return y;
}

void expectClose(const std::vector<float>& actual, const std::vector<float>& expected, float tolerance,
                 size_t count = size_t(-1)) {
    count = std::min({count, actual.size(), expected.size()});
    float scale = 0.0f;
    for (size_t i = 0; i < count; ++i)
        scale = std::max(scale, std::fabs(expected[i]));
    for (size_t i = 0; i < count; ++i) {
        ASSERT_NEAR(actual[i], expected[i], tolerance * std::max(1.0f, scale)) << "index " << i;
    }
}

} // namespace

TEST_F(GgufLayersTest, MatVecAllWeightTypes) {
    constexpr uint32_t rows = 13, columns = 512, tokens = 3;
    for (GgmlType type : {GgmlType::F32, GgmlType::F16, GgmlType::BF16, GgmlType::Q8_0, GgmlType::Q3_K, GgmlType::Q4_K,
                          GgmlType::Q5_K, GgmlType::Q6_K}) {
        SCOPED_TRACE(ggmlTypeName(type));
        const auto bytes = randomWeights(type, rows, columns);
        auto weights = weightTensor(type, bytes, rows, columns);
        auto input = tensor(gl::kMaxTokens * columns);
        auto output = tensor(gl::kMaxTokens * rows);
        auto graph = compile(gl::matVec(*vc, type, rows, columns, columns, rows), {weights, input, params, output});
        uploadWeights();
        const auto x = random(gl::kMaxTokens * columns);
        input->setData(0, x);
        output->setData(0, std::vector<float>(gl::kMaxTokens * rows, 123.0f));
        run(*graph, tokens, 0);
        const auto y = read(output);
        expectClose(y, cpuMatVec(type, bytes, rows, columns, x, tokens), 1e-4f, tokens * rows);
        EXPECT_EQ(y[tokens * rows], 123.0f) << "inactive tokens stay untouched";
    }
}

TEST_F(GgufLayersTest, MatVecAccumulateAndLastToken) {
    constexpr uint32_t rows = 8, columns = 256, tokens = 4;
    const auto bytes = randomWeights(GgmlType::Q4_K, rows, columns);
    auto weights = weightTensor(GgmlType::Q4_K, bytes, rows, columns);
    auto input = tensor(gl::kMaxTokens * columns);
    auto residual = tensor(gl::kMaxTokens * rows);
    auto last = tensor(rows);
    auto accumulate = gl::matVec(*vc, GgmlType::Q4_K, rows, columns, columns, rows, true);
    auto lastToken = gl::matVec(*vc, GgmlType::Q4_K, rows, columns, columns, rows, false, true);
    connect(accumulate, {weights, input, params, residual});
    connect(lastToken, {weights, input, params, last});
    lastToken->addDependency(accumulate);
    ComputeGraph graph(*vc, 1);
    graph.compileFrom(lastToken);
    uploadWeights();
    const auto x = random(gl::kMaxTokens * columns);
    const auto start = random(gl::kMaxTokens * rows);
    input->setData(0, x);
    residual->setData(0, start);
    run(graph, tokens, 0);
    const auto expected = cpuMatVec(GgmlType::Q4_K, bytes, rows, columns, x, tokens);
    auto sum = expected;
    for (size_t i = 0; i < sum.size(); ++i)
        sum[i] += start[i];
    expectClose(read(residual), sum, 1e-4f, tokens * rows);
    expectClose(read(last), std::vector<float>(expected.end() - rows, expected.end()), 1e-4f);
}

TEST_F(GgufLayersTest, RmsNormRowsPerToken) {
    constexpr uint32_t width = 96, rowsPerToken = 3, tokens = 2, maxTokens = 4;
    auto input = tensor(maxTokens * rowsPerToken * width);
    auto scale = tensor(width);
    auto output = tensor(maxTokens * rowsPerToken * width);
    auto graph = compile(gl::rmsNorm(*vc, width, rowsPerToken, 1e-6f, maxTokens), {input, scale, params, output});
    const auto x = random(maxTokens * rowsPerToken * width), w = random(width);
    input->setData(0, x);
    scale->setData(0, w);
    output->setData(0, std::vector<float>(maxTokens * rowsPerToken * width, 7.0f));
    run(*graph, tokens, 0);
    std::vector<float> expected(x.size(), 7.0f);
    for (uint32_t row = 0; row < tokens * rowsPerToken; ++row) {
        double squares = 0.0;
        for (uint32_t i = 0; i < width; ++i)
            squares += double(x[row * width + i]) * x[row * width + i];
        const float inverse = 1.0f / std::sqrt(float(squares / width) + 1e-6f);
        for (uint32_t i = 0; i < width; ++i)
            expected[row * width + i] = x[row * width + i] * inverse * w[i];
    }
    expectClose(read(output), expected, 1e-5f);
}

TEST_F(GgufLayersTest, SwiGlu) {
    constexpr uint32_t width = 300, tokens = 3, maxTokens = 4;
    auto gate = tensor(maxTokens * width), up = tensor(maxTokens * width), output = tensor(maxTokens * width);
    auto graph = compile(gl::swiGlu(*vc, width, maxTokens), {gate, up, params, output});
    const auto g = random(maxTokens * width, 4.0f), u = random(maxTokens * width);
    gate->setData(0, g);
    up->setData(0, u);
    output->setData(0, std::vector<float>(maxTokens * width, 0.0f));
    run(*graph, tokens, 0);
    std::vector<float> expected(g.size(), 0.0f);
    for (uint32_t i = 0; i < tokens * width; ++i)
        expected[i] = silu(g[i]) * u[i];
    expectClose(read(output), expected, 1e-5f);
}

TEST_F(GgufLayersTest, AttentionWithCacheAcrossSubmissions) {
    gl::AttentionShape shape;
    shape.headDim = 64;
    shape.queryHeads = 4;
    shape.kvHeads = 2;
    shape.ropeDims = 16;
    shape.contextLength = 16;
    shape.ropeBase = 10000.0f;
    constexpr uint32_t maxTokens = 4;
    const uint32_t D = shape.headDim, Hq = shape.queryHeads, Hkv = shape.kvHeads;

    auto queryGate = tensor(maxTokens * Hq * 2 * D), keys = tensor(maxTokens * Hkv * D),
         values = tensor(maxTokens * Hkv * D), queryNorm = tensor(D), keyNorm = tensor(D),
         queries = tensor(maxTokens * Hq * D), keyCache = tensor(Hkv * shape.contextLength * D),
         valueCache = tensor(Hkv * shape.contextLength * D), output = tensor(maxTokens * Hq * D);
    auto prep = gl::attentionPrep(*vc, shape, maxTokens);
    auto attend = gl::attention(*vc, shape, maxTokens);
    connect(prep, {queryGate, keys, values, queryNorm, keyNorm, params, queries, keyCache, valueCache});
    connect(attend, {queries, keyCache, valueCache, queryGate, params, output});
    attend->addDependency(prep);
    ComputeGraph graph(*vc, 1);
    graph.compileFrom(attend);
    const auto qn = random(D, 1.5f), kn = random(D, 1.5f);
    queryNorm->setData(0, qn);
    keyNorm->setData(0, kn);

    // CPU state: normalized, rotated keys and values per position.
    std::vector<std::vector<float>> cpuKeys(Hkv), cpuValues(Hkv);
    auto normRope = [&](const float* x, const std::vector<float>& weight, uint32_t position) {
        double squares = 0.0;
        for (uint32_t d = 0; d < D; ++d)
            squares += double(x[d]) * x[d];
        const float inverse = 1.0f / std::sqrt(float(squares / D) + shape.epsilon);
        std::vector<float> n(D), r(D);
        for (uint32_t d = 0; d < D; ++d)
            n[d] = x[d] * inverse * weight[d];
        r = n;
        const uint32_t half = shape.ropeDims / 2;
        for (uint32_t i = 0; i < half; ++i) {
            const double angle = position * std::pow(double(shape.ropeBase), -2.0 * i / shape.ropeDims);
            const float c = float(std::cos(angle)), s = float(std::sin(angle));
            r[i] = n[i] * c - n[i + half] * s;
            r[i + half] = n[i + half] * c + n[i] * s;
        }
        return r;
    };

    uint32_t position = 0;
    for (uint32_t tokens : {3u, 1u}) {
        const auto qg = random(maxTokens * Hq * 2 * D), k = random(maxTokens * Hkv * D),
                   v = random(maxTokens * Hkv * D);
        queryGate->setData(0, qg);
        keys->setData(0, k);
        values->setData(0, v);
        run(graph, tokens, position);

        std::vector<float> expected(maxTokens * Hq * D, 0.0f);
        for (uint32_t t = 0; t < tokens; ++t) {
            for (uint32_t h = 0; h < Hkv; ++h) {
                const auto key = normRope(&k[(t * Hkv + h) * D], kn, position + t);
                cpuKeys[h].insert(cpuKeys[h].end(), key.begin(), key.end());
                cpuValues[h].insert(cpuValues[h].end(), &v[(t * Hkv + h) * D], &v[(t * Hkv + h) * D] + D);
            }
        }
        for (uint32_t t = 0; t < tokens; ++t) {
            for (uint32_t h = 0; h < Hq; ++h) {
                const uint32_t kvHead = h / (Hq / Hkv);
                const auto q = normRope(&qg[(t * Hq + h) * 2 * D], qn, position + t);
                const uint32_t keyCount = position + t + 1;
                std::vector<double> scores(keyCount);
                double maximum = -1e30;
                for (uint32_t j = 0; j < keyCount; ++j) {
                    double dot = 0.0;
                    for (uint32_t d = 0; d < D; ++d)
                        dot += double(q[d]) * cpuKeys[kvHead][j * D + d];
                    scores[j] = dot / std::sqrt(double(D));
                    maximum = std::max(maximum, scores[j]);
                }
                double total = 0.0;
                for (auto& s : scores)
                    total += (s = std::exp(s - maximum));
                for (uint32_t d = 0; d < D; ++d) {
                    double value = 0.0;
                    for (uint32_t j = 0; j < keyCount; ++j)
                        value += scores[j] * cpuValues[kvHead][j * D + d];
                    const float gate = qg[((t * Hq + h) * 2 + 1) * D + d];
                    expected[(t * Hq + h) * D + d] = float(value / total) * sigmoid(gate);
                }
            }
        }
        SCOPED_TRACE(position);
        expectClose(read(output), expected, 1e-4f, tokens * Hq * D);
        position += tokens;
    }
}

TEST_F(GgufLayersTest, LinearConvStateAcrossSubmissions) {
    constexpr uint32_t channels = 100, maxTokens = 5;
    auto input = tensor(maxTokens * channels), weights = tensor(channels * 4), state = tensor(3 * channels),
         output = tensor(maxTokens * channels);
    auto graph = compile(gl::linearConv(*vc, channels), {input, weights, params, state, output});
    const auto x = random(maxTokens * channels), w = random(channels * 4), initial = random(3 * channels);
    weights->setData(0, w);

    auto reference = [&](uint32_t first, uint32_t count) {
        std::vector<float> y(count * channels);
        for (uint32_t c = 0; c < channels; ++c) {
            std::vector<float> history{initial[c], initial[channels + c], initial[2 * channels + c]};
            for (uint32_t t = 0; t < first + count; ++t) {
                const float xt = x[t * channels + c];
                const float sum = w[c * 4] * history[t] + w[c * 4 + 1] * history[t + 1] +
                                  w[c * 4 + 2] * history[t + 2] + w[c * 4 + 3] * xt;
                history.push_back(xt);
                if (t >= first)
                    y[(t - first) * channels + c] = silu(sum);
            }
        }
        return y;
    };

    state->setData(0, initial);
    input->setData(0, x);
    run(*graph, 2, 0);
    expectClose(read(output), reference(0, 2), 1e-5f, 2 * channels);
    std::vector<float> rest(maxTokens * channels, 0.0f);
    std::copy(x.begin() + 2 * channels, x.end(), rest.begin());
    input->setData(0, rest);
    run(*graph, 3, 2);
    expectClose(read(output), reference(2, 3), 1e-5f, 3 * channels);
}

TEST_F(GgufLayersTest, GatedDeltaNetAcrossSubmissions) {
    gl::DeltaNetShape shape;
    shape.keyHeads = 2;
    shape.valueHeads = 4;
    shape.keyDim = 32;
    shape.valueDim = 48;
    constexpr uint32_t maxTokens = 3;
    const uint32_t C = shape.channels(), Hk = shape.keyHeads, Hv = shape.valueHeads, Dk = shape.keyDim,
                   Dv = shape.valueDim;
    auto conv = tensor(maxTokens * C), z = tensor(maxTokens * Hv * Dv), alpha = tensor(maxTokens * Hv),
         beta = tensor(maxTokens * Hv), a = tensor(Hv), dtBias = tensor(Hv), norm = tensor(Dv),
         state = tensor(Hv * Dk * Dv), output = tensor(maxTokens * Hv * Dv);
    auto graph = compile(gl::gatedDeltaNet(*vc, shape), {conv, z, alpha, beta, a, dtBias, norm, params, state, output});
    const auto aValues = random(Hv, 2.0f), dt = random(Hv), normWeight = random(Dv, 1.5f);
    std::vector<float> negativeA(Hv);
    for (uint32_t h = 0; h < Hv; ++h)
        negativeA[h] = -std::exp(aValues[h]);
    a->setData(0, negativeA);
    dtBias->setData(0, dt);
    norm->setData(0, normWeight);
    std::vector<float> cpuState = random(Hv * Dk * Dv, 0.2f);
    state->setData(0, cpuState);

    for (uint32_t tokens : {3u, 2u}) {
        const auto c = random(maxTokens * C), zv = random(maxTokens * Hv * Dv, 3.0f), al = random(maxTokens * Hv, 3.0f),
                   be = random(maxTokens * Hv, 3.0f);
        conv->setData(0, c);
        z->setData(0, zv);
        alpha->setData(0, al);
        beta->setData(0, be);
        run(*graph, tokens, 0);

        std::vector<float> expected(maxTokens * Hv * Dv, 0.0f);
        for (uint32_t t = 0; t < tokens; ++t) {
            for (uint32_t h = 0; h < Hv; ++h) {
                const uint32_t kh = h % Hk;
                std::vector<double> q(Dk), k(Dk);
                double qq = 0.0, kk = 0.0;
                for (uint32_t i = 0; i < Dk; ++i) {
                    q[i] = c[t * C + kh * Dk + i];
                    k[i] = c[t * C + (Hk + kh) * Dk + i];
                    qq += q[i] * q[i];
                    kk += k[i] * k[i];
                }
                for (uint32_t i = 0; i < Dk; ++i) {
                    q[i] /= std::sqrt(qq + 1e-6) * std::sqrt(double(Dk));
                    k[i] /= std::sqrt(kk + 1e-6);
                }
                const double x = al[t * Hv + h] + dt[h];
                const double decay = std::exp(std::log1p(std::exp(x)) * negativeA[h]);
                const double b = 1.0 / (1.0 + std::exp(-be[t * Hv + h]));
                float* S = &cpuState[h * Dk * Dv];
                std::vector<double> o(Dv, 0.0);
                for (uint32_t j = 0; j < Dv; ++j) {
                    double recalled = 0.0;
                    for (uint32_t i = 0; i < Dk; ++i) {
                        S[i * Dv + j] = float(S[i * Dv + j] * decay);
                        recalled += S[i * Dv + j] * k[i];
                    }
                    const double delta = b * (c[t * C + 2 * Hk * Dk + h * Dv + j] - recalled);
                    for (uint32_t i = 0; i < Dk; ++i) {
                        S[i * Dv + j] = float(S[i * Dv + j] + k[i] * delta);
                        o[j] += S[i * Dv + j] * q[i];
                    }
                }
                double squares = 0.0;
                for (auto value : o)
                    squares += value * value;
                const double inverse = 1.0 / std::sqrt(squares / Dv + shape.epsilon);
                for (uint32_t j = 0; j < Dv; ++j) {
                    const float zj = zv[t * Hv * Dv + h * Dv + j];
                    expected[t * Hv * Dv + h * Dv + j] = float(o[j] * inverse * normWeight[j]) * silu(zj);
                }
            }
        }
        SCOPED_TRACE(tokens);
        expectClose(read(output), expected, 1e-4f, tokens * Hv * Dv);
        expectClose(read(state), cpuState, 1e-4f);
    }
}

TEST_F(GgufLayersTest, MatVecThroughput) {
    if (!std::getenv("KLARTRAUM_BENCHMARK"))
        GTEST_SKIP() << "Set KLARTRAUM_BENCHMARK=1 to run";
    constexpr uint32_t repetitions = 20;
    {
        // Submission overhead: a matVec with almost no work.
        const auto bytes = randomWeights(GgmlType::F32, 8, 256);
        auto weights = weightTensor(GgmlType::F32, bytes, 8, 256);
        auto input = tensor(gl::kMaxTokens * 256);
        auto output = tensor(gl::kMaxTokens * 8);
        auto graph = compile(gl::matVec(*vc, GgmlType::F32, 8, 256, 256, 8), {weights, input, params, output});
        uploadWeights();
        run(*graph, 1, 0);
        const auto start = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < repetitions; ++i)
            run(*graph, 1, 0);
        std::cout << "submission overhead: "
                  << std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() / repetitions * 1e3
                  << " ms" << std::endl;
    }
    // Back-to-back dispatches as in a forward pass: 8 matrices, each used 4 times, per submission.
    constexpr uint32_t rows = 17408, columns = 5120, matrices = 8, uses = 4;
    for (GgmlType type :
         {GgmlType::Q3_K, GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q6_K, GgmlType::Q8_0, GgmlType::F16}) {
        const auto gguf = randomWeights(type, rows, columns);
        std::vector<uint8_t> bytes(gl::gpuRowBytes(type, columns) * rows);
        gl::packWeights(type, gguf.data(), rows, columns, bytes.data());
        auto input = tensor(gl::kMaxTokens * columns);
        auto output = tensor(gl::kMaxTokens * rows);
        std::vector<std::shared_ptr<UintTensor>> weights;
        for (uint32_t m = 0; m < matrices; ++m) {
            weights.push_back(vc->create<UintTensor>(
                std::vector<uint32_t>{uint32_t((bytes.size() + gl::kWeightPaddingBytes + 3) / 4)},
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
        }
        ComputeGraphElementPtr previous;
        for (uint32_t i = 0; i < matrices * uses; ++i) {
            auto operation = gl::matVec(*vc, type, rows, columns, columns, rows);
            connect(operation, {weights[i % matrices], input, params, output});
            if (previous)
                operation->addDependency(previous);
            previous = operation;
        }
        ComputeGraph graph(*vc, 1);
        graph.compileFrom(previous);
        for (auto& weight : weights) {
            weight->getDataBuffer(0).memcopyFrom(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        }
        for (uint32_t tokens : {1u, 16u}) {
            run(graph, tokens, 0);
            const auto start = std::chrono::steady_clock::now();
            for (uint32_t i = 0; i < repetitions / 4; ++i)
                run(graph, tokens, 0);
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() /
                                   (repetitions / 4) / (matrices * uses);
            std::cout << ggmlTypeName(type) << " " << tokens << " token(s): " << seconds * 1e3 << " ms, "
                      << bytes.size() / seconds / 1e9 << " GB/s of weights" << std::endl;
        }
    }
}
