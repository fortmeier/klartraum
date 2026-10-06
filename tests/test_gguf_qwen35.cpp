// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - The qwen35 configuration is read from GGUF metadata; other architectures are rejected.
 * - Nine tokens in one submission reproduce the transformers logits at every position (F32 weights).
 * - Prompt chunks of 4 + 4 + 1 tokens carry caches and states across submissions and match as well.
 * - Token-by-token decoding with last-token logits matches every position.
 * - resetState() starts a new sequence: running the prompt again gives the same logits.
 * - With Q8_0 projection weights the logits stay close to the reference and agree on the top token.
 **/

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/gguf/gguf_network.hpp"
#include "klartraum/headless_frontend.hpp"

using namespace klartraum;

namespace {

const char* kModelF32 = "tests/data/gguf/tiny_qwen35_f32.gguf";
const char* kModelQ8 = "tests/data/gguf/tiny_qwen35_q8_0.gguf";
const char* kReference = "tests/data/gguf/tiny_qwen35_reference.gguf";
const char* kQuantBlocks = "tests/data/gguf/quant_blocks.gguf";

class GgufQwen35Test : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(kModelF32) || !std::filesystem::exists(kReference)) {
            GTEST_SKIP() << "Run scripts/qwen_gguf/make_tiny_model.py";
        }
        frontend = std::make_unique<HeadlessFrontend>();
        vc = &frontend->getKlartraumEngine().getVulkanContext();
        GgufFile reference(kReference);
        const auto& tokenInfo = reference.getTensor("tokens");
        const auto* tokenData = reinterpret_cast<const int32_t*>(reference.data(tokenInfo));
        tokens.assign(tokenData, tokenData + tokenInfo.elementCount());
        const auto& logitInfo = reference.getTensor("logits");
        const auto* logitData = reinterpret_cast<const float*>(reference.data(logitInfo));
        expected.assign(logitData, logitData + logitInfo.elementCount());
        vocabulary = uint32_t(logitInfo.rowLength());
    }

    struct Runner {
        std::shared_ptr<GgufNetwork> network;
        std::unique_ptr<ComputeGraph> graph;
    };

    Runner make(const char* model, GgufNetworkOptions options) {
        Runner runner;
        runner.network = vc->create<GgufNetwork>(model, options);
        runner.graph = std::make_unique<ComputeGraph>(*vc, 1);
        runner.graph->compileFrom(runner.network);
        return runner;
    }

    // Runs the reference tokens in chunks of `chunk`; returns the logits of every position.
    std::vector<float> runAll(Runner& runner, uint32_t chunk) {
        std::vector<float> all;
        for (uint32_t start = 0; start < tokens.size(); start += chunk) {
            const uint32_t end = std::min<uint32_t>(uint32_t(tokens.size()), start + chunk);
            runner.network->setTokens({tokens.begin() + start, tokens.begin() + end}, start);
            runner.graph->submitAndWait(vc->getGraphicsQueue(), 0);
            const auto logits = runner.network->readLogits();
            all.insert(all.end(), logits.begin(), logits.end());
        }
        return all;
    }

    void expectMatches(const std::vector<float>& actual, float tolerance) {
        ASSERT_EQ(actual.size(), expected.size());
        float scale = 0.0f, error = 0.0f;
        for (size_t i = 0; i < expected.size(); ++i) {
            scale = std::max(scale, std::fabs(expected[i]));
            error = std::max(error, std::fabs(actual[i] - expected[i]));
        }
        EXPECT_LT(error, tolerance * scale) << "max error " << error << " at logit scale " << scale;
    }

    uint32_t argmax(const std::vector<float>& logits, size_t position) const {
        const auto begin = logits.begin() + position * vocabulary;
        return uint32_t(std::max_element(begin, begin + vocabulary) - begin);
    }

    std::unique_ptr<HeadlessFrontend> frontend;
    VulkanContext* vc = nullptr;
    std::vector<int32_t> tokens;
    std::vector<float> expected;
    uint32_t vocabulary = 0;
};

} // namespace

TEST_F(GgufQwen35Test, ReadsConfiguration) {
    GgufFile file(kModelF32);
    const auto config = Qwen35Config::fromFile(file);
    EXPECT_EQ(config.vocabulary, 300u);
    EXPECT_EQ(config.hidden, 64u);
    EXPECT_EQ(config.layers, 4u);
    EXPECT_EQ(config.heads, 4u);
    EXPECT_EQ(config.kvHeads, 2u);
    EXPECT_EQ(config.headDim, 32u);
    EXPECT_EQ(config.ropeDims, 8u);
    EXPECT_EQ(config.keyHeads, 2u);
    EXPECT_EQ(config.valueHeads, 4u);
    EXPECT_EQ(config.keyDim, 16u);
    EXPECT_EQ(config.valueDim, 16u);
    EXPECT_EQ(config.convChannels(), 128u);
    EXPECT_EQ(config.recurrent, (std::vector<bool>{true, true, true, false}));
    if (std::filesystem::exists(kQuantBlocks)) {
        GgufFile other(kQuantBlocks);
        EXPECT_THROW(Qwen35Config::fromFile(other), std::runtime_error);
    }
}

TEST_F(GgufQwen35Test, SingleSubmissionMatchesTransformers) {
    auto runner = make(kModelF32, {16, 64, true});
    expectMatches(runAll(runner, 16), 1e-4f);
}

TEST_F(GgufQwen35Test, ChunkedPromptMatchesTransformers) {
    auto runner = make(kModelF32, {4, 64, true});
    expectMatches(runAll(runner, 4), 1e-4f);
}

TEST_F(GgufQwen35Test, TokenByTokenMatchesTransformers) {
    auto runner = make(kModelF32, {1, 64, false});
    expectMatches(runAll(runner, 1), 1e-4f);
}

TEST_F(GgufQwen35Test, ResetStateRestartsTheSequence) {
    auto runner = make(kModelF32, {4, 64, true});
    const auto first = runAll(runner, 4);
    runner.network->resetState();
    const auto second = runAll(runner, 4);
    EXPECT_EQ(first, second);
}

TEST_F(GgufQwen35Test, Q8WeightsStayClose) {
    auto runner = make(kModelQ8, {16, 64, true});
    const auto logits = runAll(runner, 16);
    expectMatches(logits, 0.05f);
    for (size_t position = 0; position < tokens.size(); ++position) {
        EXPECT_EQ(argmax(logits, position), argmax(expected, position)) << "position " << position;
    }
}
