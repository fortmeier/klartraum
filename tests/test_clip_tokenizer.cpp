// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - The native tokenizer reproduces Hugging Face CLIP token IDs for common and Unicode prompts.
 * - Classifier-free guidance batches the negative prompt before the positive prompt.
 * - Attention masks include the first end token and exclude padding end tokens.
 **/

#include <filesystem>
#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

#include "klartraum/sd15/clip_tokenizer.hpp"

using namespace klartraum;

namespace {

std::filesystem::path tokenizerDirectory() { return "./data/onnx/sd15_denoiser_256"; }

} // namespace

TEST(ClipTokenizerTest, MatchesHuggingFaceTokens) {
    if (!std::filesystem::exists(tokenizerDirectory() / "vocab.json")) {
        GTEST_SKIP() << "Generate tokenizer assets with scripts/sd15_onnx/export_denoiser.py";
    }
    ClipTokenizer tokenizer(tokenizerDirectory());
    const auto cat = tokenizer.encode("a photo of a cat");
    EXPECT_EQ(std::vector<int64_t>(cat.begin(), cat.begin() + 7),
              (std::vector<int64_t>{49406, 320, 1125, 539, 320, 2368, 49407}));
    const auto punctuation = tokenizer.encode("Hello, world!");
    EXPECT_EQ(std::vector<int64_t>(punctuation.begin(), punctuation.begin() + 6),
              (std::vector<int64_t>{49406, 3306, 267, 1002, 256, 49407}));
    const auto contraction = tokenizer.encode("don't stop");
    EXPECT_EQ(std::vector<int64_t>(contraction.begin(), contraction.begin() + 5),
              (std::vector<int64_t>{49406, 847, 713, 1691, 49407}));
    const auto japanese = tokenizer.encode("\xE6\x97\xA5\xE6\x9C\xAC\xE3\x81\xAE\xE7\x9F\xB3\xE7\x81\xAF\xE7\xB1\xA0");
    EXPECT_EQ(std::vector<int64_t>(japanese.begin(), japanese.begin() + 16),
              (std::vector<int64_t>{49406, 37156, 19277, 361, 4813, 362, 163, 253, 367, 163, 223, 363, 163, 109, 510,
                                    49407}));
    const auto emoji = tokenizer.encode("\xF0\x9F\x98\x80 red car");
    EXPECT_EQ(std::vector<int64_t>(emoji.begin(), emoji.begin() + 5),
              (std::vector<int64_t>{49406, 7334, 736, 1615, 49407}));
    const auto numeric = tokenizer.encode("35mm photography");
    EXPECT_EQ(std::vector<int64_t>(numeric.begin(), numeric.begin() + 6),
              (std::vector<int64_t>{49406, 274, 276, 2848, 2108, 49407}));
}

TEST(ClipTokenizerTest, EncodesNegativePromptFirst) {
    if (!std::filesystem::exists(tokenizerDirectory() / "vocab.json")) {
        GTEST_SKIP() << "Generate tokenizer assets with scripts/sd15_onnx/export_denoiser.py";
    }
    ClipTokenizer tokenizer(tokenizerDirectory());
    const auto batch = tokenizer.encodePair("", "Japanese stone lantern");
    ASSERT_EQ(batch.size(), 2 * ClipTokenizer::SequenceLength);
    EXPECT_EQ(batch[0], 49406);
    EXPECT_EQ(batch[1], 49407);
    EXPECT_EQ(batch[77], 49406);
    EXPECT_EQ(std::vector<int64_t>(batch.begin() + 78, batch.begin() + 82),
              (std::vector<int64_t>{4925, 2441, 17185, 49407}));
}

TEST(ClipTokenizerTest, CreatesAttentionMaskThroughFirstEndToken) {
    if (!std::filesystem::exists(tokenizerDirectory() / "vocab.json")) {
        GTEST_SKIP() << "Generate tokenizer assets with scripts/sd15_onnx/export_denoiser.py";
    }
    ClipTokenizer tokenizer(tokenizerDirectory());
    const auto tokens = tokenizer.encodePair("", "Japanese stone lantern");
    const auto mask = tokenizer.attentionMask(tokens);
    ASSERT_EQ(mask.size(), tokens.size());
    EXPECT_EQ(std::count(mask.begin(), mask.begin() + 77, int64_t{1}), 2);
    EXPECT_EQ(std::count(mask.begin() + 77, mask.end(), int64_t{1}), 5);
    EXPECT_TRUE(std::all_of(mask.begin(), mask.end(), [](int64_t value) { return value == 0 || value == 1; }));
}
