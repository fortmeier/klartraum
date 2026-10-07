// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

/**
 * TESTS:
 * - WordsPunctuationAndContractions: The pre-tokenizer splits words with their leading
 *   space, punctuation, and contractions
 * - DigitsAreSingleTokens: Numbers are split into single digits
 * - WhitespaceRuns: Whitespace runs keep their last space for the following word, and
 *   end after their last line break
 * - UnicodeClasses: Non-ASCII letters, combining marks and CJK text stay in word pieces;
 *   emoji are symbol pieces
 * - InvalidUtf8IsKept: Invalid UTF-8 bytes are kept, so every piece list concatenates
 *   back to its input
 * - MatchesHuggingFaceIds: With the Qwen3.6 vocabulary (downloaded GGUF), encoding
 *   matches the Hugging Face tokenizer ids for the cases in
 *   tests/data/gguf/qwen35_tokenizer_cases.gguf, and decoding restores the text
 * - SpecialTokens: Special tokens are parsed only on request, and the chat markers map
 *   to their control tokens
 **/

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "klartraum/gguf/gguf_file.hpp"
#include "klartraum/gguf/gguf_tokenizer.hpp"

using namespace klartraum;
using Pieces = std::vector<std::string>;

namespace {

const char* kCases = "tests/data/gguf/qwen35_tokenizer_cases.gguf";

// The downloaded Qwen3.6-27B GGUF (any quantization), or $KLARTRAUM_QWEN_GGUF.
std::string modelPath() {
    if (const char* path = std::getenv("KLARTRAUM_QWEN_GGUF"))
        return path;
    const std::filesystem::path directory = "data/gguf/qwen3.6-27b";
    if (!std::filesystem::exists(directory))
        return "";
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.path().extension() == ".gguf")
            return entry.path().string();
    }
    return "";
}

} // namespace

TEST(GgufTokenizerTest, WordsPunctuationAndContractions) {
    EXPECT_EQ(GgufTokenizer::pretokenize("Hello, world!"), (Pieces{"Hello", ",", " world", "!"}));
    EXPECT_EQ(GgufTokenizer::pretokenize("I'm sure they'LL go"), (Pieces{"I", "'m", " sure", " they", "'LL", " go"}));
    EXPECT_EQ(GgufTokenizer::pretokenize("x !!!\ny"), (Pieces{"x", " !!!\n", "y"}));
    EXPECT_EQ(GgufTokenizer::pretokenize("'quoted'"), (Pieces{"'quoted", "'"}));
}

TEST(GgufTokenizerTest, DigitsAreSingleTokens) {
    EXPECT_EQ(GgufTokenizer::pretokenize("in 2026"), (Pieces{"in", " ", "2", "0", "2", "6"}));
}

TEST(GgufTokenizerTest, WhitespaceRuns) {
    EXPECT_EQ(GgufTokenizer::pretokenize("a  b"), (Pieces{"a", " ", " b"}));
    EXPECT_EQ(GgufTokenizer::pretokenize("x\n\n  y"), (Pieces{"x", "\n\n", " ", " y"}));
    EXPECT_EQ(GgufTokenizer::pretokenize("end   "), (Pieces{"end", "   "}));
    EXPECT_EQ(GgufTokenizer::pretokenize("a\tb"), (Pieces{"a", "\tb"}));
    EXPECT_EQ(GgufTokenizer::pretokenize("a\n b"), (Pieces{"a", "\n", " b"}));
}

TEST(GgufTokenizerTest, UnicodeClasses) {
    EXPECT_EQ(GgufTokenizer::pretokenize("Grüße naïve"), (Pieces{"Grüße", " naïve"}));
    EXPECT_EQ(GgufTokenizer::pretokenize("हिन्दी"), (Pieces{"हिन्दी"})); // vowel signs are marks
    EXPECT_EQ(GgufTokenizer::pretokenize("日本語。"), (Pieces{"日本語", "。"}));
    EXPECT_EQ(GgufTokenizer::pretokenize("ok 🙂🚀"), (Pieces{"ok", " 🙂🚀"}));
}

TEST(GgufTokenizerTest, InvalidUtf8IsKept) {
    const std::string text = std::string("ab\xff\xfe") + " c\xc3";
    std::string joined;
    for (const auto& piece : GgufTokenizer::pretokenize(text))
        joined += piece;
    EXPECT_EQ(joined, text);
}

class GgufTokenizerModelTest : public ::testing::Test {
protected:
    void SetUp() override {
        const auto path = modelPath();
        if (path.empty())
            GTEST_SKIP() << "Download the model with scripts/qwen_gguf/download.py";
        file = std::make_unique<GgufFile>(path);
        tokenizer = std::make_unique<GgufTokenizer>(*file);
    }
    std::unique_ptr<GgufFile> file;
    std::unique_ptr<GgufTokenizer> tokenizer;
};

TEST_F(GgufTokenizerModelTest, MatchesHuggingFaceIds) {
    GgufFile cases(kCases);
    const auto& texts = cases.getStringArray("texts");
    const auto& idInfo = cases.getTensor("ids");
    const auto& countInfo = cases.getTensor("counts");
    const auto* ids = reinterpret_cast<const int32_t*>(cases.data(idInfo));
    const auto* counts = reinterpret_cast<const int32_t*>(cases.data(countInfo));
    ASSERT_EQ(texts.size(), countInfo.elementCount());
    size_t offset = 0;
    for (size_t i = 0; i < texts.size(); ++i) {
        SCOPED_TRACE(texts[i]);
        const std::vector<int32_t> expected(ids + offset, ids + offset + counts[i]);
        offset += size_t(counts[i]);
        const auto encoded = tokenizer->encode(texts[i]);
        EXPECT_EQ(encoded, expected);
        EXPECT_EQ(tokenizer->decode(encoded), texts[i]);
    }
}

TEST_F(GgufTokenizerModelTest, SpecialTokens) {
    const int32_t start = tokenizer->tokenId("<|im_start|>");
    const int32_t end = tokenizer->tokenId("<|im_end|>");
    EXPECT_TRUE(tokenizer->isControl(start));
    EXPECT_EQ(tokenizer->endOfSequence(), end);
    const auto parsed = tokenizer->encode("<|im_start|>user\nHi<|im_end|>", true);
    ASSERT_GE(parsed.size(), 4u);
    EXPECT_EQ(parsed.front(), start);
    EXPECT_EQ(parsed.back(), end);
    EXPECT_EQ(tokenizer->decode(parsed), "<|im_start|>user\nHi<|im_end|>");
    const auto plain = tokenizer->encode("<|im_start|>");
    EXPECT_GT(plain.size(), 1u);
    EXPECT_NE(plain.front(), start);
}
