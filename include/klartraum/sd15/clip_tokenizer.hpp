// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_SD15_CLIP_TOKENIZER_HPP
#define KLARTRAUM_SD15_CLIP_TOKENIZER_HPP

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace klartraum {

/**
 * @brief Tokenizes Stable Diffusion 1.5 prompts for its CLIP text encoder.
 *
 * The tokenizer implements the byte-level byte-pair encoding used by the
 * OpenAI CLIP model bundled with Stable Diffusion 1.5. Every encoded prompt is
 * truncated or padded to the fixed 77-token sequence expected by the ONNX text
 * encoder.
 */
class ClipTokenizer {
public:
    static constexpr size_t SequenceLength = 77; ///< CLIP's fixed token sequence length.

    /**
     * @brief Loads a CLIP vocabulary and merge table.
     * @param directory Directory containing `vocab.json` and `merges.txt`.
     * @throws std::runtime_error If either file is missing or malformed, or if
     *         the vocabulary lacks CLIP's start/end tokens.
     */
    explicit ClipTokenizer(const std::filesystem::path& directory);

    /**
     * @brief Encodes one prompt as a padded CLIP token sequence.
     * @param text UTF-8 prompt text.
     * @return Exactly SequenceLength token IDs, including boundary tokens.
     */
    std::vector<int64_t> encode(const std::string& text) const;

    /**
     * @brief Encodes classifier-free guidance prompts as a batch of two sequences.
     * @param negativePrompt Unconditional or negative prompt placed first.
     * @param prompt Positive prompt placed second.
     * @return `2 * SequenceLength` token IDs in negative/positive order.
     */
    std::vector<int64_t> encodePair(const std::string& negativePrompt, const std::string& prompt) const;

    /**
     * @brief Creates an attention mask for one or more complete token sequences.
     * @param tokenIds Concatenated sequences returned by encode() or encodePair().
     * @return A mask that includes the first end token and excludes its padding.
     * @throws std::runtime_error If the input does not contain complete sequences.
     */
    std::vector<int64_t> attentionMask(const std::vector<int64_t>& tokenIds) const;

private:
    std::vector<std::string> pretokenize(const std::string& text) const;
    std::vector<std::string> bytePairEncode(const std::string& token) const;

    std::unordered_map<std::string, int64_t> vocabulary;
    std::map<std::pair<std::string, std::string>, size_t> mergeRanks;
    std::vector<std::string> byteEncoder;
    int64_t beginningOfText = 0;
    int64_t endOfText = 0;
};

} // namespace klartraum

#endif // KLARTRAUM_SD15_CLIP_TOKENIZER_HPP
