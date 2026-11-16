// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_GGUF_GGUF_TOKENIZER_HPP
#define KLARTRAUM_GGUF_GGUF_TOKENIZER_HPP

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "klartraum/gguf/gguf_file.hpp"

namespace klartraum {

/**
 * @brief Byte-level BPE tokenizer read from GGUF metadata (`tokenizer.ggml.model` "gpt2").
 *
 * Text is split into pieces by the Qwen3.5 pre-tokenizer rules
 * (`tokenizer.ggml.pre` "qwen35"): contractions, words with an optional
 * leading non-letter, single digits, punctuation runs, and whitespace runs.
 * Each piece's UTF-8 bytes are mapped to the byte-level alphabet and merged
 * by the ranked merges of `tokenizer.ggml.merges`.
 *
 * Input is expected in Unicode normalization form C (as typed text usually
 * is); it is not normalized here.
 */
class GgufTokenizer {
public:
    /** @throws std::runtime_error If the file has no supported tokenizer. */
    explicit GgufTokenizer(const GgufFile& file);

    /**
     * @brief Token ids of @p text.
     * @param parseSpecial Also recognize control tokens such as `<|im_start|>` in the text.
     */
    std::vector<int32_t> encode(const std::string& text, bool parseSpecial = false) const;

    /** @brief The text (UTF-8 bytes) of @p tokens; control tokens are written as their names. */
    std::string decode(const std::vector<int32_t>& tokens) const;

    /** @brief The bytes of one token (control tokens: their name). */
    std::string tokenBytes(int32_t token) const;

    /** @brief Id of the token spelled @p text (e.g. "<|im_end|>"); throws if it does not exist. */
    int32_t tokenId(const std::string& text) const;

    uint32_t vocabularySize() const { return uint32_t(tokens.size()); }
    int32_t endOfSequence() const { return eosToken; }
    bool isControl(int32_t token) const;

    /** @brief Splits @p text into pre-tokenizer pieces (exposed for tests). */
    static std::vector<std::string> pretokenize(const std::string& text);

private:
    std::vector<std::string> tokens;
    std::vector<int32_t> tokenTypes;
    std::unordered_map<std::string, int32_t> tokenIndex;
    std::unordered_map<std::string, uint32_t> mergeRanks; // "left right" -> rank
    std::vector<std::string> specialTokens;               // control/user-defined tokens, longest first
    std::string byteSymbols[256];                         // byte -> UTF-8 of its byte-level symbol
    std::unordered_map<std::string, uint8_t> symbolBytes;
    int32_t eosToken = -1;

    void encodePiece(const std::string& piece, std::vector<int32_t>& output) const;
    void encodeText(const std::string& text, std::vector<int32_t>& output) const;
};

} // namespace klartraum

#endif // KLARTRAUM_GGUF_GGUF_TOKENIZER_HPP
