// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/gguf/gguf_tokenizer.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace klartraum {

namespace {

enum class UnicodeClass : uint8_t { Other, Letter, Mark, Number };

struct UnicodeRange {
    uint32_t first;
    uint32_t last;
    UnicodeClass type;
};

constexpr UnicodeRange kUnicodeRanges[] = {
#include "unicode_categories.inc"
};

UnicodeClass classify(uint32_t code) {
    const auto* end = std::end(kUnicodeRanges);
    const auto* range = std::upper_bound(std::begin(kUnicodeRanges), end, code,
                                         [](uint32_t value, const UnicodeRange& r) { return value < r.first; });
    if (range == std::begin(kUnicodeRanges))
        return UnicodeClass::Other;
    --range;
    return code <= range->last ? range->type : UnicodeClass::Other;
}

// The Unicode White_Space property.
bool isSpace(uint32_t c) {
    return (c >= 0x09 && c <= 0x0d) || c == 0x20 || c == 0x85 || c == 0xa0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200a) || c == 0x2028 || c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000;
}

bool isNewline(uint32_t c) { return c == '\r' || c == '\n'; }

// Code points of UTF-8 text with the byte offset of each (offsets has one
// extra entry, the text size). Invalid bytes become single code points
// outside every class, so their bytes are kept.
void decodeUtf8(const std::string& text, std::vector<uint32_t>& codes, std::vector<size_t>& offsets) {
    const auto* bytes = reinterpret_cast<const unsigned char*>(text.data());
    size_t i = 0;
    while (i < text.size()) {
        offsets.push_back(i);
        const unsigned char lead = bytes[i];
        uint32_t length = lead < 0x80           ? 1
                          : (lead >> 5) == 0x6  ? 2
                          : (lead >> 4) == 0xe  ? 3
                          : (lead >> 3) == 0x1e ? 4
                                                : 0;
        bool valid = length != 0 && i + length <= text.size();
        for (uint32_t k = 1; valid && k < length; ++k)
            valid = (bytes[i + k] & 0xc0) == 0x80;
        if (!valid) {
            codes.push_back(0x110000u + lead); // not a code point: class Other, not a space
            ++i;
            continue;
        }
        uint32_t code = length == 1 ? lead : lead & (0x7f >> length);
        for (uint32_t k = 1; k < length; ++k)
            code = (code << 6) | (bytes[i + k] & 0x3f);
        codes.push_back(code);
        i += length;
    }
    offsets.push_back(text.size());
}

void appendUtf8(std::string& out, uint32_t code) {
    if (code < 0x80) {
        out += char(code);
    } else if (code < 0x800) {
        out += char(0xc0 | (code >> 6));
        out += char(0x80 | (code & 0x3f));
    } else if (code < 0x10000) {
        out += char(0xe0 | (code >> 12));
        out += char(0x80 | ((code >> 6) & 0x3f));
        out += char(0x80 | (code & 0x3f));
    } else {
        out += char(0xf0 | (code >> 18));
        out += char(0x80 | ((code >> 12) & 0x3f));
        out += char(0x80 | ((code >> 6) & 0x3f));
        out += char(0x80 | (code & 0x3f));
    }
}

// Pre-tokenizer of Qwen3.5: at each position the first matching rule of
//   1. an apostrophe contraction: 's 't 're 've 'm 'll 'd (any case)
//   2. letters/marks, optionally preceded by one character that is not a
//      line break, letter or number (e.g. a space)
//   3. a single number character
//   4. an optional space, then characters that are neither whitespace nor
//      letters/marks/numbers, then any line breaks
//   5. whitespace up to and including its last line break
//   6. whitespace not followed by a non-space (all but the last character
//      of a run that precedes text)
//   7. whitespace
// ends the next piece.
class Pretokenizer {
public:
    explicit Pretokenizer(const std::vector<uint32_t>& codes)
        : c(codes),
          n(codes.size()) {}

    size_t match(size_t i) const {
        if (size_t end = contraction(i))
            return end;
        if (size_t end = word(i))
            return end;
        if (number(i))
            return i + 1;
        if (size_t end = punctuation(i))
            return end;
        const size_t run = spaceRun(i);
        if (run > i) {
            for (size_t k = run; k > i; --k) {
                if (isNewline(c[k - 1]))
                    return k; // rule 5
            }
            if (run == n)
                return run; // rule 6 at the end of the text
            if (run - i >= 2)
                return run - 1; // rule 6
            return run;         // rule 7
        }
        return i + 1; // unreachable: every character matches one of the rules
    }

private:
    const std::vector<uint32_t>& c;
    size_t n;

    bool letterOrMark(size_t i) const {
        const auto type = classify(c[i]);
        return type == UnicodeClass::Letter || type == UnicodeClass::Mark;
    }
    bool number(size_t i) const { return classify(c[i]) == UnicodeClass::Number; }
    bool symbol(size_t i) const { return !isSpace(c[i]) && classify(c[i]) == UnicodeClass::Other; }

    static uint32_t lower(uint32_t code) { return code >= 'A' && code <= 'Z' ? code + 32 : code; }

    size_t contraction(size_t i) const {
        if (c[i] != '\'')
            return 0;
        static const char* const suffixes[] = {"s", "t", "re", "ve", "m", "ll", "d"};
        for (const char* suffix : suffixes) {
            size_t k = 0;
            while (suffix[k] && i + 1 + k < n && lower(c[i + 1 + k]) == uint32_t(suffix[k]))
                ++k;
            if (!suffix[k])
                return i + 1 + k;
        }
        return 0;
    }

    size_t word(size_t i) const {
        size_t start = i;
        const auto type = classify(c[i]);
        const bool prefix = !isNewline(c[i]) && type != UnicodeClass::Letter && type != UnicodeClass::Number;
        if (prefix && i + 1 < n && letterOrMark(i + 1))
            start = i + 1;
        else if (!letterOrMark(i))
            return 0;
        size_t end = start;
        while (end < n && letterOrMark(end))
            ++end;
        return end;
    }

    size_t punctuation(size_t i) const {
        size_t k = i;
        if (c[k] == ' ' && k + 1 < n && symbol(k + 1))
            ++k;
        if (!symbol(k))
            return 0;
        while (k < n && symbol(k))
            ++k;
        while (k < n && isNewline(c[k]))
            ++k;
        return k;
    }

    size_t spaceRun(size_t i) const {
        size_t k = i;
        while (k < n && isSpace(c[k]))
            ++k;
        return k;
    }
};

} // namespace

std::vector<std::string> GgufTokenizer::pretokenize(const std::string& text) {
    std::vector<uint32_t> codes;
    std::vector<size_t> offsets;
    decodeUtf8(text, codes, offsets);
    Pretokenizer pretokenizer(codes);
    std::vector<std::string> pieces;
    for (size_t i = 0; i < codes.size();) {
        const size_t end = pretokenizer.match(i);
        pieces.push_back(text.substr(offsets[i], offsets[end] - offsets[i]));
        i = end;
    }
    return pieces;
}

GgufTokenizer::GgufTokenizer(const GgufFile& file) {
    const std::string model = file.getString("tokenizer.ggml.model", "");
    if (model != "gpt2")
        throw std::runtime_error("Unsupported GGUF tokenizer model: " + model);
    tokens = file.getStringArray("tokenizer.ggml.tokens");
    if (file.has("tokenizer.ggml.token_type")) {
        for (auto type : file.getIntegerArray("tokenizer.ggml.token_type"))
            tokenTypes.push_back(int32_t(type));
    }
    tokenTypes.resize(tokens.size(), 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        tokenIndex.emplace(tokens[i], int32_t(i));
        if (isControl(int32_t(i)))
            specialTokens.push_back(tokens[i]);
    }
    std::sort(specialTokens.begin(), specialTokens.end(),
              [](const std::string& a, const std::string& b) { return a.size() > b.size(); });
    const auto& merges = file.getStringArray("tokenizer.ggml.merges");
    mergeRanks.reserve(merges.size());
    for (size_t i = 0; i < merges.size(); ++i)
        mergeRanks.emplace(merges[i], uint32_t(i));
    eosToken = int32_t(file.getInteger("tokenizer.ggml.eos_token_id", -1));

    // Byte-level alphabet: printable Latin-1 bytes stand for themselves; the
    // others (controls, space, DEL, NBSP, soft hyphen) take the code points
    // from 256 on, in byte order.
    uint32_t next = 256;
    for (uint32_t byte = 0; byte < 256; ++byte) {
        const bool printable = (byte >= 0x21 && byte <= 0x7e) || (byte >= 0xa1 && byte <= 0xac) || byte >= 0xae;
        std::string symbol;
        appendUtf8(symbol, printable ? byte : next++);
        byteSymbols[byte] = symbol;
        symbolBytes[symbol] = uint8_t(byte);
    }
}

bool GgufTokenizer::isControl(int32_t token) const {
    // GGUF token types: 1 normal, 2 unknown, 3 control, 4 user defined, 5 unused, 6 byte.
    return token >= 0 && size_t(token) < tokenTypes.size() && (tokenTypes[token] == 3 || tokenTypes[token] == 4);
}

int32_t GgufTokenizer::tokenId(const std::string& text) const {
    const auto it = tokenIndex.find(text);
    if (it == tokenIndex.end())
        throw std::runtime_error("Token not in vocabulary: " + text);
    return it->second;
}

void GgufTokenizer::encodePiece(const std::string& piece, std::vector<int32_t>& output) const {
    std::vector<std::string> symbols;
    for (unsigned char byte : piece)
        symbols.push_back(byteSymbols[byte]);
    // Merge the adjacent pair with the lowest rank until no ranked pair is left.
    while (symbols.size() > 1) {
        uint32_t best = UINT32_MAX;
        size_t bestIndex = 0;
        for (size_t i = 0; i + 1 < symbols.size(); ++i) {
            const auto rank = mergeRanks.find(symbols[i] + " " + symbols[i + 1]);
            if (rank != mergeRanks.end() && rank->second < best) {
                best = rank->second;
                bestIndex = i;
            }
        }
        if (best == UINT32_MAX)
            break;
        symbols[bestIndex] += symbols[bestIndex + 1];
        symbols.erase(symbols.begin() + long(bestIndex) + 1);
    }
    for (const auto& symbol : symbols) {
        const auto it = tokenIndex.find(symbol);
        if (it == tokenIndex.end())
            throw std::runtime_error("Byte-level symbol missing from the vocabulary");
        output.push_back(it->second);
    }
}

void GgufTokenizer::encodeText(const std::string& text, std::vector<int32_t>& output) const {
    for (const auto& piece : pretokenize(text))
        encodePiece(piece, output);
}

std::vector<int32_t> GgufTokenizer::encode(const std::string& text, bool parseSpecial) const {
    std::vector<int32_t> output;
    if (!parseSpecial) {
        encodeText(text, output);
        return output;
    }
    size_t start = 0, i = 0;
    while (i < text.size()) {
        const std::string* found = nullptr;
        if (text[i] == '<') {
            for (const auto& special : specialTokens) {
                if (text.compare(i, special.size(), special) == 0) {
                    found = &special;
                    break;
                }
            }
        }
        if (!found) {
            ++i;
            continue;
        }
        if (i > start)
            encodeText(text.substr(start, i - start), output);
        output.push_back(tokenIndex.at(*found));
        i += found->size();
        start = i;
    }
    if (start < text.size())
        encodeText(text.substr(start), output);
    return output;
}

std::string GgufTokenizer::tokenBytes(int32_t token) const {
    if (token < 0 || size_t(token) >= tokens.size())
        throw std::runtime_error("Token id out of range");
    const std::string& text = tokens[token];
    if (isControl(token))
        return text;
    std::vector<uint32_t> codes;
    std::vector<size_t> offsets;
    decodeUtf8(text, codes, offsets);
    std::string bytes;
    for (size_t i = 0; i < codes.size(); ++i) {
        const auto it = symbolBytes.find(text.substr(offsets[i], offsets[i + 1] - offsets[i]));
        if (it != symbolBytes.end())
            bytes += char(it->second);
    }
    return bytes;
}

std::string GgufTokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string text;
    for (auto id : ids)
        text += tokenBytes(id);
    return text;
}

} // namespace klartraum
