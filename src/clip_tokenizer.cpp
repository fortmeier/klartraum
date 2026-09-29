// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/sd15/clip_tokenizer.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace klartraum {
namespace {

std::string readText(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("Could not open CLIP tokenizer file: " + path.string());
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void appendUtf8(std::string& output, uint32_t codepoint) {
    if (codepoint <= 0x7f) {
        output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7ff) {
        output.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else if (codepoint <= 0xffff) {
        output.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    } else {
        output.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
        output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
    }
}

uint32_t parseHex4(const std::string& source, size_t& position) {
    if (position + 4 > source.size())
        throw std::runtime_error("Truncated Unicode escape in CLIP vocabulary");
    uint32_t value = 0;
    for (size_t index = 0; index < 4; ++index) {
        const char character = source[position++];
        value <<= 4;
        if (character >= '0' && character <= '9')
            value += character - '0';
        else if (character >= 'a' && character <= 'f')
            value += character - 'a' + 10;
        else if (character >= 'A' && character <= 'F')
            value += character - 'A' + 10;
        else
            throw std::runtime_error("Invalid Unicode escape in CLIP vocabulary");
    }
    return value;
}

std::string parseJsonString(const std::string& source, size_t& position) {
    if (position >= source.size() || source[position++] != '"') {
        throw std::runtime_error("Expected string in CLIP vocabulary");
    }
    std::string result;
    while (position < source.size()) {
        const char character = source[position++];
        if (character == '"')
            return result;
        if (character != '\\') {
            result.push_back(character);
            continue;
        }
        if (position >= source.size())
            throw std::runtime_error("Truncated escape in CLIP vocabulary");
        const char escaped = source[position++];
        if (escaped == '"' || escaped == '\\' || escaped == '/')
            result.push_back(escaped);
        else if (escaped == 'b')
            result.push_back('\b');
        else if (escaped == 'f')
            result.push_back('\f');
        else if (escaped == 'n')
            result.push_back('\n');
        else if (escaped == 'r')
            result.push_back('\r');
        else if (escaped == 't')
            result.push_back('\t');
        else if (escaped == 'u') {
            uint32_t codepoint = parseHex4(source, position);
            if (codepoint >= 0xd800 && codepoint <= 0xdbff && position + 6 <= source.size() &&
                source[position] == '\\' && source[position + 1] == 'u') {
                position += 2;
                const uint32_t low = parseHex4(source, position);
                if (low < 0xdc00 || low > 0xdfff)
                    throw std::runtime_error("Invalid surrogate pair");
                codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
            }
            appendUtf8(result, codepoint);
        } else {
            throw std::runtime_error("Invalid escape in CLIP vocabulary");
        }
    }
    throw std::runtime_error("Unterminated string in CLIP vocabulary");
}

std::unordered_map<std::string, int64_t> parseVocabulary(const std::string& source) {
    std::unordered_map<std::string, int64_t> result;
    size_t position = 0;
    const auto skipWhitespace = [&]() {
        while (position < source.size() && std::isspace(static_cast<unsigned char>(source[position])))
            ++position;
    };
    skipWhitespace();
    if (position >= source.size() || source[position++] != '{')
        throw std::runtime_error("Invalid CLIP vocabulary JSON");
    while (true) {
        skipWhitespace();
        if (position < source.size() && source[position] == '}')
            break;
        const std::string token = parseJsonString(source, position);
        skipWhitespace();
        if (position >= source.size() || source[position++] != ':')
            throw std::runtime_error("Invalid CLIP vocabulary entry");
        skipWhitespace();
        size_t end = position;
        while (end < source.size() && std::isdigit(static_cast<unsigned char>(source[end])))
            ++end;
        if (end == position)
            throw std::runtime_error("Invalid token id in CLIP vocabulary");
        const int64_t id = std::stoll(source.substr(position, end - position));
        position = end;
        result.emplace(token, id);
        skipWhitespace();
        if (position < source.size() && source[position] == ',') {
            ++position;
            continue;
        }
        if (position < source.size() && source[position] == '}')
            break;
        throw std::runtime_error("Invalid separator in CLIP vocabulary");
    }
    return result;
}

struct Character {
    uint32_t codepoint;
    std::string bytes;
};

std::vector<Character> decodeUtf8(const std::string& text) {
    std::vector<Character> result;
    for (size_t position = 0; position < text.size();) {
        const size_t start = position;
        const uint8_t first = static_cast<uint8_t>(text[position++]);
        uint32_t codepoint = first;
        size_t continuation = 0;
        if ((first & 0xe0) == 0xc0) {
            codepoint = first & 0x1f;
            continuation = 1;
        } else if ((first & 0xf0) == 0xe0) {
            codepoint = first & 0x0f;
            continuation = 2;
        } else if ((first & 0xf8) == 0xf0) {
            codepoint = first & 0x07;
            continuation = 3;
        } else if (first >= 0x80)
            throw std::runtime_error("Prompt is not valid UTF-8");
        if (position + continuation > text.size())
            throw std::runtime_error("Prompt is not valid UTF-8");
        for (size_t index = 0; index < continuation; ++index) {
            const uint8_t next = static_cast<uint8_t>(text[position++]);
            if ((next & 0xc0) != 0x80)
                throw std::runtime_error("Prompt is not valid UTF-8");
            codepoint = (codepoint << 6) | (next & 0x3f);
        }
        result.push_back({codepoint, text.substr(start, position - start)});
    }
    return result;
}

bool isWhitespace(uint32_t value) {
    return value == 0x20 || (value >= 0x09 && value <= 0x0d) || value == 0x85 || value == 0xa0 || value == 0x1680 ||
           (value >= 0x2000 && value <= 0x200a) || value == 0x2028 || value == 0x2029 || value == 0x202f ||
           value == 0x205f || value == 0x3000;
}

bool isNumber(uint32_t value) {
    if (value >= '0' && value <= '9')
        return true;
    constexpr std::array<uint32_t, 22> starts = {0x0660, 0x06f0, 0x07c0, 0x0966,  0x09e6,  0x0a66, 0x0ae6, 0x0b66,
                                                 0x0be6, 0x0c66, 0x0ce6, 0x0d66,  0x0e50,  0x0ed0, 0x0f20, 0x1040,
                                                 0x17e0, 0x1810, 0xff10, 0x104a0, 0x11066, 0x1d7ce};
    for (uint32_t start : starts)
        if (value >= start && value <= start + 9)
            return true;
    return false;
}

bool isLetter(uint32_t value) {
    if ((value >= 'a' && value <= 'z') || (value >= 'A' && value <= 'Z'))
        return true;
    if (value < 0x80 || isNumber(value) || isWhitespace(value))
        return false;
    if ((value >= 0x2000 && value <= 0x206f) || (value >= 0x2190 && value <= 0x2bff) ||
        (value >= 0x3000 && value <= 0x303f) || (value >= 0xfe10 && value <= 0xfe6f) ||
        (value >= 0x1f000 && value <= 0x1faff))
        return false;
    return true;
}

bool isCjkIdeograph(uint32_t value) {
    return (value >= 0x3400 && value <= 0x4dbf) || (value >= 0x4e00 && value <= 0x9fff) ||
           (value >= 0xf900 && value <= 0xfaff) || (value >= 0x20000 && value <= 0x2a6df) ||
           (value >= 0x2a700 && value <= 0x2b73f) || (value >= 0x2b740 && value <= 0x2b81f) ||
           (value >= 0x2b820 && value <= 0x2ceaf) || (value >= 0x2f800 && value <= 0x2fa1f);
}

std::string cleanText(const std::string& text) {
    std::string lowered = text;
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char value) {
        return value < 0x80 ? static_cast<char>(std::tolower(value)) : static_cast<char>(value);
    });
    const auto characters = decodeUtf8(lowered);
    std::string result;
    bool pendingSpace = false;
    for (const auto& character : characters) {
        if (character.codepoint == 0 || character.codepoint == 0xfffd ||
            (character.codepoint < 0x20 && !isWhitespace(character.codepoint)))
            continue;
        if (isWhitespace(character.codepoint)) {
            pendingSpace = !result.empty();
        } else {
            if (pendingSpace)
                result.push_back(' ');
            pendingSpace = false;
            result += character.bytes;
        }
    }
    return result;
}

std::vector<std::string> makeByteEncoder() {
    std::vector<uint32_t> values;
    for (uint32_t value = '!'; value <= '~'; ++value)
        values.push_back(value);
    for (uint32_t value = 0xa1; value <= 0xac; ++value)
        values.push_back(value);
    for (uint32_t value = 0xae; value <= 0xff; ++value)
        values.push_back(value);
    const auto direct = values;
    std::vector<std::string> encoder(256);
    for (size_t index = 0; index < direct.size(); ++index) {
        std::string encoded;
        appendUtf8(encoded, direct[index]);
        encoder[direct[index]] = encoded;
    }
    uint32_t extra = 0;
    for (uint32_t byte = 0; byte < 256; ++byte) {
        if (!encoder[byte].empty())
            continue;
        appendUtf8(encoder[byte], 256 + extra++);
    }
    return encoder;
}

} // namespace

ClipTokenizer::ClipTokenizer(const std::filesystem::path& directory)
    : vocabulary(parseVocabulary(readText(directory / "vocab.json"))),
      byteEncoder(makeByteEncoder()) {
    const auto bos = vocabulary.find("<|startoftext|>");
    const auto eos = vocabulary.find("<|endoftext|>");
    if (bos == vocabulary.end() || eos == vocabulary.end()) {
        throw std::runtime_error("CLIP vocabulary does not define its boundary tokens");
    }
    beginningOfText = bos->second;
    endOfText = eos->second;

    std::istringstream input(readText(directory / "merges.txt"));
    std::string line;
    size_t rank = 0;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line.empty() || line.front() == '#')
            continue;
        const size_t separator = line.find(' ');
        if (separator == std::string::npos)
            continue;
        mergeRanks.emplace(std::make_pair(line.substr(0, separator), line.substr(separator + 1)), rank++);
    }
}

std::vector<std::string> ClipTokenizer::pretokenize(const std::string& text) const {
    const auto characters = decodeUtf8(cleanText(text));
    std::vector<std::string> tokens;
    for (size_t position = 0; position < characters.size();) {
        std::string token;
        if (characters[position].codepoint == ' ') {
            ++position;
            continue;
        }
        const auto matches = [&](const char* suffix) {
            std::string candidate;
            for (size_t index = position; index < characters.size() && candidate.size() < 4; ++index) {
                candidate += characters[index].bytes;
                if (candidate == suffix)
                    return index + 1;
            }
            return size_t{0};
        };
        size_t contractionEnd = 0;
        if (characters[position].codepoint == '\'') {
            for (const char* contraction : {"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"}) {
                contractionEnd = matches(contraction);
                if (contractionEnd)
                    break;
            }
        }
        if (contractionEnd) {
            while (position < contractionEnd)
                token += characters[position++].bytes;
        } else if (isNumber(characters[position].codepoint) || isCjkIdeograph(characters[position].codepoint)) {
            token = characters[position++].bytes;
        } else {
            const bool letters = isLetter(characters[position].codepoint);
            while (position < characters.size() && !isWhitespace(characters[position].codepoint) &&
                   !isNumber(characters[position].codepoint) && !isCjkIdeograph(characters[position].codepoint) &&
                   isLetter(characters[position].codepoint) == letters) {
                token += characters[position++].bytes;
            }
        }
        if (!token.empty())
            tokens.push_back(std::move(token));
    }
    return tokens;
}

std::vector<std::string> ClipTokenizer::bytePairEncode(const std::string& token) const {
    std::vector<std::string> word;
    word.reserve(token.size());
    for (const unsigned char byte : token)
        word.push_back(byteEncoder[byte]);
    if (word.empty())
        return {};
    word.back() += "</w>";
    while (word.size() > 1) {
        size_t bestRank = std::numeric_limits<size_t>::max();
        std::pair<std::string, std::string> best;
        for (size_t index = 0; index + 1 < word.size(); ++index) {
            const auto candidate = std::make_pair(word[index], word[index + 1]);
            const auto rank = mergeRanks.find(candidate);
            if (rank != mergeRanks.end() && rank->second < bestRank) {
                bestRank = rank->second;
                best = candidate;
            }
        }
        if (bestRank == std::numeric_limits<size_t>::max())
            break;
        std::vector<std::string> merged;
        for (size_t index = 0; index < word.size();) {
            if (index + 1 < word.size() && word[index] == best.first && word[index + 1] == best.second) {
                merged.push_back(word[index] + word[index + 1]);
                index += 2;
            } else {
                merged.push_back(word[index++]);
            }
        }
        word = std::move(merged);
    }
    return word;
}

std::vector<int64_t> ClipTokenizer::encode(const std::string& text) const {
    std::vector<int64_t> result{beginningOfText};
    for (const auto& token : pretokenize(text)) {
        for (const auto& piece : bytePairEncode(token)) {
            const auto value = vocabulary.find(piece);
            result.push_back(value == vocabulary.end() ? endOfText : value->second);
        }
    }
    result.push_back(endOfText);
    if (result.size() > SequenceLength) {
        result.resize(SequenceLength);
        result.back() = endOfText;
    }
    result.resize(SequenceLength, endOfText);
    return result;
}

std::vector<int64_t> ClipTokenizer::encodePair(const std::string& negativePrompt, const std::string& prompt) const {
    auto result = encode(negativePrompt);
    const auto positive = encode(prompt);
    result.insert(result.end(), positive.begin(), positive.end());
    return result;
}

std::vector<int64_t> ClipTokenizer::attentionMask(const std::vector<int64_t>& tokenIds) const {
    if (tokenIds.size() % SequenceLength != 0) {
        throw std::runtime_error("CLIP token batch does not contain complete sequences");
    }
    std::vector<int64_t> result(tokenIds.size());
    for (size_t sequence = 0; sequence < tokenIds.size(); sequence += SequenceLength) {
        bool ended = false;
        for (size_t index = sequence; index < sequence + SequenceLength; ++index) {
            result[index] = ended ? 0 : 1;
            if (tokenIds[index] == endOfText)
                ended = true;
        }
    }
    return result;
}

} // namespace klartraum
