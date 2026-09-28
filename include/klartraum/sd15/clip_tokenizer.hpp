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

class ClipTokenizer {
public:
    static constexpr size_t SequenceLength = 77;

    explicit ClipTokenizer(const std::filesystem::path& directory);

    std::vector<int64_t> encode(const std::string& text) const;
    std::vector<int64_t> encodePair(
        const std::string& negativePrompt, const std::string& prompt) const;
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
