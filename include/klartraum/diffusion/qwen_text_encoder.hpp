// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_DIFFUSION_QWEN_TEXT_ENCODER_HPP
#define KLARTRAUM_DIFFUSION_QWEN_TEXT_ENCODER_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "klartraum/computegraph/computegraphgroup.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/gguf/gguf_file.hpp"
#include "klartraum/gguf/gguf_tokenizer.hpp"
#include "klartraum/gguf/gguf_weight_store.hpp"
#include "klartraum/vulkan_context.hpp"

namespace klartraum {

/** @brief Options of a QwenTextEncoder. */
struct QwenTextEncoderOptions {
    /** Tokens per submission; shorter prompts are padded. */
    uint32_t maxTokens = 256;
    /** Runs only the first @p layers layers without the final norm, for tests (0: all, with the norm). */
    uint32_t layers = 0;
};

/**
 * @brief Qwen-Image's text encoder: the language model of Qwen2.5-VL-7B (a `qwen2vl` GGUF) over a whole prompt at
 * once, returning the final normed hidden states of every token as the prompt embedding.
 *
 * The prompt is processed in one submission of a fixed `maxTokens` tokens with causal attention; a shorter prompt is
 * padded at the end, which leaves its tokens' states unchanged. Text-only M-RoPE equals plain NEOX rotary positions;
 * the query and key rows are reordered on upload so that each head's rotate-half pairs (i, i + 64) are adjacent for
 * dit_layers::rotary() (the attention scores are unchanged by a common reordering of queries and keys).
 *
 * ```
 * auto encoder = context.create<QwenTextEncoder>("Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf");
 * ComputeGraph graph(context, 1);
 * graph.compileFrom(encoder);
 * encoder->setTokens(encoder->promptTokens("a bonsai tree"));
 * graph.submitAndWait(context.getGraphicsQueue(), 0);
 * auto embedding = encoder->readPromptEmbedding(); // [tokens - 34][3584]
 * ```
 */
class QwenTextEncoder : virtual public ComputeGraphElement, virtual public ComputeGraphGroup {
public:
    using FloatTensor = TensorElementSinglePath<float>;

    /** Tokens of the prompt template's system part, which Qwen-Image drops from the embedding. */
    static constexpr uint32_t kTemplateTokens = 34;

    /** @throws std::runtime_error If the file cannot be read or is not a `qwen2vl` model. */
    QwenTextEncoder(VulkanContext& vulkanContext, const std::string& modelPath, QwenTextEncoderOptions options = {});
    ~QwenTextEncoder();

    uint32_t getHidden() const { return hidden; }
    uint32_t getMaxTokens() const { return maxTokens; }
    const GgufTokenizer& getTokenizer() const { return *tokenizer; }

    /** @brief The token ids of @p prompt in Qwen-Image's prompt template. */
    std::vector<int32_t> promptTokens(const std::string& prompt) const;

    /** @brief Sets the tokens of the next submission (1 .. maxTokens). @throws For invalid tokens or counts. */
    void setTokens(const std::vector<int32_t>& tokens);

    /**
     * @brief The final normed hidden states of the last submission's tokens, [tokens][hidden] (with
     * QwenTextEncoderOptions::layers, the states after these layers).
     */
    std::vector<float> readHiddenStates();

    /** @brief readHiddenStates() without the template's first kTemplateTokens tokens: the prompt embedding. */
    std::vector<float> readPromptEmbedding();

    /** @brief GPU bytes of the uploaded weights. */
    uint64_t getWeightBytes() const { return weights->getBytes(); }

    // ComputeGraphGroup interface
    void checkInput(ComputeGraphElementPtr input, int index = 0) override {}
    void _setup(VulkanContext& vulkanContext, uint32_t numberPaths, VkQueue setupQueue = VK_NULL_HANDLE) override;
    void _record(VkCommandBuffer commandBuffer, uint32_t pathId) override {}
    const char* getType() const override { return "QwenTextEncoder"; }

private:
    using WordTensor = TensorElementSinglePath<uint32_t>;

    VulkanContext* vulkanContext;
    std::unique_ptr<GgufFile> file;
    std::unique_ptr<GgufTokenizer> tokenizer;
    std::unique_ptr<GgufWeightStore> weights;
    uint32_t maxTokens;
    uint32_t layerLimit;
    uint32_t tokenCount = 0;
    uint32_t hidden = 0, feedForward = 0, layers = 0, heads = 0, kvHeads = 0, vocabulary = 0;
    float ropeBase = 1e6f, epsilon = 1e-6f;

    std::shared_ptr<WordTensor> params;          // {tokens, 0, 0, 0} for rmsNorm and swiGlu
    std::shared_ptr<FloatTensor> residual;       // [maxTokens][hidden], set to the embeddings by setTokens()
    std::shared_ptr<FloatTensor> output;         // [maxTokens][hidden], the final normed states (or residual)
    std::shared_ptr<FloatTensor> cosines, sines; // rotary tables [maxTokens][64]
    std::vector<ComputeGraphElementPtr> operations;

    std::shared_ptr<FloatTensor> deviceTensor(size_t elements, bool hostVisible = false);
    void append(const std::string& name, const ComputeGraphElementPtr& operation,
                const std::vector<ComputeGraphElementPtr>& slots);
    void matMul(const std::string& weight, const std::shared_ptr<FloatTensor>& input,
                const std::shared_ptr<FloatTensor>& out, bool accumulate, bool rotaryRows = false);
    void build();
};

} // namespace klartraum

#endif // KLARTRAUM_DIFFUSION_QWEN_TEXT_ENCODER_HPP
