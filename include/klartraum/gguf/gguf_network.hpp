// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_GGUF_GGUF_NETWORK_HPP
#define KLARTRAUM_GGUF_GGUF_NETWORK_HPP

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "klartraum/computegraph/computegraphgroup.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/gguf/gguf_file.hpp"
#include "klartraum/vulkan_context.hpp"

namespace klartraum {

/** @brief Hyperparameters of a `qwen35` (Qwen3.5 / Qwen3.6 dense) GGUF model. */
struct Qwen35Config {
    uint32_t vocabulary = 0;
    uint32_t hidden = 0;
    uint32_t feedForward = 0;
    uint32_t layers = 0;
    uint32_t heads = 0;
    uint32_t kvHeads = 0;
    uint32_t headDim = 0;
    uint32_t ropeDims = 0;
    float ropeBase = 1e7f;
    float epsilon = 1e-6f;
    uint32_t keyHeads = 0;       ///< linear attention key heads (ssm.group_count)
    uint32_t valueHeads = 0;     ///< linear attention value heads (ssm.time_step_rank)
    uint32_t keyDim = 0;         ///< ssm.state_size
    uint32_t valueDim = 0;       ///< ssm.inner_size / valueHeads
    std::vector<bool> recurrent; ///< per layer: Gated DeltaNet (true) or full attention

    /** @brief Reads the configuration from GGUF metadata; throws for other architectures. */
    static Qwen35Config fromFile(const GgufFile& file);
    uint32_t convChannels() const { return 2 * keyHeads * keyDim + valueHeads * valueDim; }
};

/** @brief Options of a GgufNetwork. */
struct GgufNetworkOptions {
    /** Tokens per submission (prompt chunk size), 1 .. gguf_layers::kMaxTokens. */
    uint32_t maxTokens = 16;
    /** Positions the key/value caches hold. */
    uint32_t contextLength = 4096;
    /** Compute logits for every token of a submission instead of only the last one. */
    bool logitsForAllTokens = false;
};

/**
 * @brief Runs a GGUF language model as a Klartraum compute-graph group.
 *
 * The GGUF counterpart of OnnxNetwork: it memory-maps the file, builds the
 * model's layers (shaders/gguf) as graph elements and uploads the weights,
 * which stay in their GGUF encoding on the GPU, at setup. Supported
 * architecture: `qwen35` (Qwen3.5 / Qwen3.6 dense, e.g. Qwen3.6-27B).
 *
 * The graph processes up to `maxTokens` tokens per submission. Before each
 * submission, setTokens() places the tokens' embeddings (looked up on the
 * CPU, so the embedding table needs no GPU memory) and their position; the
 * key/value caches and the linear-attention state carry the context from
 * one submission to the next. After the submission, readLogits() returns
 * the logits of the last token (or of all tokens, see GgufNetworkOptions).
 *
 * ```
 * auto network = context.create<GgufNetwork>("model.gguf");
 * ComputeGraph graph(context, 1);
 * graph.compileFrom(network);
 * network->setTokens(promptChunk, 0);
 * graph.submitAndWait(context.getGraphicsQueue(), 0);
 * auto logits = network->readLogits();
 * ```
 */
class GgufNetwork : virtual public ComputeGraphElement, virtual public ComputeGraphGroup {
public:
    /** @throws std::runtime_error If the file cannot be read or uses an unsupported architecture or type. */
    GgufNetwork(VulkanContext& vulkanContext, const std::string& modelPath, GgufNetworkOptions options = {});
    ~GgufNetwork();

    const GgufFile& getFile() const { return *file; }
    const Qwen35Config& getConfig() const { return config; }
    const GgufNetworkOptions& getOptions() const { return options; }
    /** @brief GPU bytes of the uploaded weights. */
    uint64_t getWeightBytes() const { return weightBytes; }
    /** @brief GPU bytes of the key/value caches and linear-attention states. */
    uint64_t getStateBytes() const;

    /**
     * @brief Sets the input of the next submission: @p tokens (1 .. maxTokens)
     * at positions @p position, @p position + 1, ...
     * @throws std::runtime_error For invalid token ids, chunk sizes, or positions beyond the context.
     */
    void setTokens(const std::vector<int32_t>& tokens, uint32_t position);

    /** @brief Logits of the last submission: [vocabulary], or [tokens][vocabulary] with logitsForAllTokens. */
    std::vector<float> readLogits();

    /** @brief Clears the linear-attention states, e.g. to start a new conversation at position 0. */
    void resetState();

    // ComputeGraphGroup interface
    void checkInput(ComputeGraphElementPtr input, int index = 0) override {}
    void _setup(VulkanContext& vulkanContext, uint32_t numberPaths) override;
    void _record(VkCommandBuffer commandBuffer, uint32_t pathId) override {}
    const char* getType() const override { return "GgufNetwork"; }

private:
    using FloatTensor = TensorElementSinglePath<float>;
    using WordTensor = TensorElementSinglePath<uint32_t>;

    VulkanContext* vulkanContext;
    std::unique_ptr<GgufFile> file;
    Qwen35Config config;
    GgufNetworkOptions options;
    uint32_t lastTokenCount = 0;
    uint64_t weightBytes = 0;

    std::shared_ptr<WordTensor> params;
    std::shared_ptr<FloatTensor> hidden;
    std::shared_ptr<FloatTensor> logits;
    std::vector<std::shared_ptr<FloatTensor>> recurrentStates; // conv and delta states, zeroed by resetState()
    std::vector<std::shared_ptr<FloatTensor>> caches;

    // GGUF weights for matVec (encoded) and small float tensors for the other layers.
    std::map<std::string, std::shared_ptr<WordTensor>> encodedWeights;
    std::map<std::string, std::shared_ptr<FloatTensor>> floatWeights;
    std::vector<ComputeGraphElementPtr> operations;

    std::shared_ptr<FloatTensor> deviceTensor(size_t elements, bool hostVisible = false);
    std::shared_ptr<WordTensor> encodedWeight(const std::string& name);
    std::shared_ptr<FloatTensor> floatWeight(const std::string& name);
    ComputeGraphElementPtr matVec(const std::string& weight, uint32_t inputWidth,
                                  const std::shared_ptr<FloatTensor>& input, const std::shared_ptr<FloatTensor>& output,
                                  bool accumulate = false, bool lastTokenOnly = false);
    void append(const ComputeGraphElementPtr& operation, const std::vector<ComputeGraphElementPtr>& slots);
    void buildQwen35();
};

} // namespace klartraum

#endif // KLARTRAUM_GGUF_GGUF_NETWORK_HPP
