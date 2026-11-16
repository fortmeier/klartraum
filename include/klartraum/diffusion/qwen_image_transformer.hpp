// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_DIFFUSION_QWEN_IMAGE_TRANSFORMER_HPP
#define KLARTRAUM_DIFFUSION_QWEN_IMAGE_TRANSFORMER_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "klartraum/computegraph/computegraphgroup.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/gguf/gguf_file.hpp"
#include "klartraum/gguf/gguf_weight_store.hpp"
#include "klartraum/vulkan_context.hpp"

namespace klartraum {

/** @brief Options of a QwenImageTransformer. */
struct QwenImageTransformerOptions {
    /** Latent size in latent pixels (the image size / 8), even. */
    uint32_t latentHeight = 64;
    uint32_t latentWidth = 64;
    /** Tokens of the prompt embedding (QwenTextEncoder::readPromptEmbedding()). */
    uint32_t textTokens = 0;
    /** A LoRA converted by scripts/qwen_image/convert_lora.py (e.g. Lightning), or empty. */
    std::string lora;
    /** Runs only the first @p blocks blocks without the output layers, for tests (0: all, with them). */
    uint32_t blocks = 0;
};

/**
 * @brief Qwen-Image's transformer (a `qwen_image` GGUF, the 60-block MMDiT of diffusers' QwenImageTransformer2DModel)
 * as a Klartraum compute-graph group: one submission predicts the flow-matching velocity of a packed latent at one
 * noise level.
 *
 * Image tokens are 2x2 patches of the 16-channel latent ([tokens][64], diffusers' _pack_latents order); the text
 * stream is the prompt embedding. Each block modulates both streams from the timestep embedding, attends jointly over
 * [text; image] with per-head RMS-normed queries and keys and 3-axis rotary positions (the image centred, the text
 * after it), and adds gated attention and GELU MLP outputs. The quantized weights stay encoded on the GPU; a LoRA
 * adds its low-rank products to its layers' outputs at run time.
 *
 * ```
 * auto transformer = context.create<QwenImageTransformer>("qwen-image-Q4_K_S.gguf", options);
 * ComputeGraph graph(context, 1);
 * graph.compileFrom(transformer);
 * transformer->setPromptEmbedding(embedding);
 * transformer->setLatent(latent);
 * transformer->setSigma(sigma);
 * graph.submitAndWait(context.getGraphicsQueue(), 0);
 * auto velocity = transformer->readVelocity();
 * ```
 */
class QwenImageTransformer : virtual public ComputeGraphElement, virtual public ComputeGraphGroup {
public:
    using FloatTensor = TensorElementSinglePath<float>;

    /** @throws std::runtime_error If a file cannot be read, is not a `qwen_image` model, or the options are invalid. */
    QwenImageTransformer(VulkanContext& vulkanContext, const std::string& modelPath,
                         QwenImageTransformerOptions options);
    ~QwenImageTransformer();

    const QwenImageTransformerOptions& getOptions() const { return options; }
    uint32_t getImageTokens() const { return imageTokens; }

    /** @brief The prompt embedding [textTokens][3584]. */
    void setPromptEmbedding(const std::vector<float>& embedding);
    /** @brief The packed latent [imageTokens][64]. */
    void setLatent(const std::vector<float>& latent);
    /** @brief The noise level of the next submission, sigma in [0, 1] (the timestep / 1000). */
    void setSigma(float sigma);

    /** @brief The velocity prediction of the last submission, [imageTokens][64]. */
    std::vector<float> readVelocity();
    /** @brief The image and text streams after the last block, [tokens][3072]. */
    std::vector<float> readImageStates();
    std::vector<float> readTextStates();

    /** @brief GPU bytes of the uploaded weights. */
    uint64_t getWeightBytes() const;

    // ComputeGraphGroup interface
    void checkInput(ComputeGraphElementPtr input, int index = 0) override {}
    void _setup(VulkanContext& vulkanContext, uint32_t numberPaths, VkQueue setupQueue = VK_NULL_HANDLE) override;
    void _record(VkCommandBuffer commandBuffer, uint32_t pathId) override {}
    const char* getType() const override { return "QwenImageTransformer"; }

private:
    using WordTensor = TensorElementSinglePath<uint32_t>;
    using Tensor = std::shared_ptr<FloatTensor>;

    VulkanContext* vulkanContext;
    QwenImageTransformerOptions options;
    std::unique_ptr<GgufFile> file;
    std::unique_ptr<GgufWeightStore> weights;
    std::unique_ptr<GgufFile> loraFile;
    std::unique_ptr<GgufWeightStore> loraWeights;
    uint32_t imageTokens = 0, textTokens = 0, layers = 0;

    std::shared_ptr<WordTensor> imageParams, textParams;     // {tokens, 0, 0, 0} for rmsNorm
    Tensor latent, prompt, sinusoid;                         // inputs: [imageTokens][64], [textTokens][3584], [256]
    Tensor image, text;                                      // residual streams [tokens][3072]
    Tensor velocity;                                         // output [imageTokens][64]
    Tensor imageCosines, imageSines, textCosines, textSines; // rotary tables [tokens][64]
    Tensor loraScratch;                                      // [max tokens][rank] of a LoRA's down projection
    std::vector<ComputeGraphElementPtr> operations;

    Tensor deviceTensor(size_t elements, bool hostVisible = false);
    void append(const std::string& name, const ComputeGraphElementPtr& operation,
                const std::vector<ComputeGraphElementPtr>& slots);
    // out (+)= W x + b (+ LoRA) for the weights `<name>.weight` and `<name>.bias` and `tokens` rows of x.
    void linear(const std::string& name, const Tensor& input, const Tensor& out, uint32_t tokens,
                bool accumulate = false);
    void build();
};

} // namespace klartraum

#endif // KLARTRAUM_DIFFUSION_QWEN_IMAGE_TRANSFORMER_HPP
