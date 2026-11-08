// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/diffusion/qwen_image_transformer.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "klartraum/diffusion/dit_layers.hpp"
#include "klartraum/gguf/gguf_layers.hpp"

namespace klartraum {

namespace gl = gguf_layers;
namespace dl = dit_layers;

namespace {

constexpr uint32_t kDim = 3072, kHeads = 24, kHeadDim = 128, kTextDim = 3584, kFeedForward = 12288;
constexpr uint32_t kPatchChannels = 64, kTimestepChannels = 256;
constexpr float kEpsilon = 1e-6f;
// Rotary angles per token: 8 for the frame, 28 each for the row and the column (axes of 16, 56 and 56 values).
constexpr uint32_t kAxisPairs[3] = {8, 28, 28};

// The 64 rotary angles of a token at the axis positions p, as QwenEmbedRope: position * 10000^(-2j / axis dim).
void rotaryAngles(const double (&p)[3], float* cosines, float* sines) {
    uint32_t pair = 0;
    for (uint32_t axis = 0; axis < 3; ++axis)
        for (uint32_t j = 0; j < kAxisPairs[axis]; ++j, ++pair) {
            const float frequency = 1.0f / std::pow(10000.0f, float(2 * j) / float(2 * kAxisPairs[axis]));
            const double angle = p[axis] * double(frequency);
            cosines[pair] = float(std::cos(angle));
            sines[pair] = float(std::sin(angle));
        }
}

} // namespace

QwenImageTransformer::QwenImageTransformer(VulkanContext& vulkanContext, const std::string& modelPath,
                                           QwenImageTransformerOptions options)
    : vulkanContext(&vulkanContext),
      options(options),
      file(std::make_unique<GgufFile>(modelPath)) {
    if (file->getString("general.architecture") != "qwen_image")
        throw std::runtime_error("QwenImageTransformer: " + modelPath + " is not a qwen_image model");
    if (options.latentHeight == 0 || options.latentWidth == 0 || options.latentHeight % 2 || options.latentWidth % 2)
        throw std::runtime_error("QwenImageTransformer: the latent size must be even and positive");
    if (options.textTokens == 0)
        throw std::runtime_error("QwenImageTransformer: textTokens must be positive");
    imageTokens = (options.latentHeight / 2) * (options.latentWidth / 2);
    textTokens = options.textTokens;
    while (file->hasTensor("transformer_blocks." + std::to_string(layers) + ".img_mod.1.weight"))
        ++layers;
    weights = std::make_unique<GgufWeightStore>(vulkanContext, *file);
    if (!options.lora.empty()) {
        loraFile = std::make_unique<GgufFile>(options.lora);
        if (loraFile->getString("general.architecture") != "qwen_image_lora")
            throw std::runtime_error("QwenImageTransformer: " + options.lora + " is not a converted LoRA");
        loraWeights = std::make_unique<GgufWeightStore>(vulkanContext, *loraFile);
    }
    if (!options.controlnet.empty()) {
        controlFile = std::make_unique<GgufFile>(options.controlnet);
        if (controlFile->getString("general.architecture") != "qwen_image_controlnet")
            throw std::runtime_error("QwenImageTransformer: " + options.controlnet + " is not a converted ControlNet");
        while (controlFile->hasTensor("controlnet_blocks." + std::to_string(controlLayers) + ".weight"))
            ++controlLayers;
        controlWeights = std::make_unique<GgufWeightStore>(vulkanContext, *controlFile);
    }
    build();
}

QwenImageTransformer::~QwenImageTransformer() = default;

uint64_t QwenImageTransformer::getWeightBytes() const {
    return weights->getBytes() + (loraWeights ? loraWeights->getBytes() : 0) +
           (controlWeights ? controlWeights->getBytes() : 0);
}

QwenImageTransformer::Tensor QwenImageTransformer::deviceTensor(size_t elements, bool hostVisible) {
    constexpr VkBufferUsageFlags usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    const VkMemoryPropertyFlags memory =
        hostVisible ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                    : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    return vulkanContext->create<FloatTensor>(std::vector<uint32_t>{uint32_t(elements)}, usage,
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                              memory);
}

void QwenImageTransformer::append(const std::string& name, const ComputeGraphElementPtr& operation,
                                  const std::vector<ComputeGraphElementPtr>& slots) {
    operation->setName(name);
    for (size_t i = 0; i < slots.size(); ++i)
        operation->setInput(slots[i], int(i));
    // Scratch tensors are reused across blocks, so every operation follows the previous one.
    if (!operations.empty())
        operation->addDependency(operations.back());
    operations.push_back(operation);
}

void QwenImageTransformer::linear(const WeightSource& source, const std::string& name, const Tensor& input,
                                  const Tensor& out, uint32_t tokens, bool accumulate) {
    const auto& info = source.file->getTensor(name + ".weight");
    const uint32_t rows = uint32_t(info.rowCount()), columns = uint32_t(info.rowLength());
    const bool hasBias = source.file->hasTensor(name + ".bias");
    ComputeGraphElementPtr bias = hasBias ? ComputeGraphElementPtr(source.weights->floats(name + ".bias")) : input;
    append(name, gl::matMul(*vulkanContext, info.type, rows, columns, tokens, hasBias, accumulate),
           {source.weights->encoded(name + ".weight"), input, bias, out});
    if (!source.lora || !source.lora->hasTensor(name + ".lora_down.weight"))
        return;
    const auto& down = source.lora->getTensor(name + ".lora_down.weight");
    const auto& up = source.lora->getTensor(name + ".lora_up.weight");
    const uint32_t rank = uint32_t(down.rowCount());
    if (down.rowLength() != columns || up.rowCount() != rows || up.rowLength() != rank ||
        uint64_t(tokens) * rank > loraScratch->getDataElementCount())
        throw std::runtime_error("QwenImageTransformer: the LoRA of " + name + " does not fit the layer");
    append(name + ".lora_down", gl::matMul(*vulkanContext, down.type, rank, columns, tokens),
           {source.loraWeights->encoded(name + ".lora_down.weight"), input, input, loraScratch});
    append(name + ".lora_up", gl::matMul(*vulkanContext, up.type, rows, rank, tokens, false, true),
           {source.loraWeights->encoded(name + ".lora_up.weight"), loraScratch, loraScratch, out});
}

void QwenImageTransformer::build() {
    const uint32_t Ti = imageTokens, Tt = textTokens, joint = Ti + Tt, maxTokens = std::max(Ti, Tt);
    imageParams = vulkanContext->create<WordTensor>(std::vector<uint32_t>{4});
    textParams = vulkanContext->create<WordTensor>(std::vector<uint32_t>{4});
    latent = deviceTensor(size_t(Ti) * kPatchChannels, true);
    prompt = deviceTensor(size_t(Tt) * kTextDim, true);
    sinusoid = deviceTensor(kTimestepChannels, true);
    image = deviceTensor(size_t(Ti) * kDim, true);
    text = deviceTensor(size_t(Tt) * kDim, true);
    velocity = deviceTensor(size_t(Ti) * kPatchChannels, true);
    imageCosines = deviceTensor(size_t(Ti) * kHeadDim / 2, true);
    imageSines = deviceTensor(size_t(Ti) * kHeadDim / 2, true);
    textCosines = deviceTensor(size_t(Tt) * kHeadDim / 2, true);
    textSines = deviceTensor(size_t(Tt) * kHeadDim / 2, true);
    loraScratch = deviceTensor(size_t(maxTokens) * 128);
    if (controlFile) {
        control = deviceTensor(size_t(Ti) * kPatchChannels, true);
        controlScale = deviceTensor(kDim, true);
    }

    auto promptNormed = deviceTensor(size_t(Tt) * kTextDim);
    auto hiddenTime = deviceTensor(kDim);
    auto conditioning = deviceTensor(kDim); // silu(temb), the input of every modulation
    auto imageModulation = deviceTensor(6 * kDim), textModulation = deviceTensor(6 * kDim);
    auto imageNormed = deviceTensor(size_t(Ti) * kDim), textNormed = deviceTensor(size_t(Tt) * kDim);
    auto imageProjection = deviceTensor(size_t(Ti) * kDim), textProjection = deviceTensor(size_t(Tt) * kDim);
    auto jointQueries = deviceTensor(size_t(joint) * kDim), jointKeys = deviceTensor(size_t(joint) * kDim);
    auto jointValues = deviceTensor(size_t(joint) * kDim), jointOut = deviceTensor(size_t(joint) * kDim);
    auto feedForward = deviceTensor(size_t(maxTokens) * kFeedForward);
    auto heads = deviceTensor(size_t(maxTokens) * kDim); // per-head normed and turned queries or keys

    const WeightSource model{file.get(), weights.get(), loraFile.get(), loraWeights.get()};

    // A network's inputs: the patches and the prompt into its streams, and its timestep embedding.
    auto appendInputs = [&](const WeightSource& w, const Tensor& imageStream, const Tensor& textStream,
                            const Tensor& conditioning) {
        linear(w, "img_in", latent, imageStream, Ti);
        append("txt_norm", gl::rmsNorm(*vulkanContext, kTextDim, 1, kEpsilon, Tt),
               {prompt, w.weights->floats("txt_norm.weight"), textParams, promptNormed});
        linear(w, "txt_in", promptNormed, textStream, Tt);
        linear(w, "time_text_embed.timestep_embedder.linear_1", sinusoid, hiddenTime, 1);
        append("silu", dl::activation(*vulkanContext, dl::Activation::Silu, kDim), {hiddenTime});
        linear(w, "time_text_embed.timestep_embedder.linear_2", hiddenTime, conditioning, 1);
        append("silu", dl::activation(*vulkanContext, dl::Activation::Silu, kDim), {conditioning});
    };

    // A stream's queries, keys and values into its rows of the joint buffers, normed and turned.
    auto projectStream = [&](const WeightSource& w, const std::string& block, const Tensor& normed, uint32_t tokens,
                             uint32_t firstRow, const std::shared_ptr<WordTensor>& params, const Tensor& cosines,
                             const Tensor& sines, bool isText) {
        const std::string q = isText ? "attn.add_q_proj" : "attn.to_q", k = isText ? "attn.add_k_proj" : "attn.to_k",
                          v = isText ? "attn.add_v_proj" : "attn.to_v";
        const std::string normQ = isText ? "attn.norm_added_q" : "attn.norm_q",
                          normK = isText ? "attn.norm_added_k" : "attn.norm_k";
        const uint32_t count = tokens * kDim, offset = firstRow * kDim;
        const auto& projection = isText ? textProjection : imageProjection;
        for (const auto& [weight, norm, target] :
             {std::tuple{q, normQ, jointQueries}, std::tuple{k, normK, jointKeys}}) {
            linear(w, block + weight, normed, projection, tokens);
            append("qk_norm", gl::rmsNorm(*vulkanContext, kHeadDim, kHeads, kEpsilon, tokens),
                   {projection, w.weights->floats(block + norm + ".weight"), params, heads});
            append("rotary", dl::rotary(*vulkanContext, tokens, kHeads), {cosines, sines, heads});
            append("copy", dl::copy(*vulkanContext, count, 0, offset), {heads, target});
        }
        linear(w, block + v, normed, projection, tokens);
        append("copy", dl::copy(*vulkanContext, count, 0, offset), {projection, jointValues});
    };

    // One double-stream block (`block` the tensor name prefix) on the streams.
    auto appendBlock = [&](const WeightSource& w, const std::string& block, const Tensor& imageStream,
                           const Tensor& textStream, const Tensor& conditioning) {
        linear(w, block + "img_mod.1", conditioning, imageModulation, 1);
        linear(w, block + "txt_mod.1", conditioning, textModulation, 1);

        // Attention over the joint [text; image] sequence.
        append("img_norm1", dl::layerNormModulate(*vulkanContext, kDim, Ti, 0, kDim, kEpsilon),
               {imageStream, imageModulation, imageNormed});
        append("txt_norm1", dl::layerNormModulate(*vulkanContext, kDim, Tt, 0, kDim, kEpsilon),
               {textStream, textModulation, textNormed});
        projectStream(w, block, imageNormed, Ti, Tt, imageParams, imageCosines, imageSines, false);
        projectStream(w, block, textNormed, Tt, 0, textParams, textCosines, textSines, true);
        append("attention", gl::fullAttention(*vulkanContext, joint, kHeads, false),
               {jointQueries, jointKeys, jointValues, jointOut});
        append("copy", dl::copy(*vulkanContext, Ti * kDim, Tt * kDim, 0), {jointOut, imageNormed});
        linear(w, block + "attn.to_out.0", imageNormed, imageProjection, Ti);
        append("img_gate1", dl::gatedResidual(*vulkanContext, kDim, Ti, 2 * kDim),
               {imageProjection, imageModulation, imageStream});
        append("copy", dl::copy(*vulkanContext, Tt * kDim, 0, 0), {jointOut, textNormed});
        linear(w, block + "attn.to_add_out", textNormed, textProjection, Tt);
        append("txt_gate1", dl::gatedResidual(*vulkanContext, kDim, Tt, 2 * kDim),
               {textProjection, textModulation, textStream});

        // MLPs.
        for (const bool isText : {false, true}) {
            const uint32_t tokens = isText ? Tt : Ti;
            const auto& stream = isText ? textStream : imageStream;
            const auto& modulation = isText ? textModulation : imageModulation;
            const auto& normed = isText ? textNormed : imageNormed;
            const auto& projection = isText ? textProjection : imageProjection;
            const std::string mlp = block + (isText ? "txt_mlp" : "img_mlp");
            append("norm2", dl::layerNormModulate(*vulkanContext, kDim, tokens, 3 * kDim, 4 * kDim, kEpsilon),
                   {stream, modulation, normed});
            linear(w, mlp + ".net.0.proj", normed, feedForward, tokens);
            append("gelu", dl::activation(*vulkanContext, dl::Activation::GeluTanh, tokens * kFeedForward),
                   {feedForward});
            linear(w, mlp + ".net.2", feedForward, projection, tokens);
            append("gate2", dl::gatedResidual(*vulkanContext, kDim, tokens, 5 * kDim),
                   {projection, modulation, stream});
        }
    };

    // The ControlNet: its own inputs, with the control latent embedded into its image stream, its blocks, and per
    // block a projection of its image stream into a residual.
    if (controlFile) {
        const WeightSource net{controlFile.get(), controlWeights.get(), nullptr, nullptr};
        auto controlImage = deviceTensor(size_t(Ti) * kDim), controlText = deviceTensor(size_t(Tt) * kDim);
        auto controlConditioning = deviceTensor(kDim);
        appendInputs(net, controlImage, controlText, controlConditioning);
        linear(net, "controlnet_x_embedder", control, controlImage, Ti, true);
        for (uint32_t index = 0; index < controlLayers; ++index) {
            appendBlock(net, "transformer_blocks." + std::to_string(index) + ".", controlImage, controlText,
                        controlConditioning);
            controlResiduals.push_back(deviceTensor(size_t(Ti) * kDim, true));
            linear(net, "controlnet_blocks." + std::to_string(index), controlImage, controlResiduals.back(), Ti);
        }
    }

    appendInputs(model, image, text, conditioning);
    const uint32_t blockCount = options.blocks != 0 ? std::min(options.blocks, layers) : layers;
    // diffusers: block i gets the residual i / ceil(blocks / residuals), times the conditioning scale.
    const uint32_t interval = controlLayers != 0 ? (layers + controlLayers - 1) / controlLayers : 1;
    for (uint32_t index = 0; index < blockCount; ++index) {
        appendBlock(model, "transformer_blocks." + std::to_string(index) + ".", image, text, conditioning);
        if (controlFile)
            append("control", dl::gatedResidual(*vulkanContext, kDim, Ti, 0),
                   {controlResiduals[index / interval], controlScale, image});
    }

    if (options.blocks == 0) {
        // AdaLayerNormContinuous: scale first, then shift.
        auto outputModulation = deviceTensor(2 * kDim);
        linear(model, "norm_out.linear", conditioning, outputModulation, 1);
        append("norm_out", dl::layerNormModulate(*vulkanContext, kDim, Ti, kDim, 0, kEpsilon),
               {image, outputModulation, imageNormed});
        linear(model, "proj_out", imageNormed, velocity, Ti);
    }

    inputs[0] = options.blocks == 0 ? velocity : image;
    srcOutputSlots[0] = -1;
    outputElements[0] = operations.back();
}

void QwenImageTransformer::_setup(VulkanContext& vulkanContext, uint32_t numberPaths, VkQueue setupQueue) {
    weights->upload(setupQueue);
    if (loraWeights)
        loraWeights->upload(setupQueue);
    if (controlWeights) {
        controlWeights->upload(setupQueue);
        setControlScale(1.0f);
    }
    imageParams->getDataBuffer().memcopyFrom(std::vector<uint32_t>{imageTokens, 0, 0, 0});
    textParams->getDataBuffer().memcopyFrom(std::vector<uint32_t>{textTokens, 0, 0, 0});

    // Rotary positions: the image rows and columns centred on zero (scale_rope), the frame at 0; the text tokens
    // after the image's largest half extent on all three axes.
    const uint32_t rows = options.latentHeight / 2, columns = options.latentWidth / 2, pairs = kHeadDim / 2;
    std::vector<float> c(size_t(imageTokens) * pairs), s(c.size());
    for (uint32_t y = 0; y < rows; ++y)
        for (uint32_t x = 0; x < columns; ++x) {
            const double p[3] = {0.0, double(y) - double(rows - rows / 2), double(x) - double(columns - columns / 2)};
            rotaryAngles(p, c.data() + (y * columns + x) * pairs, s.data() + (y * columns + x) * pairs);
        }
    imageCosines->getDataBuffer().memcopyFrom(c);
    imageSines->getDataBuffer().memcopyFrom(s);
    c.assign(size_t(textTokens) * pairs, 0.0f);
    s.assign(c.size(), 0.0f);
    const uint32_t first = std::max(rows / 2, columns / 2);
    for (uint32_t t = 0; t < textTokens; ++t) {
        const double position = double(first + t), p[3] = {position, position, position};
        rotaryAngles(p, c.data() + t * pairs, s.data() + t * pairs);
    }
    textCosines->getDataBuffer().memcopyFrom(c);
    textSines->getDataBuffer().memcopyFrom(s);
    std::cout << "QwenImageTransformer: " << operations.size() << " operations, " << (getWeightBytes() >> 20)
              << " MiB weights, " << imageTokens << " image and " << textTokens << " text tokens" << std::endl;
}

void QwenImageTransformer::setPromptEmbedding(const std::vector<float>& embedding) {
    if (embedding.size() != size_t(textTokens) * kTextDim)
        throw std::runtime_error("QwenImageTransformer: the prompt embedding must hold textTokens x 3584 values");
    prompt->getDataBuffer().memcopyFrom(embedding);
}

void QwenImageTransformer::setLatent(const std::vector<float>& values) {
    if (values.size() != size_t(imageTokens) * kPatchChannels)
        throw std::runtime_error("QwenImageTransformer: the latent must hold imageTokens x 64 values");
    latent->getDataBuffer().memcopyFrom(values);
}

void QwenImageTransformer::setSigma(float sigma) {
    // diffusers' Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000): [cos, sin] of
    // 1000 sigma exp(-ln(10000) k / 128).
    std::vector<float> values(kTimestepChannels);
    const uint32_t half = kTimestepChannels / 2;
    for (uint32_t k = 0; k < half; ++k) {
        const float frequency = std::exp(-std::log(10000.0f) * float(k) / float(half));
        const float argument = 1000.0f * (sigma * frequency);
        values[k] = std::cos(argument);
        values[half + k] = std::sin(argument);
    }
    sinusoid->getDataBuffer().memcopyFrom(values);
}

void QwenImageTransformer::setControl(const std::vector<float>& values) {
    if (!control)
        throw std::runtime_error("QwenImageTransformer: setControl() needs a ControlNet");
    if (values.size() != size_t(imageTokens) * kPatchChannels)
        throw std::runtime_error("QwenImageTransformer: the control latent must hold imageTokens x 64 values");
    control->getDataBuffer().memcopyFrom(values);
}

void QwenImageTransformer::setControlScale(float scale) {
    if (!controlScale)
        throw std::runtime_error("QwenImageTransformer: setControlScale() needs a ControlNet");
    controlScale->getDataBuffer().memcopyFrom(std::vector<float>(kDim, scale));
}

std::vector<float> QwenImageTransformer::readControlResiduals() {
    std::vector<float> values;
    std::vector<float> residual(size_t(imageTokens) * kDim);
    for (const auto& tensor : controlResiduals) {
        tensor->getDataBuffer().memcopyTo(residual);
        values.insert(values.end(), residual.begin(), residual.end());
    }
    return values;
}

std::vector<float> QwenImageTransformer::readVelocity() {
    std::vector<float> values(size_t(imageTokens) * kPatchChannels);
    velocity->getDataBuffer().memcopyTo(values);
    return values;
}

std::vector<float> QwenImageTransformer::readImageStates() {
    std::vector<float> values(size_t(imageTokens) * kDim);
    image->getDataBuffer().memcopyTo(values);
    return values;
}

std::vector<float> QwenImageTransformer::readTextStates() {
    std::vector<float> values(size_t(textTokens) * kDim);
    text->getDataBuffer().memcopyTo(values);
    return values;
}

} // namespace klartraum
