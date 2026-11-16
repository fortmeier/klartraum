// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/diffusion/qwen_text_encoder.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "klartraum/diffusion/dit_layers.hpp"
#include "klartraum/gguf/gguf_layers.hpp"
#include "klartraum/gguf/gguf_quants.hpp"

namespace klartraum {

namespace gl = gguf_layers;

namespace {

constexpr uint32_t kHeadDim = 128;

const char* const kTemplate = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
                              "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
                              "<|im_start|>user\n";
const char* const kTemplateEnd = "<|im_end|>\n<|im_start|>assistant\n";

// Row i of a head in the reordered layout takes row i / 2 (even i) or 64 + i / 2 (odd i) of the original head.
uint64_t rotaryRow(uint64_t row) {
    const uint64_t head = row / kHeadDim, i = row % kHeadDim;
    return head * kHeadDim + (i % 2 == 0 ? i / 2 : kHeadDim / 2 + i / 2);
}

} // namespace

QwenTextEncoder::QwenTextEncoder(VulkanContext& vulkanContext, const std::string& modelPath,
                                 QwenTextEncoderOptions options)
    : vulkanContext(&vulkanContext),
      file(std::make_unique<GgufFile>(modelPath)),
      maxTokens(options.maxTokens),
      layerLimit(options.layers) {
    if (file->getString("general.architecture") != "qwen2vl")
        throw std::runtime_error("QwenTextEncoder: " + modelPath + " is not a qwen2vl model");
    if (maxTokens == 0)
        throw std::runtime_error("QwenTextEncoder: maxTokens must be positive");
    hidden = uint32_t(file->getInteger("qwen2vl.embedding_length"));
    feedForward = uint32_t(file->getInteger("qwen2vl.feed_forward_length"));
    layers = uint32_t(file->getInteger("qwen2vl.block_count"));
    heads = uint32_t(file->getInteger("qwen2vl.attention.head_count"));
    kvHeads = uint32_t(file->getInteger("qwen2vl.attention.head_count_kv"));
    ropeBase = float(file->getNumber("qwen2vl.rope.freq_base", 1e6));
    epsilon = float(file->getNumber("qwen2vl.attention.layer_norm_rms_epsilon", 1e-6));
    vocabulary = uint32_t(file->getTensor("token_embd.weight").rowCount());
    if (hidden != heads * kHeadDim)
        throw std::runtime_error("QwenTextEncoder: heads of 128 values expected");
    tokenizer = std::make_unique<GgufTokenizer>(*file);
    weights = std::make_unique<GgufWeightStore>(vulkanContext, *file);
    build();
}

QwenTextEncoder::~QwenTextEncoder() = default;

std::shared_ptr<QwenTextEncoder::FloatTensor> QwenTextEncoder::deviceTensor(size_t elements, bool hostVisible) {
    constexpr VkBufferUsageFlags usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    const VkMemoryPropertyFlags memory =
        hostVisible ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                    : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    return vulkanContext->create<FloatTensor>(std::vector<uint32_t>{uint32_t(elements)}, usage,
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                              memory);
}

void QwenTextEncoder::append(const std::string& name, const ComputeGraphElementPtr& operation,
                             const std::vector<ComputeGraphElementPtr>& slots) {
    operation->setName(name);
    for (size_t i = 0; i < slots.size(); ++i)
        operation->setInput(slots[i], int(i));
    // Scratch tensors are reused across layers, so every operation follows the previous one.
    if (!operations.empty())
        operation->addDependency(operations.back());
    operations.push_back(operation);
}

void QwenTextEncoder::matMul(const std::string& weight, const std::shared_ptr<FloatTensor>& input,
                             const std::shared_ptr<FloatTensor>& out, bool accumulate, bool rotaryRows) {
    const auto& info = file->getTensor(weight + ".weight");
    const std::string biasName = weight + ".bias";
    const bool hasBias = file->hasTensor(biasName);
    const GgufWeightStore::RowOrder order = rotaryRows ? GgufWeightStore::RowOrder(rotaryRow) : nullptr;
    auto operation = gl::matMul(*vulkanContext, info.type, uint32_t(info.rowCount()), uint32_t(info.rowLength()),
                                maxTokens, hasBias, accumulate);
    ComputeGraphElementPtr bias = hasBias ? ComputeGraphElementPtr(weights->floats(biasName, order)) : input;
    append(weight, operation, {weights->encoded(weight + ".weight", order), input, bias, out});
}

void QwenTextEncoder::build() {
    const uint32_t T = maxTokens, kvWidth = kvHeads * kHeadDim;
    params = vulkanContext->create<WordTensor>(std::vector<uint32_t>{4});
    residual = deviceTensor(size_t(T) * hidden, true);
    const uint32_t layerCount = layerLimit != 0 ? std::min(layerLimit, layers) : layers;
    output = layerLimit != 0 ? residual : deviceTensor(size_t(T) * hidden, true);
    cosines = deviceTensor(size_t(T) * kHeadDim / 2, true);
    sines = deviceTensor(size_t(T) * kHeadDim / 2, true);
    auto normed = deviceTensor(size_t(T) * hidden);
    auto queries = deviceTensor(size_t(T) * hidden);
    auto keys = deviceTensor(size_t(T) * kvWidth);
    auto values = deviceTensor(size_t(T) * kvWidth);
    auto attended = deviceTensor(size_t(T) * hidden);
    auto gate = deviceTensor(size_t(T) * feedForward);
    auto up = deviceTensor(size_t(T) * feedForward);
    auto gated = deviceTensor(size_t(T) * feedForward);

    for (uint32_t layer = 0; layer < layerCount; ++layer) {
        const std::string block = "blk." + std::to_string(layer) + ".";
        append("attn_norm", gl::rmsNorm(*vulkanContext, hidden, 1, epsilon, T),
               {residual, weights->floats(block + "attn_norm.weight"), params, normed});
        matMul(block + "attn_q", normed, queries, false, true);
        matMul(block + "attn_k", normed, keys, false, true);
        matMul(block + "attn_v", normed, values, false);
        append("rotary_q", dit_layers::rotary(*vulkanContext, T, heads), {cosines, sines, queries});
        append("rotary_k", dit_layers::rotary(*vulkanContext, T, kvHeads), {cosines, sines, keys});
        append("attention", gl::fullAttention(*vulkanContext, T, heads, true, kvHeads),
               {queries, keys, values, attended});
        matMul(block + "attn_output", attended, residual, true);
        append("ffn_norm", gl::rmsNorm(*vulkanContext, hidden, 1, epsilon, T),
               {residual, weights->floats(block + "ffn_norm.weight"), params, normed});
        matMul(block + "ffn_gate", normed, gate, false);
        matMul(block + "ffn_up", normed, up, false);
        append("swiglu", gl::swiGlu(*vulkanContext, feedForward, T), {gate, up, params, gated});
        matMul(block + "ffn_down", gated, residual, true);
    }
    if (output != residual)
        append("output_norm", gl::rmsNorm(*vulkanContext, hidden, 1, epsilon, T),
               {residual, weights->floats("output_norm.weight"), params, output});

    inputs[0] = output;
    srcOutputSlots[0] = -1;
    outputElements[0] = operations.back();
}

void QwenTextEncoder::_setup(VulkanContext& vulkanContext, uint32_t numberPaths, VkQueue setupQueue) {
    weights->upload(setupQueue);
    // Rotary angles of the positions 0 .. maxTokens - 1: position * base^(-i / 64) for the pairs i = 0 .. 63.
    std::vector<float> c(size_t(maxTokens) * kHeadDim / 2), s(c.size());
    for (uint32_t t = 0; t < maxTokens; ++t)
        for (uint32_t i = 0; i < kHeadDim / 2; ++i) {
            const double angle = double(t) * std::pow(double(ropeBase), -double(i) / double(kHeadDim / 2));
            c[t * kHeadDim / 2 + i] = float(std::cos(angle));
            s[t * kHeadDim / 2 + i] = float(std::sin(angle));
        }
    cosines->getDataBuffer().memcopyFrom(c);
    sines->getDataBuffer().memcopyFrom(s);
    // Every row is computed; rows past the prompt hold padding.
    params->getDataBuffer().memcopyFrom(std::vector<uint32_t>{maxTokens, 0, 0, 0});
    std::cout << "QwenTextEncoder: " << operations.size() << " operations, " << (weights->getBytes() >> 20)
              << " MiB weights" << std::endl;
}

std::vector<int32_t> QwenTextEncoder::promptTokens(const std::string& prompt) const {
    return tokenizer->encode(kTemplate + prompt + kTemplateEnd, true);
}

void QwenTextEncoder::setTokens(const std::vector<int32_t>& tokens) {
    if (tokens.empty() || tokens.size() > maxTokens)
        throw std::runtime_error("QwenTextEncoder: between 1 and " + std::to_string(maxTokens) + " tokens");
    const auto& embedding = file->getTensor("token_embd.weight");
    const uint64_t rowBytes = ggmlRowBytes(embedding.type, embedding.rowLength());
    std::vector<float> rows(size_t(maxTokens) * hidden);
    for (uint32_t t = 0; t < maxTokens; ++t) {
        // The padding repeats the last token.
        const int32_t token = tokens[std::min<size_t>(t, tokens.size() - 1)];
        if (token < 0 || uint32_t(token) >= vocabulary)
            throw std::runtime_error("QwenTextEncoder: token id " + std::to_string(token) + " is out of range");
        dequantizeRow(embedding.type, file->data(embedding) + uint64_t(token) * rowBytes, rows.data() + t * hidden,
                      hidden);
    }
    residual->getDataBuffer().memcopyFrom(rows);
    tokenCount = uint32_t(tokens.size());
}

std::vector<float> QwenTextEncoder::readHiddenStates() {
    std::vector<float> all(size_t(maxTokens) * hidden);
    output->getDataBuffer().memcopyTo(all);
    all.resize(size_t(tokenCount) * hidden);
    return all;
}

std::vector<float> QwenTextEncoder::readPromptEmbedding() {
    auto states = readHiddenStates();
    if (tokenCount <= kTemplateTokens)
        return {};
    return std::vector<float>(states.begin() + size_t(kTemplateTokens) * hidden, states.end());
}

} // namespace klartraum
