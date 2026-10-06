// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/gguf/gguf_network.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "klartraum/batched_upload.hpp"
#include "klartraum/gguf/gguf_layers.hpp"
#include "klartraum/gguf/gguf_quants.hpp"

namespace klartraum {

namespace gl = gguf_layers;

Qwen35Config Qwen35Config::fromFile(const GgufFile& file) {
    const std::string arch = file.getString("general.architecture");
    if (arch != "qwen35")
        throw std::runtime_error("Unsupported GGUF architecture: " + arch);
    auto count = [&](const std::string& key) { return uint32_t(file.getInteger(arch + "." + key)); };
    Qwen35Config config;
    config.vocabulary = uint32_t(file.getTensor("token_embd.weight").rowCount());
    config.hidden = count("embedding_length");
    config.feedForward = count("feed_forward_length");
    config.layers = count("block_count");
    config.heads = count("attention.head_count");
    config.kvHeads = count("attention.head_count_kv");
    config.headDim = count("attention.key_length");
    config.ropeDims = count("rope.dimension_count");
    config.ropeBase = float(file.getNumber(arch + ".rope.freq_base"));
    config.epsilon = float(file.getNumber(arch + ".attention.layer_norm_rms_epsilon"));
    config.keyHeads = count("ssm.group_count");
    config.valueHeads = count("ssm.time_step_rank");
    config.keyDim = count("ssm.state_size");
    config.valueDim = count("ssm.inner_size") / config.valueHeads;
    if (count("attention.value_length") != config.headDim)
        throw std::runtime_error("qwen35: key and value widths differ");
    if (count("ssm.conv_kernel") != 4)
        throw std::runtime_error("qwen35: only a convolution kernel of 4 is supported");
    const uint32_t interval = uint32_t(file.getInteger(arch + ".full_attention_interval", 4));
    for (uint32_t layer = 0; layer < config.layers; ++layer)
        config.recurrent.push_back((layer + 1) % interval != 0);
    return config;
}

GgufNetwork::GgufNetwork(VulkanContext& vulkanContext, const std::string& modelPath, GgufNetworkOptions options)
    : vulkanContext(&vulkanContext),
      file(std::make_unique<GgufFile>(modelPath)),
      options(options) {
    if (options.maxTokens < 1 || options.maxTokens > gl::kMaxTokens) {
        throw std::runtime_error("GgufNetwork: maxTokens must be between 1 and " + std::to_string(gl::kMaxTokens));
    }
    config = Qwen35Config::fromFile(*file);
    buildQwen35();
}

GgufNetwork::~GgufNetwork() = default;

std::shared_ptr<GgufNetwork::FloatTensor> GgufNetwork::deviceTensor(size_t elements, bool hostVisible) {
    constexpr VkBufferUsageFlags usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    const VkMemoryPropertyFlags memory =
        hostVisible ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                    : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    return vulkanContext->create<FloatTensor>(std::vector<uint32_t>{uint32_t(elements)}, usage,
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                              memory);
}

std::shared_ptr<GgufNetwork::WordTensor> GgufNetwork::encodedWeight(const std::string& name) {
    auto& weight = encodedWeights[name];
    if (!weight) {
        const auto& info = file->getTensor(name);
        if (!isSupportedWeightType(info.type)) {
            throw std::runtime_error("GgufNetwork: tensor " + name + " has unsupported type " +
                                     ggmlTypeName(info.type));
        }
        const uint64_t bytes = gl::gpuRowBytes(info.type, uint32_t(info.rowLength())) * info.rowCount();
        const uint64_t words = (bytes + gl::kWeightPaddingBytes + 3) / 4;
        if (words > UINT32_MAX)
            throw std::runtime_error("GgufNetwork: tensor " + name + " is too large");
        weight = vulkanContext->create<WordTensor>(
            std::vector<uint32_t>{uint32_t(words)},
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        weight->setName(name.c_str());
        weightBytes += words * 4;
    }
    return weight;
}

std::shared_ptr<GgufNetwork::FloatTensor> GgufNetwork::floatWeight(const std::string& name) {
    auto& weight = floatWeights[name];
    if (!weight) {
        weight = deviceTensor(file->getTensor(name).elementCount());
        weight->setName(name.c_str());
        weightBytes += weight->getBufferMemSize();
    }
    return weight;
}

uint64_t GgufNetwork::getStateBytes() const {
    uint64_t bytes = 0;
    for (const auto& state : recurrentStates)
        bytes += state->getBufferMemSize();
    for (const auto& cache : caches)
        bytes += cache->getBufferMemSize();
    return bytes;
}

void GgufNetwork::append(const std::string& name, const ComputeGraphElementPtr& operation,
                         const std::vector<ComputeGraphElementPtr>& slots) {
    operation->setName(name);
    for (size_t i = 0; i < slots.size(); ++i)
        operation->setInput(slots[i], int(i));
    // The layers run in the order they are appended; scratch tensors are
    // reused across layers, so every operation follows the previous one.
    if (!operations.empty())
        operation->addDependency(operations.back());
    operations.push_back(operation);
}

ComputeGraphElementPtr GgufNetwork::matVec(const std::string& weight, uint32_t inputWidth,
                                           const std::shared_ptr<FloatTensor>& input,
                                           const std::shared_ptr<FloatTensor>& output, bool accumulate,
                                           bool lastTokenOnly) {
    const auto& info = file->getTensor(weight);
    if (info.rowLength() != inputWidth) {
        throw std::runtime_error("GgufNetwork: tensor " + weight + " does not take " + std::to_string(inputWidth) +
                                 " inputs");
    }
    const uint32_t rows = uint32_t(info.rowCount());
    auto operation =
        gl::matVec(*vulkanContext, info.type, rows, inputWidth, inputWidth, rows, accumulate, lastTokenOnly);
    append(weight, operation, {encodedWeight(weight), input, params, output});
    return operation;
}

void GgufNetwork::buildQwen35() {
    const auto& c = config;
    const uint32_t T = options.maxTokens;

    params = vulkanContext->create<WordTensor>(std::vector<uint32_t>{4});
    hidden = deviceTensor(size_t(T) * c.hidden, true);
    logits = deviceTensor(size_t(options.logitsForAllTokens ? T : 1) * c.vocabulary, true);

    auto normed = deviceTensor(size_t(T) * c.hidden);
    auto ffnGate = deviceTensor(size_t(T) * c.feedForward);
    auto ffnUp = deviceTensor(size_t(T) * c.feedForward);
    auto ffnHidden = deviceTensor(size_t(T) * c.feedForward);

    gl::DeltaNetShape delta{c.keyHeads, c.valueHeads, c.keyDim, c.valueDim, c.epsilon};
    gl::AttentionShape attention{c.headDim,  c.heads,  c.kvHeads, c.ropeDims, options.contextLength,
                                 c.ropeBase, c.epsilon};
    const uint32_t valueWidth = c.valueHeads * c.valueDim;
    std::shared_ptr<FloatTensor> mixed, gate, alpha, beta, conv, deltaOut;
    std::shared_ptr<FloatTensor> queryGate, keys, values, queries, attended;
    for (uint32_t layer = 0; layer < c.layers; ++layer) {
        const std::string block = "blk." + std::to_string(layer) + ".";
        append("rms_norm", gl::rmsNorm(*vulkanContext, c.hidden, 1, c.epsilon, T),
               {hidden, floatWeight(block + "attn_norm.weight"), params, normed});
        if (c.recurrent[layer]) {
            if (!mixed) {
                mixed = deviceTensor(size_t(T) * c.convChannels());
                gate = deviceTensor(size_t(T) * valueWidth);
                alpha = deviceTensor(size_t(T) * c.valueHeads);
                beta = deviceTensor(size_t(T) * c.valueHeads);
                conv = deviceTensor(size_t(T) * c.convChannels());
                deltaOut = deviceTensor(size_t(T) * valueWidth);
            }
            auto convState = deviceTensor(size_t(3) * c.convChannels());
            auto deltaState = deviceTensor(size_t(c.valueHeads) * c.keyDim * c.valueDim);
            recurrentStates.push_back(convState);
            recurrentStates.push_back(deltaState);
            matVec(block + "attn_qkv.weight", c.hidden, normed, mixed);
            matVec(block + "attn_gate.weight", c.hidden, normed, gate);
            matVec(block + "ssm_alpha.weight", c.hidden, normed, alpha);
            matVec(block + "ssm_beta.weight", c.hidden, normed, beta);
            append("linear_conv", gl::linearConv(*vulkanContext, c.convChannels()),
                   {mixed, floatWeight(block + "ssm_conv1d.weight"), params, convState, conv});
            append("gated_delta", gl::gatedDeltaNet(*vulkanContext, delta),
                   {conv, gate, alpha, beta, floatWeight(block + "ssm_a"), floatWeight(block + "ssm_dt.bias"),
                    floatWeight(block + "ssm_norm.weight"), params, deltaState, deltaOut});
            matVec(block + "ssm_out.weight", valueWidth, deltaOut, hidden, true);
        } else {
            if (!queryGate) {
                queryGate = deviceTensor(size_t(T) * c.heads * 2 * c.headDim);
                keys = deviceTensor(size_t(T) * c.kvHeads * c.headDim);
                values = deviceTensor(size_t(T) * c.kvHeads * c.headDim);
                queries = deviceTensor(size_t(T) * c.heads * c.headDim);
                attended = deviceTensor(size_t(T) * c.heads * c.headDim);
            }
            auto keyCache = deviceTensor(size_t(c.kvHeads) * options.contextLength * c.headDim);
            auto valueCache = deviceTensor(size_t(c.kvHeads) * options.contextLength * c.headDim);
            caches.push_back(keyCache);
            caches.push_back(valueCache);
            matVec(block + "attn_q.weight", c.hidden, normed, queryGate);
            matVec(block + "attn_k.weight", c.hidden, normed, keys);
            matVec(block + "attn_v.weight", c.hidden, normed, values);
            append("attention_prep", gl::attentionPrep(*vulkanContext, attention, T),
                   {queryGate, keys, values, floatWeight(block + "attn_q_norm.weight"),
                    floatWeight(block + "attn_k_norm.weight"), params, queries, keyCache, valueCache});
            append("attention", gl::attention(*vulkanContext, attention, T),
                   {queries, keyCache, valueCache, queryGate, params, attended});
            matVec(block + "attn_output.weight", c.heads * c.headDim, attended, hidden, true);
        }
        append("rms_norm", gl::rmsNorm(*vulkanContext, c.hidden, 1, c.epsilon, T),
               {hidden, floatWeight(block + "post_attention_norm.weight"), params, normed});
        matVec(block + "ffn_gate.weight", c.hidden, normed, ffnGate);
        matVec(block + "ffn_up.weight", c.hidden, normed, ffnUp);
        append("swiglu", gl::swiGlu(*vulkanContext, c.feedForward, T), {ffnGate, ffnUp, params, ffnHidden});
        matVec(block + "ffn_down.weight", c.feedForward, ffnHidden, hidden, true);
    }
    append("rms_norm", gl::rmsNorm(*vulkanContext, c.hidden, 1, c.epsilon, T),
           {hidden, floatWeight("output_norm.weight"), params, normed});
    const std::string head = file->hasTensor("output.weight") ? "output.weight" : "token_embd.weight";
    matVec(head, c.hidden, normed, logits, false, !options.logitsForAllTokens);

    inputs[0] = logits;
    srcOutputSlots[0] = -1;
    outputElements[0] = operations.back();
}

void GgufNetwork::_setup(VulkanContext& vulkanContext, uint32_t numberPaths) {
    BatchedUpload batch(vulkanContext, getSetupQueue());
    std::vector<uint8_t> packed;
    for (const auto& [name, weight] : encodedWeights) {
        // Repacked into the GPU block layout in slices of rows, so no copy of
        // a whole large tensor is held.
        const auto& info = file->getTensor(name);
        const uint32_t columns = uint32_t(info.rowLength()), rows = uint32_t(info.rowCount());
        const uint64_t sourceRow = ggmlRowBytes(info.type, columns), gpuRow = gl::gpuRowBytes(info.type, columns);
        const uint32_t sliceRows = uint32_t(std::max<uint64_t>(1, (uint64_t(64) << 20) / gpuRow));
        for (uint32_t first = 0; first < rows; first += sliceRows) {
            const uint32_t count = std::min(sliceRows, rows - first);
            packed.resize(size_t(gpuRow * count));
            gl::packWeights(info.type, file->data(info) + first * sourceRow, count, columns, packed.data());
            batch.add(weight->getDataBuffer().getBuffer(), packed.data(), packed.size(), first * gpuRow);
        }
    }
    std::vector<float> values;
    for (const auto& [name, weight] : floatWeights) {
        const auto& info = file->getTensor(name);
        values.resize(size_t(info.elementCount()));
        for (uint64_t row = 0; row < info.rowCount(); ++row) {
            const uint64_t rowBytes = ggmlRowBytes(info.type, info.rowLength());
            dequantizeRow(info.type, file->data(info) + row * rowBytes, values.data() + row * info.rowLength(),
                          size_t(info.rowLength()));
        }
        weight->getDataBuffer().memcopyFrom(batch, values.data(), values.size());
    }
    batch.submit();
    resetState();
    std::cout << "GgufNetwork: " << operations.size() << " operations, " << (weightBytes >> 20) << " MiB weights, "
              << (getStateBytes() >> 20) << " MiB caches and states" << std::endl;
}

void GgufNetwork::resetState() {
    for (const auto& state : recurrentStates)
        state->getDataBuffer().zero(getSetupQueue());
}

void GgufNetwork::setTokens(const std::vector<int32_t>& tokens, uint32_t position) {
    if (tokens.empty() || tokens.size() > options.maxTokens) {
        throw std::runtime_error("GgufNetwork: between 1 and " + std::to_string(options.maxTokens) +
                                 " tokens per submission");
    }
    if (uint64_t(position) + tokens.size() > options.contextLength) {
        throw std::runtime_error("GgufNetwork: the context of " + std::to_string(options.contextLength) +
                                 " tokens is full");
    }
    const auto& embedding = file->getTensor("token_embd.weight");
    const uint64_t rowBytes = ggmlRowBytes(embedding.type, embedding.rowLength());
    std::vector<float> rows(tokens.size() * config.hidden);
    for (size_t t = 0; t < tokens.size(); ++t) {
        if (tokens[t] < 0 || uint32_t(tokens[t]) >= config.vocabulary) {
            throw std::runtime_error("GgufNetwork: token id " + std::to_string(tokens[t]) + " is out of range");
        }
        dequantizeRow(embedding.type, file->data(embedding) + uint64_t(tokens[t]) * rowBytes,
                      rows.data() + t * config.hidden, config.hidden);
    }
    hidden->getDataBuffer().memcopyFrom(rows.data(), rows.size());
    const std::vector<uint32_t> values{uint32_t(tokens.size()), position, 0, 0};
    params->getDataBuffer().memcopyFrom(values);
    lastTokenCount = uint32_t(tokens.size());
}

std::vector<float> GgufNetwork::readLogits() {
    std::vector<float> values(size_t(options.logitsForAllTokens ? lastTokenCount : 1) * config.vocabulary);
    logits->getDataBuffer().memcopyTo(values);
    return values;
}

} // namespace klartraum
