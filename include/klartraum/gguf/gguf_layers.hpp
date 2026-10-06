// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_GGUF_GGUF_LAYERS_HPP
#define KLARTRAUM_GGUF_GGUF_LAYERS_HPP

#include <cstdint>

#include "klartraum/computegraph/computegraphelement.hpp"
#include "klartraum/gguf/gguf_file.hpp"
#include "klartraum/vulkan_context.hpp"

/**
 * @brief Compute-graph layers of GGUF language models (`shaders/gguf/`).
 *
 * The layers process a chunk of up to `maxTokens` tokens per dispatch. Every
 * layer binds a params tensor (uint32: [0] number of active tokens, [1]
 * position of the first one), so one recorded graph serves prompt chunks and
 * single-token decoding alike; inactive tokens are neither read nor written.
 * Token-major activations are float tensors of [maxTokens][width].
 *
 * Slots follow the layers convention: inputs at 0, 1, ... in the listed
 * order, then outputs. All layers require 32-wide subgroups.
 */
namespace klartraum::gguf_layers {

/** @brief Most tokens one matVec() dispatch handles (MAX_TOKENS in matvec.glsl). */
constexpr uint32_t kMaxTokens = 16;

/** @brief Bytes of zero padding after GGUF weights in a weight buffer (read by unaligned 8-byte loads). */
constexpr uint32_t kWeightPaddingBytes = 16;

/**
 * @brief y[t] = W x[t] for a `rows` x `columns` weight matrix of @p type.
 *
 * Slots: 0 weights (uint32 tensor holding the GGUF bytes plus
 * kWeightPaddingBytes), 1 input, 2 params, 3 output. Inputs of consecutive
 * tokens are @p inputStride floats apart, outputs @p outputStride.
 * With @p accumulate the products are added to the output in place; with
 * @p lastTokenOnly only the last active token is computed, into output row 0.
 * @throws std::runtime_error For unsupported types, columns not a multiple of
 *         8 (or of the type's block size), or strides not a multiple of 4.
 */
ComputeGraphElementPtr matVec(VulkanContext& vulkanContext, GgmlType type, uint32_t rows, uint32_t columns,
                              uint32_t inputStride, uint32_t outputStride, bool accumulate = false,
                              bool lastTokenOnly = false);

/**
 * @brief RMS normalization of the `rowsPerToken` rows of `width` values of
 * every active token, scaled per element.
 *
 * Slots: 0 input, 1 scale (width), 2 params, 3 output.
 */
ComputeGraphElementPtr rmsNorm(VulkanContext& vulkanContext, uint32_t width, uint32_t rowsPerToken, float epsilon,
                               uint32_t maxTokens);

/** @brief Shape of a gated full-attention layer. */
struct AttentionShape {
    uint32_t headDim = 256;
    uint32_t queryHeads = 24;
    uint32_t kvHeads = 4;
    uint32_t ropeDims = 64;
    uint32_t contextLength = 4096;
    float ropeBase = 1e7f;
    float epsilon = 1e-6f;
};

/**
 * @brief Per-head RMS norm and NEOX RoPE of queries and keys; stores the
 * keys and values in the caches at the tokens' positions.
 *
 * Slots: 0 query/gate projection [T][queryHeads][2][headDim], 1 keys
 * [T][kvHeads][headDim], 2 values, 3 query norm (headDim), 4 key norm,
 * 5 params, 6 queries [T][queryHeads][headDim], 7 key cache
 * [kvHeads][context][headDim], 8 value cache.
 */
ComputeGraphElementPtr attentionPrep(VulkanContext& vulkanContext, const AttentionShape& shape, uint32_t maxTokens);

/**
 * @brief Causal attention against the caches, gated by sigmoid(gate).
 *
 * Slots: 0 queries, 1 key cache, 2 value cache, 3 query/gate projection,
 * 4 params, 5 output [T][queryHeads][headDim].
 */
ComputeGraphElementPtr attention(VulkanContext& vulkanContext, const AttentionShape& shape, uint32_t maxTokens);

/**
 * @brief Causal depthwise convolution (kernel 4) + SiLU with a persistent state.
 *
 * Slots: 0 input [T][channels], 1 weights [channels][4], 2 params,
 * 3 state [3][channels] (read and advanced), 4 output [T][channels].
 */
ComputeGraphElementPtr linearConv(VulkanContext& vulkanContext, uint32_t channels);

/** @brief Shape of a gated delta-net (linear attention) layer. */
struct DeltaNetShape {
    uint32_t keyHeads = 16;
    uint32_t valueHeads = 48;
    uint32_t keyDim = 128;
    uint32_t valueDim = 128;
    float epsilon = 1e-6f;

    uint32_t channels() const { return 2 * keyHeads * keyDim + valueHeads * valueDim; }
};

/**
 * @brief Gated delta rule with persistent state, then the gated RMS norm.
 *
 * Slots: 0 convolution output [T][channels], 1 z [T][valueHeads * valueDim],
 * 2 alpha [T][valueHeads], 3 beta [T][valueHeads], 4 a (valueHeads),
 * 5 dt bias (valueHeads), 6 norm weight (valueDim), 7 params,
 * 8 state [valueHeads][keyDim][valueDim] (read and updated),
 * 9 output [T][valueHeads * valueDim].
 */
ComputeGraphElementPtr gatedDeltaNet(VulkanContext& vulkanContext, const DeltaNetShape& shape);

/**
 * @brief silu(gate) * up.
 *
 * Slots: 0 gate, 1 up, 2 params, 3 output (all [T][width]).
 */
ComputeGraphElementPtr swiGlu(VulkanContext& vulkanContext, uint32_t width, uint32_t maxTokens);

} // namespace klartraum::gguf_layers

#endif // KLARTRAUM_GGUF_GGUF_LAYERS_HPP
