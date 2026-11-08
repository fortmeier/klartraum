// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_DIFFUSION_DIT_LAYERS_HPP
#define KLARTRAUM_DIFFUSION_DIT_LAYERS_HPP

#include <cstdint>
#include <vector>

#include "klartraum/computegraph/computegraphelement.hpp"
#include "klartraum/vulkan_context.hpp"

/**
 * @brief Compute-graph layers of diffusion transformers such as Qwen-Image (`shaders/dit/`), next to the GGUF
 * layers (gguf_layers: matMul(), fullAttention(), rmsNorm()). Activations are float tensors of [tokens][width].
 * Slots: inputs at 0, 1, ... in the listed order, then outputs.
 */
namespace klartraum::dit_layers {

/**
 * @brief LayerNorm without affine parameters of every token's row of @p width values, modulated by shift and scale
 * from the modulation vector: out = norm(x) (1 + scale) + shift, scale and shift @p width values each from
 * @p scaleOffset and @p shiftOffset.
 *
 * Slots: 0 input, 1 modulation, 2 output.
 */
ComputeGraphElementPtr layerNormModulate(VulkanContext& vulkanContext, uint32_t width, uint32_t tokens,
                                         uint32_t shiftOffset, uint32_t scaleOffset, float epsilon = 1e-6f);

/**
 * @brief x += gate * y for every token's row of @p width values, the gate @p width values of the modulation vector
 * from @p gateOffset.
 *
 * Slots: 0 y, 1 modulation, 2 x (added to).
 */
ComputeGraphElementPtr gatedResidual(VulkanContext& vulkanContext, uint32_t width, uint32_t tokens,
                                     uint32_t gateOffset);

/** @brief The activation of an activation(). */
enum class Activation { GeluTanh, Silu };

/** @brief An activation of @p count values, in place. Slots: 0 values. */
ComputeGraphElementPtr activation(VulkanContext& vulkanContext, Activation kind, uint32_t count);

/**
 * @brief Rotary position embedding of queries or keys [tokens][heads][128], in place: value pairs (2i, 2i + 1) turned
 * by the angles whose cosines and sines the tables hold, 64 per token.
 *
 * Slots: 0 cosines [tokens][64], 1 sines [tokens][64], 2 queries or keys (turned in place).
 */
ComputeGraphElementPtr rotary(VulkanContext& vulkanContext, uint32_t tokens, uint32_t heads);

} // namespace klartraum::dit_layers

#endif // KLARTRAUM_DIFFUSION_DIT_LAYERS_HPP
