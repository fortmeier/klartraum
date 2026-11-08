// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/diffusion/dit_layers.hpp"

#include <stdexcept>

#include "klartraum/computegraph/generalcomputation.hpp"

namespace klartraum::dit_layers {

namespace {

struct LayerNormPushConstants {
    uint32_t width, shiftOffset, scaleOffset;
    float epsilon;
};

struct GatedResidualPushConstants {
    uint32_t width, count, gateOffset;
};

struct ActivationPushConstants {
    uint32_t count, kind;
};

struct RotaryPushConstants {
    uint32_t tokens, heads;
};

// 256 invocations per workgroup over `count` values, in up to 65535 x n groups.
template <typename Operation>
void setGroups(Operation& operation, uint32_t count) {
    const uint32_t groups = (count + 255) / 256;
    const uint32_t x = groups < 65535u ? groups : 65535u;
    operation.setGroupCount(x, (groups + x - 1) / x, 1);
}

} // namespace

ComputeGraphElementPtr layerNormModulate(VulkanContext& vulkanContext, uint32_t width, uint32_t tokens,
                                         uint32_t shiftOffset, uint32_t scaleOffset, float epsilon) {
    if (width == 0 || tokens == 0)
        throw std::invalid_argument("layerNormModulate: width and tokens must be positive");
    auto operation =
        vulkanContext.create<GeneralComputation<LayerNormPushConstants>>("shaders/dit/layer_norm_modulate.comp.spv");
    operation->setPushConstants({{width, shiftOffset, scaleOffset, epsilon}});
    operation->setGroupCount(tokens, 1, 1);
    return operation;
}

ComputeGraphElementPtr gatedResidual(VulkanContext& vulkanContext, uint32_t width, uint32_t tokens,
                                     uint32_t gateOffset) {
    if (width == 0 || tokens == 0)
        throw std::invalid_argument("gatedResidual: width and tokens must be positive");
    auto operation =
        vulkanContext.create<GeneralComputation<GatedResidualPushConstants>>("shaders/dit/gated_residual.comp.spv");
    operation->setPushConstants({{width, width * tokens, gateOffset}});
    setGroups(*operation, width * tokens);
    return operation;
}

ComputeGraphElementPtr activation(VulkanContext& vulkanContext, Activation kind, uint32_t count) {
    if (count == 0)
        throw std::invalid_argument("activation: the count must be positive");
    auto operation =
        vulkanContext.create<GeneralComputation<ActivationPushConstants>>("shaders/dit/activation.comp.spv");
    operation->setPushConstants({{count, kind == Activation::GeluTanh ? 0u : 1u}});
    setGroups(*operation, count);
    return operation;
}

ComputeGraphElementPtr rotary(VulkanContext& vulkanContext, uint32_t tokens, uint32_t heads) {
    if (tokens == 0 || heads == 0)
        throw std::invalid_argument("rotary: tokens and heads must be positive");
    auto operation = vulkanContext.create<GeneralComputation<RotaryPushConstants>>("shaders/dit/rotary.comp.spv");
    operation->setPushConstants({{tokens, heads}});
    operation->setGroupCount(tokens, heads, 1);
    return operation;
}

} // namespace klartraum::dit_layers
