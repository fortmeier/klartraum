// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_LAYERS_TRAINING_HPP
#define KLARTRAUM_LAYERS_TRAINING_HPP

#include <cstdint>

#include "klartraum/computegraph/computegraphelement.hpp"
#include "klartraum/layers/layers.hpp"
#include "klartraum/vulkan_context.hpp"

/**
 * @brief Compute-graph elements for training networks built from the layers in layers.hpp.
 *
 * The backward elements of the non-linear layers run shaders whose derivatives come from Slang's automatic
 * differentiation (`bwd_diff` of the `[Differentiable]` functions in `shaders/training/differentiable.slang`);
 * linear layers need no own backward element, their gradients are products of the layers in layers.hpp. A
 * training step is one pre-recorded graph: the forward layers, the backward elements and one adamStep() per
 * parameter tensor, ordered after every read of that tensor with addDependency(). Slots follow the convention
 * of layers.hpp: inputs first, then outputs.
 */
namespace klartraum::training {

/** @brief Element-wise activations with a backward element, matching the forward layers. */
enum class Activation {
    Relu,   ///< layers::relu()
    Sigmoid ///< layers::unary(..., UnaryOp::Sigmoid, ...)
};

/**
 * @brief The input gradient of @p activation: f'(x) * output gradient, element-wise over @p shape.
 *
 * Slots: 0 pre-activation input x, 1 output gradient, 2 input gradient.
 */
ComputeGraphElementPtr activationBackward(VulkanContext& vulkanContext, Activation activation,
                                          const layers::Shape& shape);

/**
 * @brief The mean squared error of sigmoid(logits) against targets: the logits' gradients and each element's
 * squared error (their mean is the loss).
 *
 * Slots: 0 logits, 1 targets, 2 logit gradients, 3 squared errors; all of @p shape.
 */
ComputeGraphElementPtr sigmoidMseLossBackward(VulkanContext& vulkanContext, const layers::Shape& shape);

/**
 * @brief Column sums of a rows x columns matrix, e.g. the gradient of a bias added to every row.
 *
 * Slots: 0 input (rows x columns), 1 sums (columns).
 */
ComputeGraphElementPtr columnSum(VulkanContext& vulkanContext, uint32_t rows, uint32_t columns);

/** @brief Hyperparameters of adamStep(). */
struct AdamParameters {
    float learningRate = 1e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
};

/**
 * @brief One Adam step on @p count parameters, in place.
 *
 * The step number comes from a one-element uint32 counter that stepCounter() increments once per submission;
 * order the counter before the steps with addDependency(). The moments start at zero.
 * Slots: 0 parameters, 1 gradients, 2 first moments, 3 second moments, 4 step counter.
 */
ComputeGraphElementPtr adamStep(VulkanContext& vulkanContext, uint32_t count, const AdamParameters& parameters);

/**
 * @brief Adds one to a one-element uint32 counter, e.g. the step number of adamStep().
 *
 * Slots: 0 counter.
 */
ComputeGraphElementPtr stepCounter(VulkanContext& vulkanContext);

} // namespace klartraum::training

#endif // KLARTRAUM_LAYERS_TRAINING_HPP
