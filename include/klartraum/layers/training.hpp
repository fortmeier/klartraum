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
    Relu,     ///< layers::relu()
    Sigmoid,  ///< layers::unary(..., UnaryOp::Sigmoid, ...)
    Elu,      ///< layers::elu() with alpha 1
    LeakyRelu ///< leakyRelu(): slope 0.2 below zero
};

/**
 * @brief The leaky ReLU with slope 0.2 below zero, element-wise over @p shape.
 *
 * Slots: 0 input, 1 output.
 */
ComputeGraphElementPtr leakyRelu(VulkanContext& vulkanContext, const layers::Shape& shape);

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

/**
 * @brief The input gradient of layers::conv() (NCHW input, OIHW weights, ungrouped, undilated): every input element
 * gathers the weights times the output gradients of the outputs it reaches.
 *
 * Slots: 0 output gradient (@p output), 1 weights (@p weights), 2 input gradient (@p input).
 */
ComputeGraphElementPtr convBackwardInput(VulkanContext& vulkanContext, const layers::ConvAttributes& attributes,
                                         const layers::Shape& input, const layers::Shape& weights,
                                         const layers::Shape& output);

/**
 * @brief The weight and bias gradients of layers::conv(): a tiled matrix product of the output gradients with the
 * inputs each weight met (and ones for the bias) over the batch and the output positions, split into slices whose
 * partial sums are then added in a fixed order.
 *
 * Slots: 0 input (@p input), 1 output gradient (@p output), 2 weight gradients (@p weights), 3 bias gradients (O).
 */
ComputeGraphElementPtr convBackwardWeights(VulkanContext& vulkanContext, const layers::ConvAttributes& attributes,
                                           const layers::Shape& input, const layers::Shape& weights,
                                           const layers::Shape& output);

/**
 * @brief ChannelNorm: every pixel of an N x C x H x W tensor normalized over its channels (eps 1e-5), then scaled and
 * shifted per channel; keeps each pixel's mean and 1 / standard deviation for the backward pass.
 *
 * Slots: 0 input, 1 scale (C), 2 shift (C), 3 output, 4 statistics (N x 2 x H x W).
 */
ComputeGraphElementPtr channelNorm(VulkanContext& vulkanContext, const layers::Shape& shape);

/**
 * @brief The input gradient of channelNorm().
 *
 * Slots: 0 input, 1 scale, 2 statistics, 3 output gradient, 4 input gradient.
 */
ComputeGraphElementPtr channelNormBackwardInput(VulkanContext& vulkanContext, const layers::Shape& shape);

/**
 * @brief The scale and shift gradients of channelNorm(), reduced per channel in a fixed order.
 *
 * Slots: 0 input, 1 statistics, 2 output gradient, 3 scale gradient (C), 4 shift gradient (C).
 */
ComputeGraphElementPtr channelNormBackwardParameters(VulkanContext& vulkanContext, const layers::Shape& shape);

/** @brief Where channelCopy() reads and writes: channel ranges of two N x C x H x W tensors. */
struct ChannelRange {
    uint32_t channels = 0; ///< of the whole tensor
    uint32_t offset = 0;   ///< of the first copied channel
};

/**
 * @brief Copies @p count channels from @p source to @p target (or adds them to it, with @p accumulate): a concat
 * along the channels, and its backward pass, adding where a tensor's gradient comes from several consumers.
 *
 * Slots: 0 source, 1 target.
 */
ComputeGraphElementPtr channelCopy(VulkanContext& vulkanContext, uint32_t batch, uint32_t pixels, ChannelRange source,
                                   ChannelRange target, uint32_t count, bool accumulate = false);

/**
 * @brief Sets @p count floats to @p value, e.g. a gradient that elements add to.
 *
 * Slots: 0 buffer.
 */
ComputeGraphElementPtr fill(VulkanContext& vulkanContext, uint32_t count, float value = 0.0f);

/**
 * @brief The sum of @p count floats, in a fixed order.
 *
 * Slots: 0 data, 1 result (one float).
 */
ComputeGraphElementPtr sum(VulkanContext& vulkanContext, uint32_t count);

/** @brief The sizes of a linear layer (nn.Linear): a batch of input vectors to output vectors. */
struct LinearShape {
    uint32_t batch = 1;
    uint32_t inputs = 0;
    uint32_t outputs = 0;
};

/**
 * @brief y = x W^T + bias + @p offset, W as [outputs][inputs] (nn.Linear's layout).
 *
 * Slots: 0 input (batch x inputs), 1 weights, 2 bias (outputs), 3 output (batch x outputs).
 */
ComputeGraphElementPtr linear(VulkanContext& vulkanContext, const LinearShape& shape, float offset = 0.0f);

/**
 * @brief The input gradient of linear(); with @p accumulate added to it (several layers reading one input).
 *
 * Slots: 0 output gradient, 1 weights, 2 input gradient.
 */
ComputeGraphElementPtr linearBackwardInput(VulkanContext& vulkanContext, const LinearShape& shape,
                                           bool accumulate = false);

/**
 * @brief The weight and bias gradients of linear().
 *
 * Slots: 0 input, 1 output gradient, 2 weight gradients, 3 bias gradients.
 */
ComputeGraphElementPtr linearBackwardWeights(VulkanContext& vulkanContext, const LinearShape& shape);

/**
 * @brief y = x * @p scale + @p shift for @p count elements; with @p backward the input gradient dy * @p scale.
 * Written or, with @p accumulate, added.
 *
 * Slots: 0 input (backward: output gradient), 1 output (backward: input gradient).
 */
ComputeGraphElementPtr affine(VulkanContext& vulkanContext, uint32_t count, float scale, float shift,
                              bool backward = false, bool accumulate = false);

/** @brief Hyperparameters of adamStep(). */
struct AdamParameters {
    float learningRate = 1e-3f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    /// Steps over which the learning rate changes to learningRate x finalFactor (it stays there after); 0 for a
    /// constant learning rate.
    uint32_t decaySteps = 0;
    float finalFactor = 1.0f; ///< learning rate after decaySteps, relative to learningRate
    /// How it changes: exponentially (false), or along half a cosine (true): slowly at first and at the end.
    bool cosine = false;
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
