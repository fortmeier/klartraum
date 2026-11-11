// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/layers/training.hpp"

#include <array>
#include <stdexcept>

#include "klartraum/computegraph/generalcomputation.hpp"

namespace klartraum::training {

namespace {

struct CountPushConstants {
    uint32_t elementCount;
};

struct ColumnSumPushConstants {
    uint32_t rows, columns;
};

// The geometry of a 2D convolution, for its backward shaders.
struct ConvBackwardPushConstants {
    uint32_t batch, inputChannels, inputHeight, inputWidth;
    uint32_t outputChannels, outputHeight, outputWidth;
    uint32_t kernelHeight, kernelWidth, strideY, strideX, padTop, padLeft;
};

ConvBackwardPushConstants convGeometry(const layers::ConvAttributes& attributes, const layers::Shape& input,
                                       const layers::Shape& weights, const layers::Shape& output) {
    if (input.size() != 4 || weights.size() != 4 || output.size() != 4 || attributes.group != 1 ||
        attributes.dilations != std::array<uint32_t, 2>{1, 1} || weights[1] != input[1] || weights[0] != output[1])
        throw std::invalid_argument("conv backward: needs an ungrouped, undilated NCHW convolution");
    return {input[0],
            input[1],
            input[2],
            input[3],
            output[1],
            output[2],
            output[3],
            weights[2],
            weights[3],
            attributes.strides[0],
            attributes.strides[1],
            attributes.pads[0],
            attributes.pads[1]};
}

struct AdamPushConstants {
    uint32_t elementCount;
    float learningRate, beta1, beta2, epsilon;
    uint32_t decaySteps;
    float finalFactor;
    uint32_t cosine;
};

} // namespace

ComputeGraphElementPtr activationBackward(VulkanContext& vulkanContext, Activation activation,
                                          const layers::Shape& shape) {
    const char* shader = activation == Activation::Relu ? "shaders/training/relu_backward.comp.spv"
                                                        : "shaders/training/sigmoid_backward.comp.spv";
    auto operation = vulkanContext.create<GeneralComputation<CountPushConstants>>(shader);
    const uint32_t count = layers::elementCount(shape);
    operation->setPushConstants({{count}});
    operation->setGroupCountX((count + 255) / 256);
    return operation;
}

ComputeGraphElementPtr sigmoidMseLossBackward(VulkanContext& vulkanContext, const layers::Shape& shape) {
    auto operation =
        vulkanContext.create<GeneralComputation<CountPushConstants>>("shaders/training/sigmoid_mse_backward.comp.spv");
    const uint32_t count = layers::elementCount(shape);
    operation->setPushConstants({{count}});
    operation->setGroupCountX((count + 255) / 256);
    return operation;
}

ComputeGraphElementPtr columnSum(VulkanContext& vulkanContext, uint32_t rows, uint32_t columns) {
    auto operation =
        vulkanContext.create<GeneralComputation<ColumnSumPushConstants>>("shaders/training/column_sum.comp.spv");
    operation->setPushConstants({{rows, columns}});
    operation->setGroupCountX((columns + 63) / 64);
    return operation;
}

ComputeGraphElementPtr convBackwardInput(VulkanContext& vulkanContext, const layers::ConvAttributes& attributes,
                                         const layers::Shape& input, const layers::Shape& weights,
                                         const layers::Shape& output) {
    const auto constants = convGeometry(attributes, input, weights, output);
    auto operation = vulkanContext.create<GeneralComputation<ConvBackwardPushConstants>>(
        "shaders/training/conv_backward_input.comp.spv");
    operation->setPushConstants({constants});
    operation->setGroupCountX((layers::elementCount(input) + 255) / 256);
    return operation;
}

ComputeGraphElementPtr convBackwardWeights(VulkanContext& vulkanContext, const layers::ConvAttributes& attributes,
                                           const layers::Shape& input, const layers::Shape& weights,
                                           const layers::Shape& output) {
    const auto constants = convGeometry(attributes, input, weights, output);
    auto operation = vulkanContext.create<GeneralComputation<ConvBackwardPushConstants>>(
        "shaders/training/conv_backward_weights.comp.spv");
    operation->setPushConstants({constants});
    // One workgroup per weight, then one per bias.
    operation->setGroupCountX(layers::elementCount(weights) + weights[0]);
    return operation;
}

ComputeGraphElementPtr adamStep(VulkanContext& vulkanContext, uint32_t count, const AdamParameters& parameters) {
    auto operation = vulkanContext.create<GeneralComputation<AdamPushConstants>>("shaders/training/adam_step.comp.spv");
    operation->setPushConstants(
        {{count, parameters.learningRate, parameters.beta1, parameters.beta2, parameters.epsilon, parameters.decaySteps,
          parameters.finalFactor, parameters.cosine ? 1u : 0u}});
    operation->setGroupCountX((count + 255) / 256);
    return operation;
}

ComputeGraphElementPtr stepCounter(VulkanContext& vulkanContext) {
    // Push constants give the dispatch its trailing memory barrier, which orders the Adam steps after it.
    auto operation =
        vulkanContext.create<GeneralComputation<CountPushConstants>>("shaders/training/step_counter.comp.spv");
    operation->setPushConstants({{1}});
    operation->setGroupCountX(1);
    return operation;
}

} // namespace klartraum::training
