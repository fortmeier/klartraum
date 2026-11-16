// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/layers/training.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"

namespace klartraum::training {

namespace {

struct CountPushConstants {
    uint32_t elementCount;
};

struct ColumnSumPushConstants {
    uint32_t rows, columns;
};

struct FillPushConstants {
    uint32_t count;
    float value;
};

struct ScalarAffinePushConstants {
    uint32_t count;
    float scale, shift;
    uint32_t backward, accumulate;
};

struct ChannelCopyPushConstants {
    uint32_t batch, pixels, sourceChannels, sourceOffset, targetChannels, targetOffset, count, accumulate;
};

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
    const char* shader = activation == Activation::Relu      ? "shaders/training/relu_backward.comp.spv"
                         : activation == Activation::Sigmoid ? "shaders/training/sigmoid_backward.comp.spv"
                         : activation == Activation::Elu     ? "shaders/training/elu_backward.comp.spv"
                                                             : "shaders/training/leaky_relu_backward.comp.spv";
    auto operation = vulkanContext.create<GeneralComputation<CountPushConstants>>(shader);
    const uint32_t count = layers::elementCount(shape);
    operation->setPushConstants({{count}});
    operation->setGroupCountX((count + 255) / 256);
    return operation;
}

ComputeGraphElementPtr leakyRelu(VulkanContext& vulkanContext, const layers::Shape& shape) {
    auto operation =
        vulkanContext.create<GeneralComputation<CountPushConstants>>("shaders/training/leaky_relu.comp.spv");
    const uint32_t count = layers::elementCount(shape);
    operation->setPushConstants({{count}});
    operation->setGroupCountX((count + 255) / 256);
    return operation;
}

ComputeGraphElementPtr channelCopy(VulkanContext& vulkanContext, uint32_t batch, uint32_t pixels, ChannelRange source,
                                   ChannelRange target, uint32_t count, bool accumulate) {
    if (source.offset + count > source.channels || target.offset + count > target.channels)
        throw std::invalid_argument("channelCopy: channel range out of the tensor");
    auto operation =
        vulkanContext.create<GeneralComputation<ChannelCopyPushConstants>>("shaders/training/channel_copy.comp.spv");
    operation->setPushConstants(
        {{batch, pixels, source.channels, source.offset, target.channels, target.offset, count, accumulate ? 1u : 0u}});
    operation->setGroupCountX((batch * count * pixels + 255) / 256);
    return operation;
}

ComputeGraphElementPtr fill(VulkanContext& vulkanContext, uint32_t count, float value) {
    auto operation = vulkanContext.create<GeneralComputation<FillPushConstants>>("shaders/training/fill.comp.spv");
    operation->setPushConstants({{count, value}});
    operation->setGroupCountX((count + 255) / 256);
    return operation;
}

ComputeGraphElementPtr sum(VulkanContext& vulkanContext, uint32_t count) {
    auto operation = vulkanContext.create<GeneralComputation<CountPushConstants>>("shaders/training/sum.comp.spv");
    operation->setPushConstants({{count}});
    operation->setGroupCountX(1);
    return operation;
}

ComputeGraphElementPtr affine(VulkanContext& vulkanContext, uint32_t count, float scale, float shift, bool backward,
                              bool accumulate) {
    auto operation =
        vulkanContext.create<GeneralComputation<ScalarAffinePushConstants>>("shaders/training/affine.comp.spv");
    operation->setPushConstants({{count, scale, shift, backward ? 1u : 0u, accumulate ? 1u : 0u}});
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
