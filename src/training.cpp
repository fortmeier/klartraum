// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/layers/training.hpp"

#include "klartraum/computegraph/generalcomputation.hpp"

namespace klartraum::training {

namespace {

struct CountPushConstants {
    uint32_t elementCount;
};

struct ColumnSumPushConstants {
    uint32_t rows, columns;
};

struct AdamPushConstants {
    uint32_t elementCount;
    float learningRate, beta1, beta2, epsilon;
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

ComputeGraphElementPtr adamStep(VulkanContext& vulkanContext, uint32_t count, const AdamParameters& parameters) {
    auto operation = vulkanContext.create<GeneralComputation<AdamPushConstants>>("shaders/training/adam_step.comp.spv");
    operation->setPushConstants(
        {{count, parameters.learningRate, parameters.beta1, parameters.beta2, parameters.epsilon}});
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
