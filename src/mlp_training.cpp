// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/layers/mlp_training.hpp"

#include <cmath>
#include <stdexcept>

#include "klartraum/layers/layers.hpp"

namespace klartraum::training {

MlpTraining::MlpTraining(VulkanContext& vulkanContext, const MlpOptions& options)
    : vulkanContext(vulkanContext),
      options(options) {
    const auto& sizes = options.layerSizes;
    if (sizes.size() < 2 || options.batch == 0)
        throw std::runtime_error("MlpTraining needs at least an input and an output width and a batch");
    const uint32_t batch = options.batch;
    auto tensor = [&vulkanContext](const layers::Shape& shape) { return vulkanContext.create<Tensor>(shape); };
    inputs = tensor({batch, sizes.front()});
    targets = tensor({batch, sizes.back()});
    stepCount = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, 1);
    layers.resize(sizes.size() - 1);

    // Forward: product = input x weights, preActivation = product + bias, activation = f(preActivation).
    for (size_t l = 0; l < layers.size(); ++l) {
        Layer& layer = layers[l];
        layer.inputs = sizes[l];
        layer.outputs = sizes[l + 1];
        const layers::Shape outputShape{batch, layer.outputs};
        layer.weights = tensor({layer.inputs, layer.outputs});
        layer.bias = tensor({layer.outputs});
        layer.product = tensor(outputShape);
        layer.preActivation = tensor(outputShape);
        layer.matMul = layers::matMul(vulkanContext, {batch, layer.inputs}, {layer.inputs, layer.outputs}, outputShape);
        if (l == 0)
            layer.matMul->setInput(inputs, 0);
        else
            layer.matMul->setInput(layers[l - 1].activate, 0, 1);
        layer.matMul->setInput(layer.weights, 1);
        layer.matMul->setInput(layer.product, 2);
        layer.addBias = layers::binary(vulkanContext, layers::BinaryOp::Add, outputShape, {layer.outputs}, outputShape);
        layer.addBias->setInput(layer.matMul, 0, 2);
        layer.addBias->setInput(layer.bias, 1);
        layer.addBias->setInput(layer.preActivation, 2);
        if (l + 1 < layers.size()) {
            layer.activation = tensor(outputShape);
            layer.activate = options.hidden == Activation::Relu
                                 ? layers::relu(vulkanContext, outputShape)
                                 : layers::unary(vulkanContext, layers::UnaryOp::Sigmoid, outputShape);
            layer.activate->setInput(layer.addBias, 0, 2);
            layer.activate->setInput(layer.activation, 1);
        }
    }

    // The loss: the last layer's preActivation are the logits.
    logitGradients = tensor({batch, sizes.back()});
    squaredErrors = tensor({batch, sizes.back()});
    auto loss = sigmoidMseLossBackward(vulkanContext, {batch, sizes.back()});
    loss->setInput(layers.back().addBias, 0, 2);
    loss->setInput(targets, 1);
    loss->setInput(logitGradients, 2);
    loss->setInput(squaredErrors, 3);

    // Backward, from the last layer: the gradients of preActivation come from the loss or from the activation's
    // automatic derivative; weightGradient = inputᵀ x that, biasGradient = its column sums, inputGradient =
    // that x weightsᵀ.
    for (size_t l = layers.size(); l-- > 0;) {
        Layer& layer = layers[l];
        const auto [gradientSource, gradientSlot] =
            l + 1 == layers.size() ? std::pair{loss, 2} : std::pair{layer.activationGradient, 2};
        layer.transposedInput = tensor({layer.inputs, batch});
        layer.transposeInput = layers::transpose(vulkanContext, {batch, layer.inputs}, {layer.inputs, batch}, {1, 0});
        if (l == 0)
            layer.transposeInput->setInput(inputs, 0);
        else
            layer.transposeInput->setInput(layers[l - 1].activate, 0, 1);
        layer.transposeInput->setInput(layer.transposedInput, 1);
        layer.weightGradient = tensor({layer.inputs, layer.outputs});
        layer.weightGradientProduct =
            layers::matMul(vulkanContext, {layer.inputs, batch}, {batch, layer.outputs}, {layer.inputs, layer.outputs});
        layer.weightGradientProduct->setInput(layer.transposeInput, 0, 1);
        layer.weightGradientProduct->setInput(gradientSource, 1, gradientSlot);
        layer.weightGradientProduct->setInput(layer.weightGradient, 2);
        layer.biasGradient = tensor({layer.outputs});
        layer.biasGradientSum = columnSum(vulkanContext, batch, layer.outputs);
        layer.biasGradientSum->setInput(gradientSource, 0, gradientSlot);
        layer.biasGradientSum->setInput(layer.biasGradient, 1);
        if (l == 0)
            continue;
        layer.transposedWeightTensor = tensor({layer.outputs, layer.inputs});
        layer.transposeWeights =
            layers::transpose(vulkanContext, {layer.inputs, layer.outputs}, {layer.outputs, layer.inputs}, {1, 0});
        layer.transposeWeights->setInput(layer.weights, 0);
        layer.transposeWeights->setInput(layer.transposedWeightTensor, 1);
        layer.inputGradient = tensor({batch, layer.inputs});
        layer.inputGradientProduct =
            layers::matMul(vulkanContext, {batch, layer.outputs}, {layer.outputs, layer.inputs}, {batch, layer.inputs});
        layer.inputGradientProduct->setInput(gradientSource, 0, gradientSlot);
        layer.inputGradientProduct->setInput(layer.transposeWeights, 1, 1);
        layer.inputGradientProduct->setInput(layer.inputGradient, 2);
        Layer& previous = layers[l - 1];
        previous.preActivationGradient = tensor({batch, layer.inputs});
        previous.activationGradient = activationBackward(vulkanContext, options.hidden, {batch, layer.inputs});
        previous.activationGradient->setInput(previous.addBias, 0, 2);
        previous.activationGradient->setInput(layer.inputGradientProduct, 1, 2);
        previous.activationGradient->setInput(previous.preActivationGradient, 2);
    }

    // Updates: Adam after the step counter and after every read of the tensor it changes, one after another.
    auto counter = stepCounter(vulkanContext);
    counter->setInput(stepCount, 0);
    ComputeGraphElementPtr last = counter;
    auto update = [&](const std::shared_ptr<Tensor>& parameters, const ComputeGraphElementPtr& gradient,
                      int gradientSlot, std::shared_ptr<Tensor>(&moments)[2],
                      std::initializer_list<ComputeGraphElementPtr> readers) {
        moments[0] = tensor({layers::elementCount(parameters->getDimensions())});
        moments[1] = tensor({layers::elementCount(parameters->getDimensions())});
        auto adam = adamStep(vulkanContext, layers::elementCount(parameters->getDimensions()), options.adam);
        adam->setInput(parameters, 0);
        adam->setInput(gradient, 1, gradientSlot);
        adam->setInput(moments[0], 2);
        adam->setInput(moments[1], 3);
        adam->setInput(stepCount, 4);
        adam->addDependency(last);
        for (const auto& reader : readers)
            if (reader)
                adam->addDependency(reader);
        last = adam;
    };
    for (auto& layer : layers) {
        update(layer.weights, layer.weightGradientProduct, 2, layer.weightMoments,
               {layer.matMul, layer.transposeWeights});
        update(layer.bias, layer.biasGradientSum, 1, layer.biasMoments, {layer.addBias});
    }
    trainingGraph = std::make_unique<ComputeGraph>(vulkanContext, 1);
    trainingGraph->compileFrom(last);

    // Glorot-uniform weights from a linear congruential generator, zero biases and moments, step 0.
    uint32_t state = options.seed * 747796405u + 2891336453u;
    for (auto& layer : layers) {
        const float limit = std::sqrt(6.0f / float(layer.inputs + layer.outputs));
        std::vector<float> weights(size_t(layer.inputs) * layer.outputs);
        for (auto& w : weights) {
            state = state * 1664525u + 1013904223u;
            w = (float(state >> 8) / float(1u << 24) * 2.0f - 1.0f) * limit;
        }
        layer.weights->getDataBuffer(0).memcopyFrom(weights);
        layer.bias->getDataBuffer(0).memcopyFrom(std::vector<float>(layer.outputs, 0.0f));
        for (auto* moments : {&layer.weightMoments, &layer.biasMoments})
            for (auto& moment : *moments)
                moment->getDataBuffer(0).memcopyFrom(std::vector<float>(moment->getDataElementCount(), 0.0f));
    }
    stepCount->getBuffer(0).memcopyFrom(std::vector<uint32_t>{0});
}

void MlpTraining::setInputs(const std::vector<float>& values) {
    if (values.size() != inputs->getDataElementCount())
        throw std::runtime_error("MlpTraining: wrong number of inputs");
    inputs->getDataBuffer(0).memcopyFrom(values);
}

void MlpTraining::setTargets(const std::vector<float>& values) {
    if (values.size() != targets->getDataElementCount())
        throw std::runtime_error("MlpTraining: wrong number of targets");
    targets->getDataBuffer(0).memcopyFrom(values);
}

void MlpTraining::step() { trainingGraph->submitAndWait(vulkanContext.getGraphicsQueue(), 0); }

std::vector<float> MlpTraining::read(const std::shared_ptr<Tensor>& tensor) const {
    std::vector<float> values(tensor->getDataElementCount());
    tensor->getDataBuffer(0).memcopyTo(values);
    return values;
}

float MlpTraining::loss() const {
    double sum = 0.0;
    const auto errors = read(squaredErrors);
    for (float e : errors)
        sum += e;
    return float(sum / double(errors.size()));
}

std::vector<float> MlpTraining::outputs() const {
    // The output layer's preActivation are the logits.
    auto values = read(layers.back().preActivation);
    for (auto& v : values)
        v = 1.0f / (1.0f + std::exp(-v));
    return values;
}

std::vector<float> MlpTraining::parameters(size_t layer, bool bias) const {
    return read(bias ? layers.at(layer).bias : layers.at(layer).weights);
}

void MlpTraining::setParameters(size_t layer, bool bias, const std::vector<float>& values) {
    auto& tensor = bias ? layers.at(layer).bias : layers.at(layer).weights;
    if (values.size() != tensor->getDataElementCount())
        throw std::runtime_error("MlpTraining: wrong number of parameters");
    tensor->getDataBuffer(0).memcopyFrom(values);
}

std::vector<float> MlpTraining::gradients(size_t layer, bool bias) const {
    return read(bias ? layers.at(layer).biasGradient : layers.at(layer).weightGradient);
}

} // namespace klartraum::training
