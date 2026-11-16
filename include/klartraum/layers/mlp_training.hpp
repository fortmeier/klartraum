// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_LAYERS_MLP_TRAINING_HPP
#define KLARTRAUM_LAYERS_MLP_TRAINING_HPP

#include <cstdint>
#include <memory>
#include <vector>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/layers/training.hpp"
#include "klartraum/vulkan_context.hpp"

namespace klartraum::training {

/** @brief The network and optimizer of an MlpTraining. */
struct MlpOptions {
    /// Widths from the input to the output, e.g. {2, 64, 64, 1}: a fully connected layer between neighbours.
    std::vector<uint32_t> layerSizes;
    uint32_t batch = 1;                   ///< samples per training step
    Activation hidden = Activation::Relu; ///< activation after every layer but the last
    AdamParameters adam;                  ///< optimizer of all weights and biases
    uint32_t seed = 1;                    ///< seed of the weight initialization
};

/**
 * @brief Trains a fully connected network with a sigmoid output on the mean squared error, on the GPU.
 *
 * One training step is one pre-recorded compute graph: the forward pass with layers::matMul(), a bias Add and
 * the activation layers, the loss gradient, the backward pass (activation gradients by automatic differentiation
 * in Slang, the linear layers' gradients as products of transposes) and an Adam step per weight and bias tensor.
 * Weights start uniform in +-sqrt(6 / (inputs + outputs)) (Glorot), biases at zero.
 */
class MlpTraining {
public:
    MlpTraining(VulkanContext& vulkanContext, const MlpOptions& options);

    /** @brief Sets the batch's inputs, batch x layerSizes.front() values, sample-major. */
    void setInputs(const std::vector<float>& inputs);

    /** @brief Sets the batch's targets in [0, 1], batch x layerSizes.back() values. */
    void setTargets(const std::vector<float>& targets);

    /** @brief Runs one training step on the current batch and waits for it. */
    void step();

    /** @brief The mean squared error of the last step's outputs (before its parameter update). */
    float loss() const;

    /** @brief The last step's outputs, sigmoid of the logits, batch x layerSizes.back() values. */
    std::vector<float> outputs() const;

    /** @brief The weights (inputs x outputs, row-major) or bias of layer @p layer, counted from 0. */
    std::vector<float> parameters(size_t layer, bool bias) const;

    /** @brief Replaces the weights or bias of layer @p layer, e.g. to check gradients. */
    void setParameters(size_t layer, bool bias, const std::vector<float>& values);

    /**
     * @brief The last step's gradient of the loss with respect to the weights or bias of layer @p layer.
     *
     * With a learning rate of zero, step() computes the loss and gradients without changing the parameters.
     */
    std::vector<float> gradients(size_t layer, bool bias) const;

private:
    using Tensor = TensorElement<float>;

    // The tensors of one fully connected layer and the elements that write them. The forward elements write
    // product (matMul), preActivation (addBias) and activation (activate, hidden layers only); the backward
    // elements the input's transpose, the weight and bias gradients, the transposed weights and the input
    // gradient (not for the first layer), and preActivationGradient (activationGradient, hidden layers only).
    struct Layer {
        uint32_t inputs = 0, outputs = 0;
        std::shared_ptr<Tensor> weights, bias, product, preActivation, activation;
        std::shared_ptr<Tensor> transposedInput, weightGradient, biasGradient, transposedWeightTensor, inputGradient;
        std::shared_ptr<Tensor> preActivationGradient;
        std::shared_ptr<Tensor> weightMoments[2], biasMoments[2];
        ComputeGraphElementPtr matMul, addBias, activate;
        ComputeGraphElementPtr transposeInput, weightGradientProduct, biasGradientSum, transposeWeights;
        ComputeGraphElementPtr inputGradientProduct, activationGradient;
    };

    std::vector<float> read(const std::shared_ptr<Tensor>& tensor) const;

    // The context the graphs run on; it outlives this object.
    VulkanContext& vulkanContext;
    MlpOptions options;
    std::vector<Layer> layers;
    // The batch's inputs and targets.
    std::shared_ptr<Tensor> inputs, targets;
    // The logits' gradients and squared errors of the last step.
    std::shared_ptr<Tensor> logitGradients, squaredErrors;
    // The step number of the Adam updates, incremented by the graph.
    std::shared_ptr<BufferElement<VulkanBuffer<uint32_t>>> stepCount;
    // The training step, compiled in the constructor.
    std::unique_ptr<ComputeGraph> trainingGraph;
};

} // namespace klartraum::training

#endif // KLARTRAUM_LAYERS_MLP_TRAINING_HPP
