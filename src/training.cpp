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

// The geometry of a 2D convolution, for its backward shaders.
struct ConvBackwardPushConstants {
    uint32_t batch, inputChannels, inputHeight, inputWidth;
    uint32_t outputChannels, outputHeight, outputWidth;
    uint32_t kernelHeight, kernelWidth, strideY, strideX, padTop, padLeft;
};

struct ConvWeightsPushConstants {
    ConvBackwardPushConstants geometry;
    uint32_t splits, sliceLength, phase;
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

struct FillPushConstants {
    uint32_t count;
    float value;
};

struct LossPushConstants {
    uint32_t planes, height, width;
    float l1Weight, colourWeight;
};

struct LinearPushConstants {
    uint32_t batch, inputs, outputs;
    float offset;
    uint32_t accumulate;
};

struct EmbeddingPushConstants {
    uint32_t batch, rows, size;
    float decay;
};

struct ClipPushConstants {
    uint32_t count;
    float maxNorm;
};

struct EmaPushConstants {
    uint32_t count;
    float rate;
    uint32_t warmupSteps;
};

struct PoolPushConstants {
    uint32_t planes, inputHeight, inputWidth;
};

struct DistancePushConstants {
    uint32_t count;
    float weight;
    uint32_t accumulate;
};

struct AffinePushConstants {
    uint32_t planes, channels, pixels, backward, accumulate;
    float scale[4], shift[4];
};

struct AugmentPushConstants {
    uint32_t batch, channels, height, width;
};

struct AveragePoolPushConstants {
    uint32_t planes, inputHeight, inputWidth, accumulate;
};

struct SoftplusPushConstants {
    uint32_t offset, count;
    float sign, weight;
};

struct ScalarAffinePushConstants {
    uint32_t count;
    float scale, shift;
    uint32_t backward, accumulate;
};

struct NormPushConstants {
    uint32_t batch, channels, pixels;
    float epsilon;
};

struct UpsamplePushConstants {
    uint32_t planes, inputHeight, inputWidth;
};

struct ChannelCopyPushConstants {
    uint32_t batch, pixels, sourceChannels, sourceOffset, targetChannels, targetOffset, count, accumulate;
};

struct FilmPushConstants {
    uint32_t batch, channels, pixels;
};

// N, C and H x W of a rank-four shape.
std::array<uint32_t, 3> planesOf(const layers::Shape& shape) {
    if (shape.size() != 4)
        throw std::invalid_argument("training: needs an N x C x H x W shape");
    return {shape[0], shape[1], shape[2] * shape[3]};
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

ComputeGraphElementPtr channelNorm(VulkanContext& vulkanContext, const layers::Shape& shape) {
    const auto [n, c, pixels] = planesOf(shape);
    auto operation =
        vulkanContext.create<GeneralComputation<NormPushConstants>>("shaders/training/channel_norm.comp.spv");
    operation->setPushConstants({{n, c, pixels, 1e-5f}});
    operation->setGroupCountX((n * pixels + 255) / 256);
    return operation;
}

ComputeGraphElementPtr channelNormBackwardInput(VulkanContext& vulkanContext, const layers::Shape& shape) {
    const auto [n, c, pixels] = planesOf(shape);
    auto operation = vulkanContext.create<GeneralComputation<NormPushConstants>>(
        "shaders/training/channel_norm_backward_input.comp.spv");
    operation->setPushConstants({{n, c, pixels, 1e-5f}});
    operation->setGroupCountX((n * pixels + 255) / 256);
    return operation;
}

ComputeGraphElementPtr channelNormBackwardParameters(VulkanContext& vulkanContext, const layers::Shape& shape) {
    const auto [n, c, pixels] = planesOf(shape);
    auto operation = vulkanContext.create<GeneralComputation<NormPushConstants>>(
        "shaders/training/channel_norm_backward_parameters.comp.spv");
    operation->setPushConstants({{n, c, pixels, 1e-5f}});
    operation->setGroupCountX(c);
    return operation;
}

ComputeGraphElementPtr upsample2xBackward(VulkanContext& vulkanContext, const layers::Shape& input) {
    if (input.size() != 4)
        throw std::invalid_argument("upsample2xBackward: needs an N x C x H x W shape");
    auto operation = vulkanContext.create<GeneralComputation<UpsamplePushConstants>>(
        "shaders/training/upsample2x_backward.comp.spv");
    operation->setPushConstants({{input[0] * input[1], input[2], input[3]}});
    operation->setGroupCountX((layers::elementCount(input) + 255) / 256);
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

ComputeGraphElementPtr film(VulkanContext& vulkanContext, const layers::Shape& shape) {
    const auto [n, c, pixels] = planesOf(shape);
    auto operation = vulkanContext.create<GeneralComputation<FilmPushConstants>>("shaders/training/film.comp.spv");
    operation->setPushConstants({{n, c, pixels}});
    operation->setGroupCountX((n * c * pixels + 255) / 256);
    return operation;
}

ComputeGraphElementPtr filmBackward(VulkanContext& vulkanContext, const layers::Shape& shape) {
    const auto [n, c, pixels] = planesOf(shape);
    auto operation =
        vulkanContext.create<GeneralComputation<FilmPushConstants>>("shaders/training/film_backward.comp.spv");
    operation->setPushConstants({{n, c, pixels}});
    operation->setGroupCountX(n * c);
    return operation;
}

ComputeGraphElementPtr fill(VulkanContext& vulkanContext, uint32_t count, float value) {
    auto operation = vulkanContext.create<GeneralComputation<FillPushConstants>>("shaders/training/fill.comp.spv");
    operation->setPushConstants({{count, value}});
    operation->setGroupCountX((count + 255) / 256);
    return operation;
}

ComputeGraphElementPtr photometricLoss(VulkanContext& vulkanContext, const layers::Shape& shape, float l1Weight,
                                       float colourWeight) {
    const auto [n, c, pixels] = planesOf(shape);
    auto operation =
        vulkanContext.create<GeneralComputation<LossPushConstants>>("shaders/training/photometric_loss.comp.spv");
    operation->setPushConstants({{n * c, shape[2], shape[3], l1Weight, colourWeight}});
    operation->setGroupCountX((n * c * pixels + 255) / 256);
    return operation;
}

ComputeGraphElementPtr sum(VulkanContext& vulkanContext, uint32_t count) {
    auto operation = vulkanContext.create<GeneralComputation<CountPushConstants>>("shaders/training/sum.comp.spv");
    operation->setPushConstants({{count}});
    operation->setGroupCountX(1);
    return operation;
}

ComputeGraphElementPtr linear(VulkanContext& vulkanContext, const LinearShape& shape, float offset) {
    auto operation = vulkanContext.create<GeneralComputation<LinearPushConstants>>("shaders/training/linear.comp.spv");
    operation->setPushConstants({{shape.batch, shape.inputs, shape.outputs, offset, 0u}});
    operation->setGroupCountX((shape.batch * shape.outputs + 63) / 64);
    return operation;
}

ComputeGraphElementPtr linearBackwardInput(VulkanContext& vulkanContext, const LinearShape& shape, bool accumulate) {
    auto operation = vulkanContext.create<GeneralComputation<LinearPushConstants>>(
        "shaders/training/linear_backward_input.comp.spv");
    operation->setPushConstants({{shape.batch, shape.inputs, shape.outputs, 0.0f, accumulate ? 1u : 0u}});
    operation->setGroupCountX((shape.batch * shape.inputs + 63) / 64);
    return operation;
}

ComputeGraphElementPtr linearBackwardWeights(VulkanContext& vulkanContext, const LinearShape& shape) {
    auto operation = vulkanContext.create<GeneralComputation<LinearPushConstants>>(
        "shaders/training/linear_backward_weights.comp.spv");
    operation->setPushConstants({{shape.batch, shape.inputs, shape.outputs, 0.0f, 0u}});
    operation->setGroupCountX((shape.outputs * (shape.inputs + 1) + 63) / 64);
    return operation;
}

ComputeGraphElementPtr embedding(VulkanContext& vulkanContext, uint32_t batch, uint32_t rows, uint32_t size) {
    auto operation =
        vulkanContext.create<GeneralComputation<EmbeddingPushConstants>>("shaders/training/embedding.comp.spv");
    operation->setPushConstants({{batch, rows, size, 0.0f}});
    operation->setGroupCountX((batch * size + 63) / 64);
    return operation;
}

ComputeGraphElementPtr embeddingBackward(VulkanContext& vulkanContext, uint32_t batch, uint32_t rows, uint32_t size,
                                         float decay) {
    auto operation = vulkanContext.create<GeneralComputation<EmbeddingPushConstants>>(
        "shaders/training/embedding_backward.comp.spv");
    operation->setPushConstants({{batch, rows, size, decay}});
    operation->setGroupCountX((rows * size + 63) / 64);
    return operation;
}

ComputeGraphElementPtr squaredNorm(VulkanContext& vulkanContext, uint32_t count) {
    auto operation =
        vulkanContext.create<GeneralComputation<CountPushConstants>>("shaders/training/squared_norm.comp.spv");
    operation->setPushConstants({{count}});
    operation->setGroupCountX((count + 255) / 256);
    return operation;
}

ComputeGraphElementPtr clipGradient(VulkanContext& vulkanContext, uint32_t count, float maxNorm) {
    auto operation =
        vulkanContext.create<GeneralComputation<ClipPushConstants>>("shaders/training/clip_gradient.comp.spv");
    operation->setPushConstants({{count, maxNorm}});
    operation->setGroupCountX((count + 255) / 256);
    return operation;
}

ComputeGraphElementPtr ema(VulkanContext& vulkanContext, uint32_t count, float rate, uint32_t warmupSteps) {
    auto operation = vulkanContext.create<GeneralComputation<EmaPushConstants>>("shaders/training/ema.comp.spv");
    operation->setPushConstants({{count, rate, warmupSteps}});
    operation->setGroupCountX((count + 255) / 256);
    return operation;
}

ComputeGraphElementPtr maxPool(VulkanContext& vulkanContext, const layers::Shape& input) {
    const auto [n, c, pixels] = planesOf(input);
    auto operation = vulkanContext.create<GeneralComputation<PoolPushConstants>>("shaders/training/max_pool.comp.spv");
    operation->setPushConstants({{n * c, input[2], input[3]}});
    operation->setGroupCountX((n * c * (input[2] / 2) * (input[3] / 2) + 255) / 256);
    return operation;
}

ComputeGraphElementPtr maxPoolBackward(VulkanContext& vulkanContext, const layers::Shape& input) {
    const auto [n, c, pixels] = planesOf(input);
    auto operation =
        vulkanContext.create<GeneralComputation<PoolPushConstants>>("shaders/training/max_pool_backward.comp.spv");
    operation->setPushConstants({{n * c, input[2], input[3]}});
    operation->setGroupCountX((n * c * pixels + 255) / 256);
    return operation;
}

ComputeGraphElementPtr l1Distance(VulkanContext& vulkanContext, uint32_t count, float weight, bool accumulate) {
    auto operation =
        vulkanContext.create<GeneralComputation<DistancePushConstants>>("shaders/training/l1_distance.comp.spv");
    operation->setPushConstants({{count, weight, accumulate ? 1u : 0u}});
    operation->setGroupCountX((count + 255) / 256);
    return operation;
}

ComputeGraphElementPtr channelAffine(VulkanContext& vulkanContext, const layers::Shape& shape,
                                     const std::vector<float>& scale, const std::vector<float>& shift, bool backward,
                                     bool accumulate) {
    const auto [n, c, pixels] = planesOf(shape);
    if (c > 4 || scale.size() != c || shift.size() != c)
        throw std::invalid_argument("channelAffine: needs a scale and shift for each of at most four channels");
    AffinePushConstants constants{n * c, c, pixels, backward ? 1u : 0u, accumulate ? 1u : 0u, {}, {}};
    std::copy(scale.begin(), scale.end(), constants.scale);
    std::copy(shift.begin(), shift.end(), constants.shift);
    auto operation =
        vulkanContext.create<GeneralComputation<AffinePushConstants>>("shaders/training/channel_affine.comp.spv");
    operation->setPushConstants({constants});
    operation->setGroupCountX((n * c * pixels + 255) / 256);
    return operation;
}

ComputeGraphElementPtr augmentPair(VulkanContext& vulkanContext, const layers::Shape& shape) {
    if (shape.size() != 4 || shape[1] < 3)
        throw std::invalid_argument("augmentPair: needs N x C x H x W pairs with an image in the last three channels");
    auto operation =
        vulkanContext.create<GeneralComputation<AugmentPushConstants>>("shaders/training/augment_pair.comp.spv");
    operation->setPushConstants({{shape[0], shape[1], shape[2], shape[3]}});
    operation->setGroupCountX((layers::elementCount(shape) + 255) / 256);
    return operation;
}

ComputeGraphElementPtr augmentPairBackward(VulkanContext& vulkanContext, const layers::Shape& shape) {
    if (shape.size() != 4 || shape[1] < 3)
        throw std::invalid_argument("augmentPairBackward: needs N x C x H x W pairs with an image in the last three "
                                    "channels");
    auto operation = vulkanContext.create<GeneralComputation<AugmentPushConstants>>(
        "shaders/training/augment_pair_backward.comp.spv");
    operation->setPushConstants({{shape[0], shape[1], shape[2], shape[3]}});
    operation->setGroupCountX((layers::elementCount(shape) + 255) / 256);
    return operation;
}

ComputeGraphElementPtr averagePool(VulkanContext& vulkanContext, const layers::Shape& input) {
    const auto [n, c, pixels] = planesOf(input);
    auto operation =
        vulkanContext.create<GeneralComputation<AveragePoolPushConstants>>("shaders/training/average_pool.comp.spv");
    operation->setPushConstants({{n * c, input[2], input[3], 0u}});
    operation->setGroupCountX((n * c * (input[2] / 2) * (input[3] / 2) + 255) / 256);
    return operation;
}

ComputeGraphElementPtr averagePoolBackward(VulkanContext& vulkanContext, const layers::Shape& input, bool accumulate) {
    const auto [n, c, pixels] = planesOf(input);
    auto operation = vulkanContext.create<GeneralComputation<AveragePoolPushConstants>>(
        "shaders/training/average_pool_backward.comp.spv");
    operation->setPushConstants({{n * c, input[2], input[3], accumulate ? 1u : 0u}});
    operation->setGroupCountX((n * c * pixels + 255) / 256);
    return operation;
}

ComputeGraphElementPtr softplusLoss(VulkanContext& vulkanContext, uint32_t offset, uint32_t count, float sign,
                                    float weight) {
    auto operation =
        vulkanContext.create<GeneralComputation<SoftplusPushConstants>>("shaders/training/softplus_loss.comp.spv");
    operation->setPushConstants({{offset, count, sign, weight}});
    operation->setGroupCountX((count + 255) / 256);
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

ComputeGraphElementPtr convBackwardInput(VulkanContext& vulkanContext, const layers::ConvAttributes& attributes,
                                         const layers::Shape& input, const layers::Shape& weights,
                                         const layers::Shape& output) {
    const auto constants = convGeometry(attributes, input, weights, output);
    auto operation = vulkanContext.create<GeneralComputation<ConvBackwardPushConstants>>(
        "shaders/training/conv_backward_input.comp.spv");
    operation->setPushConstants({constants});
    // 64 x 64 tiles of input positions (x) and input channels (y).
    operation->setGroupCount((input[0] * input[2] * input[3] + 63) / 64, (input[1] + 63) / 64, 1);
    return operation;
}

ComputeGraphElementPtr convBackwardWeights(VulkanContext& vulkanContext, const layers::ConvAttributes& attributes,
                                           const layers::Shape& input, const layers::Shape& weights,
                                           const layers::Shape& output) {
    const auto geometry = convGeometry(attributes, input, weights, output);
    // A matrix product of outputs x columns (weights per output channel and the bias) over k = batch x positions:
    // 64 x 64 tiles, and k split into slices so that there are about 512 workgroups.
    const uint32_t columns = weights[1] * weights[2] * weights[3] + 1;
    const uint32_t tilesX = (columns + 63) / 64, tilesY = (weights[0] + 63) / 64;
    const uint32_t total = output[0] * output[2] * output[3];
    const uint32_t wanted = std::max(1u, 512u / (tilesX * tilesY));
    const uint32_t sliceLength = std::max(16u, ((total + wanted - 1) / wanted + 15) / 16 * 16);
    const uint32_t splits = (total + sliceLength - 1) / sliceLength;
    auto operation = vulkanContext.create<GeneralComputation<ConvWeightsPushConstants>>(
        "shaders/training/conv_backward_weights.comp.spv");
    // Phase 0 then phase 1 over the same grid, which covers every result of phase 1 too.
    operation->setPushConstants({{geometry, splits, sliceLength, 0u}, {geometry, splits, sliceLength, 1u}});
    operation->setGroupCount(tilesX, tilesY, splits);
    auto partial =
        std::make_shared<BufferElementSinglePath<VulkanBuffer<float>>>(vulkanContext, splits * weights[0] * columns);
    operation->addScratchBufferElement(partial, false);
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
