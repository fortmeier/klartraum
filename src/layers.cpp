// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/layers/layers.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/layers/layer_push_constants.hpp"

namespace klartraum::layers {

namespace {

void padDimensions(const Shape& source, uint32_t (&destination)[4]) {
    if (source.size() > 4)
        throw std::runtime_error("tensors with rank above four are unsupported");
    std::fill(std::begin(destination), std::end(destination), 1u);
    const size_t offset = 4 - source.size();
    for (size_t i = 0; i < source.size(); ++i)
        destination[offset + i] = source[i];
}

// Copies the first four dimensions, filling missing ones with 1.
void copyDimensions(const Shape& source, uint32_t (&destination)[4]) {
    for (size_t i = 0; i < 4; ++i)
        destination[i] = i < source.size() ? source[i] : 1;
}

void checkAxis(uint32_t axis, const Shape& shape, const char* layer) {
    if (axis >= shape.size())
        throw std::runtime_error(std::string(layer) + " axis is outside the tensor rank");
}

template <typename PushConstants>
std::shared_ptr<GeneralComputation<PushConstants>> computation(VulkanContext& vulkanContext, const std::string& shader,
                                                               const PushConstants& constants) {
    auto operation = vulkanContext.create<GeneralComputation<PushConstants>>(shader);
    operation->setPushConstants({constants});
    return operation;
}

} // namespace

uint32_t elementCount(const Shape& shape) {
    uint64_t count = 1;
    for (const auto dimension : shape)
        count *= dimension;
    if (count > UINT32_MAX) {
        throw std::runtime_error("Tensor is too large for a Vulkan dispatch");
    }
    return static_cast<uint32_t>(count);
}

Shape broadcastShape(const Shape& lhs, const Shape& rhs) {
    Shape result(std::max(lhs.size(), rhs.size()), 1);
    for (size_t i = 0; i < result.size(); ++i) {
        const uint32_t l = i < result.size() - lhs.size() ? 1 : lhs[i - (result.size() - lhs.size())];
        const uint32_t r = i < result.size() - rhs.size() ? 1 : rhs[i - (result.size() - rhs.size())];
        if (l != r && l != 1 && r != 1)
            throw std::runtime_error("tensor shapes cannot be broadcast together");
        result[i] = std::max(l, r);
    }
    return result;
}

ComputeGraphElementPtr binary(VulkanContext& vulkanContext, BinaryOp op, const Shape& lhs, const Shape& rhs,
                              const Shape& output) {
    const char* shader = op == BinaryOp::Add   ? "shaders/onnx/add.comp.spv"
                         : op == BinaryOp::Sub ? "shaders/onnx/sub.comp.spv"
                         : op == BinaryOp::Mul ? "shaders/onnx/mul.comp.spv"
                                               : "shaders/onnx/div.comp.spv";
    BinaryBroadcastPushConstants constants{};
    constants.rank = static_cast<uint32_t>(output.size());
    constants.elementCount = elementCount(output);
    padDimensions(lhs, constants.lhsDims);
    padDimensions(rhs, constants.rhsDims);
    padDimensions(output, constants.outputDims);
    auto operation = computation(vulkanContext, shader, constants);
    operation->setGroupCountX((constants.elementCount + 63) / 64);
    return operation;
}

ComputeGraphElementPtr unary(VulkanContext& vulkanContext, UnaryOp op, const Shape& shape) {
    const char* shader = op == UnaryOp::Sigmoid ? "shaders/onnx/sigmoid.comp.spv"
                         : op == UnaryOp::Cos   ? "shaders/onnx/cos.comp.spv"
                         : op == UnaryOp::Sin   ? "shaders/onnx/sin.comp.spv"
                         : op == UnaryOp::Sqrt  ? "shaders/onnx/sqrt.comp.spv"
                         : op == UnaryOp::Erf   ? "shaders/onnx/erf.comp.spv"
                                                : "shaders/onnx/cast_int64_float.comp.spv";
    UnaryPushConstants constants{elementCount(shape)};
    auto operation = computation(vulkanContext, shader, constants);
    operation->setGroupCountX((constants.elementCount + 63) / 64);
    return operation;
}

ComputeGraphElementPtr relu(VulkanContext& vulkanContext, const Shape& shape) {
    TensorOpPushConstants constants;
    copyDimensions(shape, constants.dimInput);
    copyDimensions(shape, constants.dimOutput);
    auto operation = computation(vulkanContext, "shaders/onnx/relu.comp.spv", constants);
    operation->setGroupCountX((elementCount(shape) + 63) / 64);
    return operation;
}

ComputeGraphElementPtr conv(VulkanContext& vulkanContext, const ConvAttributes& attributes, const Shape& input,
                            const Shape& weights, const Shape& bias, const Shape& output) {
    if (output.size() != 4)
        throw std::runtime_error("Conv requires a rank-four output");
    ConvPushConstants constants;
    std::copy(attributes.dilations.begin(), attributes.dilations.end(), constants.dilations);
    constants.groups[0] = attributes.group;
    std::copy(attributes.kernelShape.begin(), attributes.kernelShape.end(), constants.kernel_shape);
    std::copy(attributes.pads.begin(), attributes.pads.end(), constants.pads);
    std::copy(attributes.strides.begin(), attributes.strides.end(), constants.strides);
    copyDimensions(input, constants.dimInput);
    copyDimensions(weights, constants.dimWeights);
    constants.dimBias[0] = bias.empty() ? 1 : bias[0];
    for (size_t i = 0; i < 4; ++i)
        constants.dimOutput[i] = output[i];

    const bool unitStride = constants.strides[0] == 1 && constants.strides[1] == 1 && constants.dilations[0] == 1 &&
                            constants.dilations[1] == 1 && constants.groups[0] == 1;
    const bool specialized3x3 =
        unitStride && constants.kernel_shape[0] == 3 && constants.kernel_shape[1] == 3 &&
        std::all_of(std::begin(constants.pads), std::end(constants.pads), [](uint32_t pad) { return pad == 1; });
    const bool specialized1x1 =
        unitStride && constants.kernel_shape[0] == 1 && constants.kernel_shape[1] == 1 &&
        std::all_of(std::begin(constants.pads), std::end(constants.pads), [](uint32_t pad) { return pad == 0; });
    const std::string shader = specialized3x3   ? "shaders/onnx/conv_3x3.comp.spv"
                               : specialized1x1 ? "shaders/onnx/conv_1x1.comp.spv"
                                                : "shaders/onnx/conv.comp.spv";
    auto operation = computation(vulkanContext, shader, constants);
    // One invocation per output element; the output shape also covers
    // asymmetric padding.
    if (specialized1x1) {
        operation->setGroupCountX((output[2] * output[3] + 15) / 16);
        operation->setGroupCountY((output[1] + 15) / 16);
        operation->setGroupCountZ(output[0]);
    } else {
        operation->setGroupCountX((output[3] + 7) / 8);
        operation->setGroupCountY((output[2] + 7) / 8);
        operation->setGroupCountZ(output[0] * (specialized3x3 ? (output[1] + 3) / 4 : output[1]));
    }
    return operation;
}

ComputeGraphElementPtr convTranspose(VulkanContext& vulkanContext, const ConvAttributes& attributes, const Shape& input,
                                     const Shape& weights, const Shape& bias) {
    if (input.size() != 4 || weights.size() < 2)
        throw std::runtime_error("ConvTranspose requires NCHW input");
    // The shader reads up to eight entries per attribute; entries past the
    // attribute's own are 1, as for a missing ONNX attribute entry.
    ConvTransposePushConstants constants;
    auto fill = [](const auto& values, uint32_t (&destination)[8]) {
        std::fill(std::begin(destination), std::end(destination), 1u);
        std::copy(values.begin(), values.end(), destination);
    };
    fill(attributes.dilations, constants.dilations);
    std::fill(std::begin(constants.groups), std::end(constants.groups), attributes.group);
    fill(attributes.kernelShape, constants.kernel_shape);
    fill(attributes.pads, constants.pads);
    fill(attributes.strides, constants.strides);
    for (size_t i = 0; i < 4; ++i) {
        constants.dimInput[i] = input[i];
        constants.dimWeights[i] = i < weights.size() ? weights[i] : 1;
    }
    constants.dimBias[0] = bias.empty() ? 1 : bias[0];
    // output = (input - 1) * stride - 2 * pad + kernel
    constants.dimOutput[0] = input[0];
    constants.dimOutput[1] = weights[1];
    constants.dimOutput[2] = (input[2] - 1) * constants.strides[0] - 2 * constants.pads[0] + constants.kernel_shape[0];
    constants.dimOutput[3] = (input[3] - 1) * constants.strides[1] - 2 * constants.pads[1] + constants.kernel_shape[1];

    auto operation = computation(vulkanContext, "shaders/onnx/conv_transpose.comp.spv", constants);
    // One invocation per output element (local size 8x8x1).
    operation->setGroupCountX((constants.dimOutput[3] + 7) / 8);
    operation->setGroupCountY((constants.dimOutput[2] + 7) / 8);
    operation->setGroupCountZ(constants.dimOutput[1]);
    return operation;
}

ComputeGraphElementPtr instanceNormalization(VulkanContext& vulkanContext, const Shape& input, float epsilon) {
    if (input.size() < 3)
        throw std::runtime_error("InstanceNormalization requires rank >= 3");
    InstanceNormalizationPushConstants constants{};
    constants.batch = input[0];
    constants.channels = input[1];
    constants.spatialSize = 1;
    for (size_t i = 2; i < input.size(); ++i)
        constants.spatialSize *= input[i];
    constants.epsilon = epsilon;
    auto operation = computation(vulkanContext, "shaders/onnx/instance_normalization.comp.spv", constants);
    operation->setGroupCountX(constants.batch * constants.channels);
    return operation;
}

ComputeGraphElementPtr layerNormalization(VulkanContext& vulkanContext, const Shape& input, float epsilon) {
    if (input.empty())
        throw std::runtime_error("LayerNormalization requires rank >= 1");
    LayerNormalizationPushConstants constants{};
    constants.axisSize = input.back();
    constants.outerCount = elementCount(input) / constants.axisSize;
    constants.epsilon = epsilon;
    auto operation = computation(vulkanContext, "shaders/onnx/layer_normalization.comp.spv", constants);
    operation->setGroupCountX((constants.outerCount + 63) / 64);
    return operation;
}

ComputeGraphElementPtr matMul(VulkanContext& vulkanContext, const Shape& lhs, const Shape& rhs, const Shape& output) {
    if (lhs.size() < 2 || rhs.size() < 2 || output.size() < 2) {
        throw std::runtime_error("MatMul requires rank >= 2");
    }
    MatMulPushConstants constants{};
    constants.rows = output[output.size() - 2];
    constants.columns = output.back();
    constants.reduction = lhs.back();
    constants.batchCount = elementCount(output) / (constants.rows * constants.columns);
    constants.lhsBatchCount = elementCount(lhs) / (constants.rows * constants.reduction);
    constants.rhsBatchCount = elementCount(rhs) / (constants.reduction * constants.columns);
    if (constants.lhsBatchCount != 1 && constants.lhsBatchCount != constants.batchCount) {
        throw std::runtime_error("Unsupported MatMul left batch broadcasting");
    }
    if (constants.rhsBatchCount != 1 && constants.rhsBatchCount != constants.batchCount) {
        throw std::runtime_error("Unsupported MatMul right batch broadcasting");
    }
    auto operation = computation(vulkanContext, "shaders/onnx/matmul.comp.spv", constants);
    operation->setGroupCount((constants.columns + 31) / 32, (constants.rows + 15) / 16, constants.batchCount);
    return operation;
}

ComputeGraphElementPtr gemm(VulkanContext& vulkanContext, const Shape& input, const Shape& weights) {
    if (input.size() != 2 || weights.size() != 2)
        throw std::runtime_error("Gemm requires rank-two tensors");
    GemmPushConstants constants{input[0], weights[0], input[1]};
    auto operation = computation(vulkanContext, "shaders/onnx/gemm.comp.spv", constants);
    operation->setGroupCount((constants.columns + 7) / 8, (constants.rows + 7) / 8, 1);
    return operation;
}

ComputeGraphElementPtr fusedAttention(VulkanContext& vulkanContext, const Shape& query, const Shape& key,
                                      const Shape& value, const Shape& output) {
    if (query.size() != 4 || key.size() != 4 || value.size() != 4 || output.size() != 4) {
        throw std::runtime_error("FusedAttention requires rank-four tensors");
    }
    FusedAttentionPushConstants constants{query[0] * query[1], query[2], key[3], query[3], value[3]};
    if (key[0] * key[1] != constants.batchCount || value[0] * value[1] != constants.batchCount ||
        key[2] != constants.queryDepth || value[2] != constants.keyCount ||
        output[0] * output[1] != constants.batchCount || output[2] != constants.queryCount ||
        output[3] != constants.valueDepth || constants.queryDepth > 512 || constants.valueDepth > 512) {
        throw std::runtime_error("Unsupported FusedAttention tensor shapes");
    }
    // Head widths 40 and 80 (the SD1.5 UNet's two highest-resolution
    // attention levels) use the key-tiled kernel with one query per invocation
    // and 64 queries per group.
    if (constants.queryDepth == constants.valueDepth && (constants.queryDepth == 40 || constants.queryDepth == 80)) {
        const std::string shader =
            "shaders/onnx/fused_attention_tiled_d" + std::to_string(constants.queryDepth) + ".comp.spv";
        auto operation = computation(vulkanContext, shader, constants);
        operation->setGroupCount((constants.queryCount + 63) / 64, constants.batchCount, 1);
        return operation;
    }
    auto operation = computation(vulkanContext, "shaders/onnx/fused_attention.comp.spv", constants);
    operation->setGroupCount((constants.queryCount + 7) / 8, constants.batchCount, 1);
    return operation;
}

ComputeGraphElementPtr softmax(VulkanContext& vulkanContext, const Shape& shape) {
    if (shape.empty())
        throw std::runtime_error("Softmax requires rank >= 1");
    SoftmaxPushConstants constants{elementCount(shape) / shape.back(), shape.back()};
    auto operation = computation(vulkanContext, "shaders/onnx/softmax.comp.spv", constants);
    operation->setGroupCountX((constants.outerCount + 63) / 64);
    return operation;
}

ComputeGraphElementPtr concat(VulkanContext& vulkanContext, const Shape& lhs, const Shape& rhs, uint32_t axis,
                              const Shape& output) {
    checkAxis(axis, output, "Concat");
    uint32_t outer = 1, inner = 1;
    for (uint32_t i = 0; i < axis; ++i)
        outer *= output[i];
    for (size_t i = axis + 1; i < output.size(); ++i)
        inner *= output[i];
    ConcatPushConstants constants{outer, lhs[axis], rhs[axis], inner, elementCount(output)};
    auto operation = computation(vulkanContext, "shaders/onnx/concat.comp.spv", constants);
    operation->setGroupCountX((constants.elementCount + 63) / 64);
    return operation;
}

ComputeGraphElementPtr split3(VulkanContext& vulkanContext, const Shape& input, uint32_t axis, uint32_t firstSize,
                              uint32_t secondSize) {
    checkAxis(axis, input, "Split");
    uint32_t outer = 1, inner = 1;
    for (uint32_t i = 0; i < axis; ++i)
        outer *= input[i];
    for (size_t i = axis + 1; i < input.size(); ++i)
        inner *= input[i];
    SplitPushConstants constants{outer, input[axis], inner, firstSize, secondSize};
    auto operation = computation(vulkanContext, "shaders/onnx/split3.comp.spv", constants);
    operation->setGroupCountX((elementCount(input) + 63) / 64);
    return operation;
}

ComputeGraphElementPtr expand(VulkanContext& vulkanContext, ElementType type, const Shape& input, const Shape& output) {
    if (input.size() > 4 || output.size() > 4 || input.size() > output.size()) {
        throw std::runtime_error("Expand supports ranks up to four");
    }
    ExpandPushConstants constants{};
    constants.elementCount = elementCount(output);
    constants.rank = static_cast<uint32_t>(output.size());
    padDimensions(input, constants.inputDims);
    padDimensions(output, constants.outputDims);
    for (size_t i = 0; i < 4; ++i) {
        if (constants.inputDims[i] != 1 && constants.inputDims[i] != constants.outputDims[i]) {
            throw std::runtime_error("Expand input shape cannot broadcast to its output shape");
        }
    }
    const char* shader =
        type == ElementType::Float32 ? "shaders/onnx/expand_float.comp.spv" : "shaders/onnx/expand_int64.comp.spv";
    auto operation = computation(vulkanContext, shader, constants);
    operation->setGroupCountX((constants.elementCount + 63) / 64);
    return operation;
}

ComputeGraphElementPtr gather(VulkanContext& vulkanContext, const Shape& data, const Shape& indices, uint32_t axis,
                              const Shape& output) {
    checkAxis(axis, data, "Gather");
    GatherPushConstants constants{1, data[axis], 1, elementCount(indices), elementCount(output)};
    for (uint32_t i = 0; i < axis; ++i)
        constants.outerCount *= data[i];
    for (size_t i = axis + 1; i < data.size(); ++i)
        constants.innerSize *= data[i];
    auto operation = computation(vulkanContext, "shaders/onnx/gather_float_int64.comp.spv", constants);
    operation->setGroupCountX((constants.outputCount + 63) / 64);
    return operation;
}

ComputeGraphElementPtr resizeNearest(VulkanContext& vulkanContext, const Shape& input, const Shape& output) {
    if (input.size() != 4 || output.size() != 4)
        throw std::runtime_error("Resize requires rank four");
    ResizePushConstants constants{input[0], input[1], input[2], input[3], output[2], output[3]};
    auto operation = computation(vulkanContext, "shaders/onnx/resize_nearest.comp.spv", constants);
    operation->setGroupCountX((elementCount(output) + 63) / 64);
    return operation;
}

ComputeGraphElementPtr transpose(VulkanContext& vulkanContext, const Shape& input, const Shape& output,
                                 const std::vector<uint32_t>& permutation) {
    if (input.size() > 8 || output.size() != input.size() || permutation.size() > 8) {
        throw std::runtime_error("Transpose supports ranks up to eight");
    }
    TransposePushConstants constants{};
    constants.dims = static_cast<uint32_t>(input.size());
    std::copy(input.begin(), input.end(), constants.dimInput);
    std::copy(output.begin(), output.end(), constants.dimOutput);
    std::fill(std::begin(constants.dimPerm), std::end(constants.dimPerm), 1u);
    std::copy(permutation.begin(), permutation.end(), constants.dimPerm);
    auto operation = computation(vulkanContext, "shaders/onnx/transpose.comp.spv", constants);
    operation->setGroupCountX((elementCount(output) + 63) / 64);
    return operation;
}

ComputeGraphElementPtr slice(VulkanContext& vulkanContext, ElementType type, const Shape& input, const Shape& output,
                             uint32_t start, SliceParameters parameters) {
    if (input.size() != output.size() || input.size() > 4)
        throw std::runtime_error("Unsupported Slice rank");
    SlicePushConstants constants{};
    constants.rank = static_cast<uint32_t>(input.size());
    constants.elementCount = elementCount(output);
    constants.start = start;
    bool foundAxis = false;
    for (size_t i = 0; i < input.size(); ++i) {
        constants.inputDims[i] = input[i];
        constants.outputDims[i] = output[i];
        if (input[i] != output[i]) {
            if (foundAxis)
                throw std::runtime_error("Only single-axis Slice is supported");
            constants.axis = static_cast<uint32_t>(i);
            foundAxis = true;
        }
    }
    if (!foundAxis)
        throw std::runtime_error("Slice must change one dimension");
    const std::string suffix = type == ElementType::Int64 ? "_int64" : "";
    const std::string shader = parameters == SliceParameters::StartsEnds       ? "shaders/onnx/slice3" + suffix
                               : parameters == SliceParameters::StartsEndsAxes ? "shaders/onnx/slice" + suffix
                                                                               : "shaders/onnx/slice5" + suffix;
    auto operation = computation(vulkanContext, shader + ".comp.spv", constants);
    operation->setGroupCountX((constants.elementCount + 63) / 64);
    return operation;
}

} // namespace klartraum::layers
