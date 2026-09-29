// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_ONNX_ONNX_OPERATIONS_HPP
#define KLARTRAUM_ONNX_ONNX_OPERATIONS_HPP

#include "klartraum/computegraph/generalcomputation.hpp"

#include <algorithm>
#include <cstring>

namespace klartraum {

template <typename T, size_t N>
void parseAttributes(const onnx::NodeProto& node, const std::string& attrName, T (&outputArray)[N]) {
    // Find the attribute with the given name
    for (int i = 0; i < node.attribute_size(); ++i) {
        const auto& attr = node.attribute(i);
        if (attr.name() == attrName) {
            if (attr.ints_size() > 0) {
                // Handle integer arrays
                size_t copySize = std::min(static_cast<size_t>(attr.ints_size()), N);
                for (size_t j = 0; j < copySize; ++j) {
                    outputArray[j] = static_cast<T>(attr.ints(j));
                }
                // Fill remaining with default values if needed
                for (size_t j = copySize; j < N; ++j) {
                    outputArray[j] = static_cast<T>(1); // Default value for most ONNX attributes
                }
                return;
            } else if (attr.floats_size() > 0) {
                // Handle float arrays
                size_t copySize = std::min(static_cast<size_t>(attr.floats_size()), N);
                for (size_t j = 0; j < copySize; ++j) {
                    outputArray[j] = static_cast<T>(attr.floats(j));
                }
                // Fill remaining with default values if needed
                for (size_t j = copySize; j < N; ++j) {
                    outputArray[j] = static_cast<T>(1); // Default value
                }
                return;
            } else if (attr.has_i()) {
                // Single integer value - broadcast to array
                T value = static_cast<T>(attr.i());
                for (size_t j = 0; j < N; ++j) {
                    outputArray[j] = value;
                }
                return;
            } else if (attr.has_f()) {
                // Single float value - broadcast to array
                T value = static_cast<T>(attr.f());
                for (size_t j = 0; j < N; ++j) {
                    outputArray[j] = value;
                }
                return;
            }
        }
    }
    throw std::runtime_error("Attribute " + attrName + " not found in node " + node.name());
}

// Helper function to extract tensor dimensions from ONNX ValueInfoProto or initializer
std::vector<uint32_t> getTensorDimensions(const std::string& tensorName,
                                          const std::map<std::string, const onnx::ValueInfoProto*>& name2ValueInfoProto,
                                          const onnx::GraphProto& graph) {
    std::vector<uint32_t> dimensions;
    bool dimensionsFound = false;

    // First check if it's in the name2ValueInfoProto map (inputs, outputs, value_info)
    auto valueIt = name2ValueInfoProto.find(tensorName);
    if (valueIt != name2ValueInfoProto.end()) {
        const onnx::ValueInfoProto* valueInfo = valueIt->second;
        if (valueInfo->has_type() && valueInfo->type().has_tensor_type()) {
            const onnx::TypeProto::Tensor& tensorType = valueInfo->type().tensor_type();
            if (tensorType.has_shape()) {
                for (int i = 0; i < tensorType.shape().dim_size(); ++i) {
                    const auto& dim = tensorType.shape().dim(i);
                    if (dim.has_dim_value()) {
                        dimensions.push_back(static_cast<uint32_t>(dim.dim_value()));
                    } else {
                        // For dynamic dimensions, use 1 as placeholder
                        dimensions.push_back(1);
                    }
                    dimensionsFound = true;
                }
            }
        }
    } else {
        // Check if it's an initializer (weights, biases, constants)
        for (int i = 0; i < graph.initializer_size(); ++i) {
            const auto& init = graph.initializer(i);
            if (init.name() == tensorName) {
                for (int j = 0; j < init.dims_size(); ++j) {
                    dimensions.push_back(static_cast<uint32_t>(init.dims(j)));
                    dimensionsFound = true;
                }
                break;
            }
        }
    }

    if (!dimensionsFound) {
        // If no dimensions found, throw a runtime exception
        throw std::runtime_error("Failed to retrieve dimensions for tensor: " + tensorName);
    }

    return dimensions;
}

uint32_t tensorElementCount(const std::vector<uint32_t>& dimensions) {
    uint64_t count = 1;
    for (const auto dimension : dimensions)
        count *= dimension;
    if (count > UINT32_MAX) {
        throw std::runtime_error("Tensor is too large for a Vulkan dispatch");
    }
    return static_cast<uint32_t>(count);
}

void padDimensions(const std::vector<uint32_t>& source, uint32_t (&destination)[4]) {
    if (source.size() > 4)
        throw std::runtime_error("ONNX tensors with rank above four are unsupported");
    std::fill(std::begin(destination), std::end(destination), 1u);
    const size_t offset = 4 - source.size();
    for (size_t i = 0; i < source.size(); ++i)
        destination[offset + i] = source[i];
}

int64_t getIntegerAttribute(const onnx::NodeProto& node, const std::string& name, int64_t fallback) {
    for (const auto& attribute : node.attribute()) {
        if (attribute.name() == name && attribute.has_i())
            return attribute.i();
    }
    return fallback;
}

std::vector<int64_t> getInt64InitializerValues(const std::string& name, const onnx::GraphProto& graph) {
    for (const auto& initializer : graph.initializer()) {
        if (initializer.name() != name)
            continue;
        if (initializer.data_type() != onnx::TensorProto::INT64) {
            throw std::runtime_error("Expected INT64 initializer for " + name);
        }
        if (!initializer.raw_data().empty()) {
            if (initializer.raw_data().size() % sizeof(int64_t) != 0) {
                throw std::runtime_error("Invalid INT64 initializer byte count for " + name);
            }
            std::vector<int64_t> values(initializer.raw_data().size() / sizeof(int64_t));
            std::memcpy(values.data(), initializer.raw_data().data(), initializer.raw_data().size());
            return values;
        }
        return {initializer.int64_data().begin(), initializer.int64_data().end()};
    }
    throw std::runtime_error("Initializer not found: " + name);
}

onnx::TensorProto::DataType getTensorElementType(const std::string& name,
                                                 const std::map<std::string, const onnx::ValueInfoProto*>& infos,
                                                 const onnx::GraphProto& graph) {
    const auto info = infos.find(name);
    if (info != infos.end()) {
        return static_cast<onnx::TensorProto::DataType>(info->second->type().tensor_type().elem_type());
    }
    for (const auto& initializer : graph.initializer()) {
        if (initializer.name() == name)
            return static_cast<onnx::TensorProto::DataType>(initializer.data_type());
    }
    throw std::runtime_error("Failed to retrieve data type for tensor: " + name);
}

ComputeGraphElementPtr createConv(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                  const std::map<std::string, const onnx::ValueInfoProto*>& name2ValueInfoProto,
                                  const onnx::GraphProto& graph) {
    ConvPushConstants pushConstants;

    // parse attributes

    parseAttributes<uint32_t>(node, "dilations", pushConstants.dilations);
    parseAttributes<uint32_t>(node, "group", pushConstants.groups);
    parseAttributes<uint32_t>(node, "kernel_shape", pushConstants.kernel_shape);
    parseAttributes<uint32_t>(node, "pads", pushConstants.pads);
    parseAttributes<uint32_t>(node, "strides", pushConstants.strides);

    // set dimension constants

    auto inputName = node.input(0);
    auto weightsName = node.input(1);
    auto biasName = (node.input_size() > 2) ? node.input(2) : "";

    // Extract dimensions directly from ONNX graph
    auto inputDim = getTensorDimensions(inputName, name2ValueInfoProto, graph);
    auto weightDim = getTensorDimensions(weightsName, name2ValueInfoProto, graph);
    auto biasDim =
        biasName.empty() ? std::vector<uint32_t>() : getTensorDimensions(biasName, name2ValueInfoProto, graph);

    // Copy dimensions to push constants
    for (size_t i = 0; i < 4; ++i) {
        pushConstants.dimInput[i] = (i < inputDim.size()) ? inputDim[i] : 1;
        pushConstants.dimWeights[i] = (i < weightDim.size()) ? weightDim[i] : 1;
    }
    pushConstants.dimBias[0] = (biasDim.size() > 0) ? biasDim[0] : 1;

    std::string shaderFilename = "shaders/onnx/conv.comp.spv";

    auto operation = vulkanContext->create<GeneralComputation<ConvPushConstants>>(shaderFilename);
    const auto outputDim = getTensorDimensions(node.output(0), name2ValueInfoProto, graph);
    for (size_t i = 0; i < 4; ++i)
        pushConstants.dimOutput[i] = outputDim[i];
    operation->setPushConstants({pushConstants});
    operation->setGroupCountX((outputDim[3] + 7) / 8);
    operation->setGroupCountY((outputDim[2] + 7) / 8);
    operation->setGroupCountZ(outputDim[0] * outputDim[1]);

    return operation;
}

ComputeGraphElementPtr
createConvTranspose(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                    const std::map<std::string, const onnx::ValueInfoProto*>& name2ValueInfoProto,
                    const onnx::GraphProto& graph) {
    ConvTransposePushConstants pushConstants;

    // parse attributes
    parseAttributes<uint32_t>(node, "dilations", pushConstants.dilations);
    parseAttributes<uint32_t>(node, "group", pushConstants.groups);
    parseAttributes<uint32_t>(node, "kernel_shape", pushConstants.kernel_shape);
    parseAttributes<uint32_t>(node, "pads", pushConstants.pads);
    parseAttributes<uint32_t>(node, "strides", pushConstants.strides);
    // parseAttributes<uint32_t>(node, "output_padding", pushConstants.output_padding);

    // set dimension constants
    auto inputName = node.input(0);
    auto weightsName = node.input(1);
    auto biasName = (node.input_size() > 2) ? node.input(2) : "";

    // Extract dimensions directly from ONNX graph
    auto inputDim = getTensorDimensions(inputName, name2ValueInfoProto, graph);
    auto weightDim = getTensorDimensions(weightsName, name2ValueInfoProto, graph);
    auto biasDim =
        biasName.empty() ? std::vector<uint32_t>() : getTensorDimensions(biasName, name2ValueInfoProto, graph);

    // Copy dimensions to push constants
    for (size_t i = 0; i < 4; ++i) {
        pushConstants.dimInput[i] = (i < inputDim.size()) ? inputDim[i] : 1;
        pushConstants.dimWeights[i] = (i < weightDim.size()) ? weightDim[i] : 1;
    }
    pushConstants.dimBias[0] = (biasDim.size() > 0) ? biasDim[0] : 1;

    // Calculate output dimensions for ConvTranspose
    // output_size = (input_size - 1) * stride - 2 * padding + kernel_size + output_padding
    uint32_t output_height = (inputDim[2] - 1) * pushConstants.strides[0] - 2 * pushConstants.pads[0] +
                             pushConstants.kernel_shape[0]; // + pushConstants.output_padding[0];
    uint32_t output_width = (inputDim[3] - 1) * pushConstants.strides[1] - 2 * pushConstants.pads[1] +
                            pushConstants.kernel_shape[1]; // + pushConstants.output_padding[1];

    pushConstants.dimOutput[0] = inputDim[0];  // batch size
    pushConstants.dimOutput[1] = weightDim[1]; // output channels from weights
    pushConstants.dimOutput[2] = output_height;
    pushConstants.dimOutput[3] = output_width;

    std::string shaderFilename = "shaders/onnx/conv_transpose.comp.spv";

    auto operation = vulkanContext->create<GeneralComputation<ConvTransposePushConstants>>(shaderFilename);
    operation->setPushConstants({pushConstants});
    // One invocation per output element: x covers the output width, y the
    // output height, z the output channels (see shaders/onnx/conv_transpose.comp,
    // local size 8x8x1).
    operation->setGroupCountX((pushConstants.dimOutput[3] + 7) / 8);
    operation->setGroupCountY((pushConstants.dimOutput[2] + 7) / 8);
    operation->setGroupCountZ(pushConstants.dimOutput[1]);

    return operation;
}

ComputeGraphElementPtr createRelu(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                  const std::map<std::string, const onnx::ValueInfoProto*>& name2ValueInfoProto,
                                  const onnx::GraphProto& graph) {
    TensorOpPushConstants pushConstants;

    auto inputName = node.input(0);
    auto inputDim = getTensorDimensions(inputName, name2ValueInfoProto, graph);

    // set dimension constants
    for (size_t i = 0; i < 4; ++i) {
        pushConstants.dimInput[i] = (i < inputDim.size()) ? inputDim[i] : 1;
        pushConstants.dimOutput[i] = (i < inputDim.size()) ? inputDim[i] : 1; // Output dimensions are the same as input
    }

    std::string shaderFilename = "shaders/onnx/relu.comp.spv";

    auto operation = vulkanContext->create<GeneralComputation<TensorOpPushConstants>>(shaderFilename);
    operation->setPushConstants({pushConstants});
    uint32_t elementCount = 1;
    for (const auto dimension : inputDim) {
        elementCount *= dimension;
    }
    operation->setGroupCountX((elementCount + 63) / 64);

    return operation;
}

ComputeGraphElementPtr createConstant(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                      const std::map<std::string, const onnx::ValueInfoProto*>& name2ValueInfoProto,
                                      const onnx::GraphProto& graph) {
    // Constant operations don't perform computation - they just provide constant data
    // Use NoOp to pass the constant data through without any GPU operations
    auto operation = vulkanContext->create<NoOp>();

    return operation;
}

ComputeGraphElementPtr createReshape(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                     const std::map<std::string, const onnx::ValueInfoProto*>& name2ValueInfoProto,
                                     const onnx::GraphProto& graph) {
    // ReshapePushConstants pushConstants;

    auto inputName = node.input(0);
    auto shapeName = node.input(1);

    auto inputDim = getTensorDimensions(inputName, name2ValueInfoProto, graph);
    auto shapeDim = getTensorDimensions(shapeName, name2ValueInfoProto, graph);

    // // Copy dimensions to push constants
    // for (size_t i = 0; i < 4; ++i) {
    //     pushConstants.dimInput[i] = (i < inputDim.size()) ? inputDim[i] : 1;
    // }

    // for (size_t i = 0; i < 6; ++i) {
    //     pushConstants.dimShape[i] = (i < shapeDim.size()) ? shapeDim[i] : 1;
    // }

    // std::string shaderFilename = "shaders/onnx/reshape.comp.spv";

    // auto operation = vulkanContext->create<GeneralComputation<ReshapePushConstants>>(shaderFilename);
    // operation->setPushConstants({pushConstants});

    auto operation = vulkanContext->create<CopyBuffer>();
    operation->setName(node.name());

    // input with index 1 is the shape tensor
    // so the copy operation needs to copy from input 0 to input 2 (reshaped)
    operation->setSrcIndex(0);
    operation->setDstIndex(2);

    return operation;
}

ComputeGraphElementPtr createTranspose(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                       const std::map<std::string, const onnx::ValueInfoProto*>& name2ValueInfoProto,
                                       const onnx::GraphProto& graph) {
    TransposePushConstants pushConstants;

    const auto inputDimensions = getTensorDimensions(node.input(0), name2ValueInfoProto, graph);
    const auto outputDimensions = getTensorDimensions(node.output(0), name2ValueInfoProto, graph);
    pushConstants = {};
    pushConstants.dims = static_cast<uint32_t>(inputDimensions.size());
    for (size_t i = 0; i < inputDimensions.size(); ++i) {
        pushConstants.dimInput[i] = inputDimensions[i];
        pushConstants.dimOutput[i] = outputDimensions[i];
    }

    // Parse the perm attribute
    parseAttributes<uint32_t>(node, "perm", pushConstants.dimPerm);

    std::string shaderFilename = "shaders/onnx/transpose.comp.spv";

    auto operation = vulkanContext->create<GeneralComputation<TransposePushConstants>>(shaderFilename);
    operation->setPushConstants({pushConstants});
    operation->setGroupCountX((tensorElementCount(outputDimensions) + 63) / 64);

    return operation;
}

ComputeGraphElementPtr createBinaryBroadcast(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                             const std::map<std::string, const onnx::ValueInfoProto*>& infos,
                                             const onnx::GraphProto& graph, const std::string& shader) {
    const auto lhs = getTensorDimensions(node.input(0), infos, graph);
    const auto rhs = getTensorDimensions(node.input(1), infos, graph);
    const auto output = getTensorDimensions(node.output(0), infos, graph);
    BinaryBroadcastPushConstants constants{};
    constants.rank = static_cast<uint32_t>(output.size());
    constants.elementCount = tensorElementCount(output);
    padDimensions(lhs, constants.lhsDims);
    padDimensions(rhs, constants.rhsDims);
    padDimensions(output, constants.outputDims);
    auto operation = vulkanContext->create<GeneralComputation<BinaryBroadcastPushConstants>>(shader);
    operation->setPushConstants({constants});
    operation->setGroupCountX((constants.elementCount + 63) / 64);
    return operation;
}

ComputeGraphElementPtr createUnary(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                   const std::map<std::string, const onnx::ValueInfoProto*>& infos,
                                   const onnx::GraphProto& graph, const std::string& shader) {
    UnaryPushConstants constants{tensorElementCount(getTensorDimensions(node.output(0), infos, graph))};
    auto operation = vulkanContext->create<GeneralComputation<UnaryPushConstants>>(shader);
    operation->setPushConstants({constants});
    operation->setGroupCountX((constants.elementCount + 63) / 64);
    return operation;
}

ComputeGraphElementPtr createInstanceNormalization(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                                   const std::map<std::string, const onnx::ValueInfoProto*>& infos,
                                                   const onnx::GraphProto& graph) {
    const auto dimensions = getTensorDimensions(node.input(0), infos, graph);
    if (dimensions.size() < 3)
        throw std::runtime_error("InstanceNormalization requires rank >= 3");
    InstanceNormalizationPushConstants constants{};
    constants.batch = dimensions[0];
    constants.channels = dimensions[1];
    constants.spatialSize = 1;
    for (size_t i = 2; i < dimensions.size(); ++i)
        constants.spatialSize *= dimensions[i];
    constants.epsilon = 1e-5f;
    for (const auto& attribute : node.attribute()) {
        if (attribute.name() == "epsilon")
            constants.epsilon = attribute.f();
    }
    auto operation = vulkanContext->create<GeneralComputation<InstanceNormalizationPushConstants>>(
        "shaders/onnx/instance_normalization.comp.spv");
    operation->setPushConstants({constants});
    operation->setGroupCountX((constants.batch * constants.channels + 63) / 64);
    return operation;
}

ComputeGraphElementPtr createMatMul(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                    const std::map<std::string, const onnx::ValueInfoProto*>& infos,
                                    const onnx::GraphProto& graph) {
    const auto lhs = getTensorDimensions(node.input(0), infos, graph);
    const auto rhs = getTensorDimensions(node.input(1), infos, graph);
    const auto output = getTensorDimensions(node.output(0), infos, graph);
    if (lhs.size() < 2 || rhs.size() < 2 || output.size() < 2) {
        throw std::runtime_error("MatMul requires rank >= 2");
    }
    MatMulPushConstants constants{};
    constants.rows = output[output.size() - 2];
    constants.columns = output.back();
    constants.reduction = lhs.back();
    constants.batchCount = tensorElementCount(output) / (constants.rows * constants.columns);
    constants.lhsBatchCount = tensorElementCount(lhs) / (constants.rows * constants.reduction);
    constants.rhsBatchCount = tensorElementCount(rhs) / (constants.reduction * constants.columns);
    if (constants.lhsBatchCount != 1 && constants.lhsBatchCount != constants.batchCount) {
        throw std::runtime_error("Unsupported MatMul left batch broadcasting");
    }
    if (constants.rhsBatchCount != 1 && constants.rhsBatchCount != constants.batchCount) {
        throw std::runtime_error("Unsupported MatMul right batch broadcasting");
    }
    auto operation = vulkanContext->create<GeneralComputation<MatMulPushConstants>>("shaders/onnx/matmul.comp.spv");
    operation->setPushConstants({constants});
    operation->setGroupCount((constants.columns + 7) / 8, (constants.rows + 7) / 8, constants.batchCount);
    return operation;
}

ComputeGraphElementPtr createFusedAttention(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                            const std::map<std::string, const onnx::ValueInfoProto*>& infos,
                                            const onnx::GraphProto& graph) {
    const auto query = getTensorDimensions(node.input(0), infos, graph);
    const auto key = getTensorDimensions(node.input(1), infos, graph);
    const auto value = getTensorDimensions(node.input(2), infos, graph);
    const auto output = getTensorDimensions(node.output(0), infos, graph);
    if (query.size() != 4 || key.size() != 4 || value.size() != 4 || output.size() != 4) {
        throw std::runtime_error("FusedAttention requires rank-four tensors");
    }
    FusedAttentionPushConstants constants{query[0] * query[1], query[2], key[3], query[3], value[3]};
    if (key[0] * key[1] != constants.batchCount || value[0] * value[1] != constants.batchCount ||
        key[2] != constants.queryDepth || value[2] != constants.keyCount ||
        output[0] * output[1] != constants.batchCount || output[2] != constants.queryCount ||
        output[3] != constants.valueDepth || constants.valueDepth > 512) {
        throw std::runtime_error("Unsupported FusedAttention tensor shapes");
    }
    auto operation =
        vulkanContext->create<GeneralComputation<FusedAttentionPushConstants>>("shaders/onnx/fused_attention.comp.spv");
    operation->setPushConstants({constants});
    operation->setGroupCount(constants.queryCount, constants.batchCount, 1);
    return operation;
}

ComputeGraphElementPtr createGemm(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                  const std::map<std::string, const onnx::ValueInfoProto*>& infos,
                                  const onnx::GraphProto& graph) {
    if (getIntegerAttribute(node, "transA", 0) != 0 || getIntegerAttribute(node, "transB", 0) != 1) {
        throw std::runtime_error("Gemm supports only transA=0 and transB=1");
    }
    for (const auto& attribute : node.attribute()) {
        if ((attribute.name() == "alpha" || attribute.name() == "beta") && attribute.f() != 1.0f) {
            throw std::runtime_error("Gemm supports only alpha=beta=1");
        }
    }
    const auto input = getTensorDimensions(node.input(0), infos, graph);
    const auto weights = getTensorDimensions(node.input(1), infos, graph);
    if (input.size() != 2 || weights.size() != 2)
        throw std::runtime_error("Gemm requires rank-two tensors");
    GemmPushConstants constants{input[0], weights[0], input[1]};
    auto operation = vulkanContext->create<GeneralComputation<GemmPushConstants>>("shaders/onnx/gemm.comp.spv");
    operation->setPushConstants({constants});
    operation->setGroupCount((constants.columns + 7) / 8, (constants.rows + 7) / 8, 1);
    return operation;
}

ComputeGraphElementPtr createLayerNormalization(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                                const std::map<std::string, const onnx::ValueInfoProto*>& infos,
                                                const onnx::GraphProto& graph) {
    const auto input = getTensorDimensions(node.input(0), infos, graph);
    int64_t axis = getIntegerAttribute(node, "axis", -1);
    if (axis < 0)
        axis += static_cast<int64_t>(input.size());
    if (axis != static_cast<int64_t>(input.size()) - 1) {
        throw std::runtime_error("Only last-axis LayerNormalization is supported");
    }
    LayerNormalizationPushConstants constants{};
    constants.axisSize = input.back();
    constants.outerCount = tensorElementCount(input) / constants.axisSize;
    constants.epsilon = 1e-5f;
    for (const auto& attribute : node.attribute())
        if (attribute.name() == "epsilon")
            constants.epsilon = attribute.f();
    auto operation = vulkanContext->create<GeneralComputation<LayerNormalizationPushConstants>>(
        "shaders/onnx/layer_normalization.comp.spv");
    operation->setPushConstants({constants});
    operation->setGroupCountX((constants.outerCount + 63) / 64);
    return operation;
}

ComputeGraphElementPtr createConcat(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                    const std::map<std::string, const onnx::ValueInfoProto*>& infos,
                                    const onnx::GraphProto& graph) {
    if (node.input_size() != 2)
        throw std::runtime_error("Only two-input Concat is supported");
    const auto lhs = getTensorDimensions(node.input(0), infos, graph);
    const auto rhs = getTensorDimensions(node.input(1), infos, graph);
    const auto output = getTensorDimensions(node.output(0), infos, graph);
    int64_t axis = getIntegerAttribute(node, "axis", 0);
    if (axis < 0)
        axis += static_cast<int64_t>(output.size());
    uint32_t outer = 1, inner = 1;
    for (int64_t i = 0; i < axis; ++i)
        outer *= output[i];
    for (size_t i = static_cast<size_t>(axis) + 1; i < output.size(); ++i)
        inner *= output[i];
    ConcatPushConstants constants{outer, lhs[axis], rhs[axis], inner, tensorElementCount(output)};
    auto operation = vulkanContext->create<GeneralComputation<ConcatPushConstants>>("shaders/onnx/concat.comp.spv");
    operation->setPushConstants({constants});
    operation->setGroupCountX((constants.elementCount + 63) / 64);
    return operation;
}

ComputeGraphElementPtr createCast(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                  const std::map<std::string, const onnx::ValueInfoProto*>& infos,
                                  const onnx::GraphProto& graph) {
    const auto inputType = getTensorElementType(node.input(0), infos, graph);
    const auto outputType = getTensorElementType(node.output(0), infos, graph);
    if (inputType == outputType) {
        auto operation = vulkanContext->create<CopyBuffer>();
        operation->setSrcIndex(0);
        operation->setDstIndex(1);
        return operation;
    }
    if (inputType != onnx::TensorProto::INT64 || outputType != onnx::TensorProto::FLOAT) {
        throw std::runtime_error("Cast supports only identity and INT64-to-FLOAT conversions");
    }
    return createUnary(vulkanContext, node, infos, graph, "shaders/onnx/cast_int64_float.comp.spv");
}

ComputeGraphElementPtr createExpand(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                    const std::map<std::string, const onnx::ValueInfoProto*>& infos,
                                    const onnx::GraphProto& graph) {
    const auto inputType = getTensorElementType(node.input(0), infos, graph);
    const auto outputType = getTensorElementType(node.output(0), infos, graph);
    if (inputType != outputType || (inputType != onnx::TensorProto::INT64 && inputType != onnx::TensorProto::FLOAT)) {
        throw std::runtime_error("Expand supports matching FLOAT or INT64 tensors");
    }
    const auto input = getTensorDimensions(node.input(0), infos, graph);
    const auto output = getTensorDimensions(node.output(0), infos, graph);
    if (input.size() > 4 || output.size() > 4 || input.size() > output.size()) {
        throw std::runtime_error("Expand supports ranks up to four");
    }
    ExpandPushConstants constants{};
    constants.elementCount = tensorElementCount(output);
    constants.rank = static_cast<uint32_t>(output.size());
    std::fill(std::begin(constants.inputDims), std::end(constants.inputDims), 1);
    std::fill(std::begin(constants.outputDims), std::end(constants.outputDims), 1);
    const size_t inputOffset = 4 - input.size();
    const size_t outputOffset = 4 - output.size();
    for (size_t index = 0; index < input.size(); ++index)
        constants.inputDims[inputOffset + index] = input[index];
    for (size_t index = 0; index < output.size(); ++index)
        constants.outputDims[outputOffset + index] = output[index];
    for (size_t index = 0; index < 4; ++index) {
        if (constants.inputDims[index] != 1 && constants.inputDims[index] != constants.outputDims[index]) {
            throw std::runtime_error("Expand input shape cannot broadcast to its output shape");
        }
    }
    const char* shader = inputType == onnx::TensorProto::FLOAT ? "shaders/onnx/expand_float.comp.spv"
                                                               : "shaders/onnx/expand_int64.comp.spv";
    auto operation = vulkanContext->create<GeneralComputation<ExpandPushConstants>>(shader);
    operation->setPushConstants({constants});
    operation->setGroupCountX((constants.elementCount + 63) / 64);
    return operation;
}

ComputeGraphElementPtr createGather(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                    const std::map<std::string, const onnx::ValueInfoProto*>& infos,
                                    const onnx::GraphProto& graph) {
    if (getTensorElementType(node.input(0), infos, graph) != onnx::TensorProto::FLOAT ||
        getTensorElementType(node.input(1), infos, graph) != onnx::TensorProto::INT64 ||
        getTensorElementType(node.output(0), infos, graph) != onnx::TensorProto::FLOAT) {
        throw std::runtime_error("Gather supports FLOAT data with INT64 indices");
    }
    const auto data = getTensorDimensions(node.input(0), infos, graph);
    const auto indices = getTensorDimensions(node.input(1), infos, graph);
    int64_t axis = getIntegerAttribute(node, "axis", 0);
    if (axis < 0)
        axis += static_cast<int64_t>(data.size());
    if (axis < 0 || axis >= static_cast<int64_t>(data.size())) {
        throw std::runtime_error("Gather axis is outside the data tensor rank");
    }
    GatherPushConstants constants{1, data[axis], 1, tensorElementCount(indices),
                                  tensorElementCount(getTensorDimensions(node.output(0), infos, graph))};
    for (int64_t dimension = 0; dimension < axis; ++dimension)
        constants.outerCount *= data[dimension];
    for (size_t dimension = static_cast<size_t>(axis) + 1; dimension < data.size(); ++dimension) {
        constants.innerSize *= data[dimension];
    }
    auto operation =
        vulkanContext->create<GeneralComputation<GatherPushConstants>>("shaders/onnx/gather_float_int64.comp.spv");
    operation->setPushConstants({constants});
    operation->setGroupCountX((constants.outputCount + 63) / 64);
    return operation;
}

ComputeGraphElementPtr createSoftmax(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                     const std::map<std::string, const onnx::ValueInfoProto*>& infos,
                                     const onnx::GraphProto& graph) {
    const auto dimensions = getTensorDimensions(node.input(0), infos, graph);
    int64_t axis = getIntegerAttribute(node, "axis", -1);
    if (axis < 0)
        axis += static_cast<int64_t>(dimensions.size());
    if (axis != static_cast<int64_t>(dimensions.size()) - 1) {
        throw std::runtime_error("Only last-axis Softmax is supported");
    }
    SoftmaxPushConstants constants{tensorElementCount(dimensions) / dimensions.back(), dimensions.back()};
    auto operation = vulkanContext->create<GeneralComputation<SoftmaxPushConstants>>("shaders/onnx/softmax.comp.spv");
    operation->setPushConstants({constants});
    operation->setGroupCountX((constants.outerCount + 63) / 64);
    return operation;
}

ComputeGraphElementPtr createSplit(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                   const std::map<std::string, const onnx::ValueInfoProto*>& infos,
                                   const onnx::GraphProto& graph) {
    if (node.output_size() != 3)
        throw std::runtime_error("Only three-way Split is supported");
    const auto input = getTensorDimensions(node.input(0), infos, graph);
    int64_t axis = getIntegerAttribute(node, "axis", 0);
    if (axis < 0)
        axis += static_cast<int64_t>(input.size());
    uint32_t outer = 1, inner = 1;
    for (int64_t i = 0; i < axis; ++i)
        outer *= input[i];
    for (size_t i = static_cast<size_t>(axis) + 1; i < input.size(); ++i)
        inner *= input[i];
    const auto output0 = getTensorDimensions(node.output(0), infos, graph);
    const auto output1 = getTensorDimensions(node.output(1), infos, graph);
    SplitPushConstants constants{outer, input[axis], inner, output0[axis], output1[axis]};
    auto operation = vulkanContext->create<GeneralComputation<SplitPushConstants>>("shaders/onnx/split3.comp.spv");
    operation->setPushConstants({constants});
    operation->setGroupCountX((tensorElementCount(input) + 63) / 64);
    return operation;
}

ComputeGraphElementPtr createResize(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                    const std::map<std::string, const onnx::ValueInfoProto*>& infos,
                                    const onnx::GraphProto& graph) {
    const auto input = getTensorDimensions(node.input(0), infos, graph);
    const auto output = getTensorDimensions(node.output(0), infos, graph);
    if (input.size() != 4 || output.size() != 4)
        throw std::runtime_error("Resize requires rank four");
    ResizePushConstants constants{input[0], input[1], input[2], input[3], output[2], output[3]};
    auto operation =
        vulkanContext->create<GeneralComputation<ResizePushConstants>>("shaders/onnx/resize_nearest.comp.spv");
    operation->setPushConstants({constants});
    operation->setGroupCountX((tensorElementCount(output) + 63) / 64);
    return operation;
}

ComputeGraphElementPtr createSlice(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                   const std::map<std::string, const onnx::ValueInfoProto*>& infos,
                                   const onnx::GraphProto& graph) {
    const auto input = getTensorDimensions(node.input(0), infos, graph);
    const auto output = getTensorDimensions(node.output(0), infos, graph);
    if (input.size() != output.size() || input.size() > 4)
        throw std::runtime_error("Unsupported Slice rank");
    if (input == output) {
        const auto starts = getInt64InitializerValues(node.input(1), graph);
        const auto ends = getInt64InitializerValues(node.input(2), graph);
        const auto axes =
            node.input_size() > 3 ? getInt64InitializerValues(node.input(3), graph) : std::vector<int64_t>{0};
        const auto steps =
            node.input_size() > 4 ? getInt64InitializerValues(node.input(4), graph) : std::vector<int64_t>{1};
        int64_t fullAxis = axes.size() == 1 ? axes[0] : 0;
        if (fullAxis < 0)
            fullAxis += static_cast<int64_t>(input.size());
        if (starts.size() != 1 || ends.size() != 1 || axes.size() != 1 || steps.size() != 1 || fullAxis < 0 ||
            fullAxis >= static_cast<int64_t>(input.size()) || starts[0] != 0 || steps[0] != 1 ||
            ends[0] < static_cast<int64_t>(input[static_cast<size_t>(fullAxis)])) {
            throw std::runtime_error("Unsupported shape-preserving Slice");
        }
        return vulkanContext->create<NoOp>();
    }
    SlicePushConstants constants{};
    constants.rank = static_cast<uint32_t>(input.size());
    constants.elementCount = tensorElementCount(output);
    constants.start = 0;
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
    const auto starts = getInt64InitializerValues(node.input(1), graph);
    if (starts.size() != 1)
        throw std::runtime_error("Only single-axis Slice is supported");
    int64_t start = starts[0];
    if (start < 0)
        start += input[constants.axis];
    if (start < 0 || start > UINT32_MAX)
        throw std::runtime_error("Unsupported Slice start");
    constants.start = static_cast<uint32_t>(start);
    if (node.input_size() == 5) {
        const auto steps = getInt64InitializerValues(node.input(4), graph);
        if (steps.size() != 1 || steps[0] != 1)
            throw std::runtime_error("Only unit Slice steps are supported");
    }
    const auto inputType = getTensorElementType(node.input(0), infos, graph);
    const auto outputType = getTensorElementType(node.output(0), infos, graph);
    if (inputType != outputType || (inputType != onnx::TensorProto::FLOAT && inputType != onnx::TensorProto::INT64)) {
        throw std::runtime_error("Slice supports matching FLOAT or INT64 input and output tensors");
    }
    const std::string typeSuffix = inputType == onnx::TensorProto::INT64 ? "_int64" : "";
    std::string shader;
    if (node.input_size() == 3)
        shader = "shaders/onnx/slice3" + typeSuffix + ".comp.spv";
    else if (node.input_size() == 4)
        shader = "shaders/onnx/slice" + typeSuffix + ".comp.spv";
    else if (node.input_size() == 5)
        shader = "shaders/onnx/slice5" + typeSuffix + ".comp.spv";
    else
        throw std::runtime_error("Slice requires 3, 4, or 5 inputs");
    auto operation = vulkanContext->create<GeneralComputation<SlicePushConstants>>(shader);
    operation->setPushConstants({constants});
    operation->setGroupCountX((constants.elementCount + 63) / 64);
    return operation;
}

ComputeGraphElementPtr
createTensorOperation(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                      const std::map<std::string, const onnx::ValueInfoProto*>& name2ValueInfoProto,
                      const onnx::GraphProto& graph) {
    auto output = node.output();
    auto x = output.size();
    // Create a compute operation for each node
    std::string operationType = node.op_type();

    ComputeGraphElementPtr operation;

    if (operationType == "Conv") {
        operation = createConv(vulkanContext, node, name2ValueInfoProto, graph);
    } else if (operationType == "Add") {
        operation = createBinaryBroadcast(vulkanContext, node, name2ValueInfoProto, graph, "shaders/onnx/add.comp.spv");
    } else if (operationType == "Sub") {
        operation = createBinaryBroadcast(vulkanContext, node, name2ValueInfoProto, graph, "shaders/onnx/sub.comp.spv");
    } else if (operationType == "Mul") {
        operation = createBinaryBroadcast(vulkanContext, node, name2ValueInfoProto, graph, "shaders/onnx/mul.comp.spv");
    } else if (operationType == "Div") {
        operation = createBinaryBroadcast(vulkanContext, node, name2ValueInfoProto, graph, "shaders/onnx/div.comp.spv");
    } else if (operationType == "Sigmoid") {
        operation = createUnary(vulkanContext, node, name2ValueInfoProto, graph, "shaders/onnx/sigmoid.comp.spv");
    } else if (operationType == "Cos") {
        operation = createUnary(vulkanContext, node, name2ValueInfoProto, graph, "shaders/onnx/cos.comp.spv");
    } else if (operationType == "Sin") {
        operation = createUnary(vulkanContext, node, name2ValueInfoProto, graph, "shaders/onnx/sin.comp.spv");
    } else if (operationType == "Sqrt") {
        operation = createUnary(vulkanContext, node, name2ValueInfoProto, graph, "shaders/onnx/sqrt.comp.spv");
    } else if (operationType == "Erf") {
        operation = createUnary(vulkanContext, node, name2ValueInfoProto, graph, "shaders/onnx/erf.comp.spv");
    } else if (operationType == "InstanceNormalization") {
        operation = createInstanceNormalization(vulkanContext, node, name2ValueInfoProto, graph);
    } else if (operationType == "FusedAttention") {
        operation = createFusedAttention(vulkanContext, node, name2ValueInfoProto, graph);
    } else if (operationType == "MatMul") {
        operation = createMatMul(vulkanContext, node, name2ValueInfoProto, graph);
    } else if (operationType == "Gemm") {
        operation = createGemm(vulkanContext, node, name2ValueInfoProto, graph);
    } else if (operationType == "LayerNormalization") {
        operation = createLayerNormalization(vulkanContext, node, name2ValueInfoProto, graph);
    } else if (operationType == "Concat") {
        operation = createConcat(vulkanContext, node, name2ValueInfoProto, graph);
    } else if (operationType == "Cast") {
        operation = createCast(vulkanContext, node, name2ValueInfoProto, graph);
    } else if (operationType == "Expand") {
        operation = createExpand(vulkanContext, node, name2ValueInfoProto, graph);
    } else if (operationType == "Gather") {
        operation = createGather(vulkanContext, node, name2ValueInfoProto, graph);
    } else if (operationType == "Unsqueeze") {
        operation = createReshape(vulkanContext, node, name2ValueInfoProto, graph);
    } else if (operationType == "Softmax") {
        operation = createSoftmax(vulkanContext, node, name2ValueInfoProto, graph);
    } else if (operationType == "Split") {
        operation = createSplit(vulkanContext, node, name2ValueInfoProto, graph);
    } else if (operationType == "Resize") {
        operation = createResize(vulkanContext, node, name2ValueInfoProto, graph);
    } else if (operationType == "Slice") {
        operation = createSlice(vulkanContext, node, name2ValueInfoProto, graph);
    } else if (operationType == "Relu") {
        operation = createRelu(vulkanContext, node, name2ValueInfoProto, graph);
    } else if (operationType == "Constant") {
        operation = createConstant(vulkanContext, node, name2ValueInfoProto, graph);
    } else if (operationType == "Reshape") {
        operation = createReshape(vulkanContext, node, name2ValueInfoProto, graph);
    } else if (operationType == "Transpose") {
        operation = createTranspose(vulkanContext, node, name2ValueInfoProto, graph);
    } else if (operationType == "ConvTranspose") {
        operation = createConvTranspose(vulkanContext, node, name2ValueInfoProto, graph);
    } else {
        throw std::runtime_error("Unsupported operation type: " + operationType);
    }

    return operation;
}

} // namespace klartraum

#endif // KLARTRAUM_ONNX_ONNX_OPERATIONS_HPP
