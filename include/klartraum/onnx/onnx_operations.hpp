// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_ONNX_ONNX_OPERATIONS_HPP
#define KLARTRAUM_ONNX_ONNX_OPERATIONS_HPP

#include "klartraum/computegraph/copybuffer.hpp"
#include "klartraum/computegraph/noop.hpp"
#include "klartraum/layers/layers.hpp"

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
                // A shape without dimensions is a scalar.
                dimensionsFound = true;
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
                dimensionsFound = true;
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

using ValueInfos = std::map<std::string, const onnx::ValueInfoProto*>;

// Adapters from ONNX nodes to the layer factories: they read shapes and
// attributes from the model and check what the layers support.

layers::ConvAttributes getConvAttributes(const onnx::NodeProto& node) {
    uint32_t dilations[2], group[1], kernelShape[2], pads[4], strides[2];
    parseAttributes<uint32_t>(node, "dilations", dilations);
    parseAttributes<uint32_t>(node, "group", group);
    parseAttributes<uint32_t>(node, "kernel_shape", kernelShape);
    parseAttributes<uint32_t>(node, "pads", pads);
    parseAttributes<uint32_t>(node, "strides", strides);
    layers::ConvAttributes attributes;
    std::copy(std::begin(dilations), std::end(dilations), attributes.dilations.begin());
    attributes.group = group[0];
    std::copy(std::begin(kernelShape), std::end(kernelShape), attributes.kernelShape.begin());
    std::copy(std::begin(pads), std::end(pads), attributes.pads.begin());
    std::copy(std::begin(strides), std::end(strides), attributes.strides.begin());
    return attributes;
}

std::vector<uint32_t> getBiasDimensions(const onnx::NodeProto& node, const ValueInfos& infos,
                                        const onnx::GraphProto& graph) {
    return node.input_size() > 2 ? getTensorDimensions(node.input(2), infos, graph) : std::vector<uint32_t>();
}

// An axis attribute, made non-negative for a tensor of `rank`.
uint32_t getAxis(const onnx::NodeProto& node, int64_t fallback, size_t rank) {
    int64_t axis = getIntegerAttribute(node, "axis", fallback);
    if (axis < 0)
        axis += static_cast<int64_t>(rank);
    if (axis < 0 || axis >= static_cast<int64_t>(rank)) {
        throw std::runtime_error(node.op_type() + " axis is outside the tensor rank");
    }
    return static_cast<uint32_t>(axis);
}

float getEpsilon(const onnx::NodeProto& node) {
    float epsilon = 1e-5f;
    for (const auto& attribute : node.attribute()) {
        if (attribute.name() == "epsilon")
            epsilon = attribute.f();
    }
    return epsilon;
}

ComputeGraphElementPtr createConv3d(VulkanContext* vulkanContext, const onnx::NodeProto& node, const ValueInfos& infos,
                                    const onnx::GraphProto& graph) {
    if (node.input_size() != 3)
        throw std::runtime_error("Conv3d requires a bias input");
    const auto weights = getTensorDimensions(node.input(1), infos, graph);
    layers::Conv3dAttributes attributes;
    for (size_t axis = 0; axis < 3; ++axis)
        attributes.kernelShape[axis] = weights[2 + axis];
    for (const auto& attribute : node.attribute()) {
        const auto copy = [&attribute](auto& target) {
            if (static_cast<size_t>(attribute.ints_size()) != target.size()) {
                throw std::runtime_error("Conv3d attribute " + attribute.name() + " has the wrong length");
            }
            for (size_t i = 0; i < target.size(); ++i)
                target[i] = static_cast<uint32_t>(attribute.ints(i));
        };
        if (attribute.name() == "strides")
            copy(attributes.strides);
        else if (attribute.name() == "pads")
            copy(attributes.pads);
        else if (attribute.name() == "dilations")
            copy(attributes.dilations);
        else if (attribute.name() == "group")
            attributes.group = static_cast<uint32_t>(attribute.i());
        else if (attribute.name() == "auto_pad" && attribute.s() != "NOTSET") {
            throw std::runtime_error("Conv3d supports only explicit pads");
        }
    }
    return layers::conv3d(*vulkanContext, attributes, getTensorDimensions(node.input(0), infos, graph), weights,
                          getTensorDimensions(node.output(0), infos, graph));
}

ComputeGraphElementPtr createConv(VulkanContext* vulkanContext, const onnx::NodeProto& node, const ValueInfos& infos,
                                  const onnx::GraphProto& graph) {
    if (getTensorDimensions(node.input(0), infos, graph).size() == 5) {
        return createConv3d(vulkanContext, node, infos, graph);
    }
    return layers::conv(*vulkanContext, getConvAttributes(node), getTensorDimensions(node.input(0), infos, graph),
                        getTensorDimensions(node.input(1), infos, graph), getBiasDimensions(node, infos, graph),
                        getTensorDimensions(node.output(0), infos, graph));
}

ComputeGraphElementPtr createConvTranspose(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                           const ValueInfos& infos, const onnx::GraphProto& graph) {
    return layers::convTranspose(
        *vulkanContext, getConvAttributes(node), getTensorDimensions(node.input(0), infos, graph),
        getTensorDimensions(node.input(1), infos, graph), getBiasDimensions(node, infos, graph));
}

ComputeGraphElementPtr createRelu(VulkanContext* vulkanContext, const onnx::NodeProto& node, const ValueInfos& infos,
                                  const onnx::GraphProto& graph) {
    return layers::relu(*vulkanContext, getTensorDimensions(node.input(0), infos, graph));
}

ComputeGraphElementPtr createConstant(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                      const ValueInfos& infos, const onnx::GraphProto& graph) {
    // Constant operations don't perform computation - they just provide constant data
    // Use NoOp to pass the constant data through without any GPU operations
    return vulkanContext->create<NoOp>();
}

ComputeGraphElementPtr createReshape(VulkanContext* vulkanContext, const onnx::NodeProto& node, const ValueInfos& infos,
                                     const onnx::GraphProto& graph) {
    // The data is copied unchanged; input 1 is the shape tensor, so the copy
    // goes from input 0 to slot 2 (the reshaped output).
    auto operation = vulkanContext->create<CopyBuffer>();
    operation->setName(node.name());
    operation->setSrcIndex(0);
    operation->setDstIndex(2);
    return operation;
}

ComputeGraphElementPtr createTranspose(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                       const ValueInfos& infos, const onnx::GraphProto& graph) {
    uint32_t permutation[8];
    parseAttributes<uint32_t>(node, "perm", permutation);
    const auto input = getTensorDimensions(node.input(0), infos, graph);
    return layers::transpose(*vulkanContext, input, getTensorDimensions(node.output(0), infos, graph),
                             std::vector<uint32_t>(permutation, permutation + input.size()));
}

ComputeGraphElementPtr createBinaryBroadcast(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                             const ValueInfos& infos, const onnx::GraphProto& graph,
                                             layers::BinaryOp op) {
    return layers::binary(*vulkanContext, op, getTensorDimensions(node.input(0), infos, graph),
                          getTensorDimensions(node.input(1), infos, graph),
                          getTensorDimensions(node.output(0), infos, graph));
}

ComputeGraphElementPtr createUnary(VulkanContext* vulkanContext, const onnx::NodeProto& node, const ValueInfos& infos,
                                   const onnx::GraphProto& graph, layers::UnaryOp op) {
    return layers::unary(*vulkanContext, op, getTensorDimensions(node.output(0), infos, graph));
}

ComputeGraphElementPtr createInstanceNormalization(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                                   const ValueInfos& infos, const onnx::GraphProto& graph) {
    return layers::instanceNormalization(*vulkanContext, getTensorDimensions(node.input(0), infos, graph),
                                         getEpsilon(node));
}

ComputeGraphElementPtr createMatMul(VulkanContext* vulkanContext, const onnx::NodeProto& node, const ValueInfos& infos,
                                    const onnx::GraphProto& graph) {
    return layers::matMul(*vulkanContext, getTensorDimensions(node.input(0), infos, graph),
                          getTensorDimensions(node.input(1), infos, graph),
                          getTensorDimensions(node.output(0), infos, graph));
}

ComputeGraphElementPtr createFusedAttention(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                            const ValueInfos& infos, const onnx::GraphProto& graph) {
    return layers::fusedAttention(*vulkanContext, getTensorDimensions(node.input(0), infos, graph),
                                  getTensorDimensions(node.input(1), infos, graph),
                                  getTensorDimensions(node.input(2), infos, graph),
                                  getTensorDimensions(node.output(0), infos, graph));
}

ComputeGraphElementPtr createGemm(VulkanContext* vulkanContext, const onnx::NodeProto& node, const ValueInfos& infos,
                                  const onnx::GraphProto& graph) {
    if (getIntegerAttribute(node, "transA", 0) != 0 || getIntegerAttribute(node, "transB", 0) != 1) {
        throw std::runtime_error("Gemm supports only transA=0 and transB=1");
    }
    for (const auto& attribute : node.attribute()) {
        if ((attribute.name() == "alpha" || attribute.name() == "beta") && attribute.f() != 1.0f) {
            throw std::runtime_error("Gemm supports only alpha=beta=1");
        }
    }
    return layers::gemm(*vulkanContext, getTensorDimensions(node.input(0), infos, graph),
                        getTensorDimensions(node.input(1), infos, graph));
}

ComputeGraphElementPtr createLayerNormalization(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                                const ValueInfos& infos, const onnx::GraphProto& graph) {
    const auto input = getTensorDimensions(node.input(0), infos, graph);
    if (getAxis(node, -1, input.size()) != input.size() - 1) {
        throw std::runtime_error("Only last-axis LayerNormalization is supported");
    }
    return layers::layerNormalization(*vulkanContext, input, getEpsilon(node));
}

ComputeGraphElementPtr createConcat(VulkanContext* vulkanContext, const onnx::NodeProto& node, const ValueInfos& infos,
                                    const onnx::GraphProto& graph) {
    if (node.input_size() != 2)
        throw std::runtime_error("Only two-input Concat is supported");
    const auto output = getTensorDimensions(node.output(0), infos, graph);
    return layers::concat(*vulkanContext, getTensorDimensions(node.input(0), infos, graph),
                          getTensorDimensions(node.input(1), infos, graph), getAxis(node, 0, output.size()), output);
}

ComputeGraphElementPtr createCast(VulkanContext* vulkanContext, const onnx::NodeProto& node, const ValueInfos& infos,
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
    return createUnary(vulkanContext, node, infos, graph, layers::UnaryOp::CastInt64ToFloat);
}

layers::ElementType getLayerElementType(onnx::TensorProto::DataType type, const char* operation) {
    if (type == onnx::TensorProto::FLOAT)
        return layers::ElementType::Float32;
    if (type == onnx::TensorProto::INT64)
        return layers::ElementType::Int64;
    throw std::runtime_error(std::string(operation) + " supports FLOAT or INT64 tensors");
}

ComputeGraphElementPtr createExpand(VulkanContext* vulkanContext, const onnx::NodeProto& node, const ValueInfos& infos,
                                    const onnx::GraphProto& graph) {
    const auto inputType = getTensorElementType(node.input(0), infos, graph);
    if (inputType != getTensorElementType(node.output(0), infos, graph)) {
        throw std::runtime_error("Expand supports matching FLOAT or INT64 tensors");
    }
    return layers::expand(*vulkanContext, getLayerElementType(inputType, "Expand"),
                          getTensorDimensions(node.input(0), infos, graph),
                          getTensorDimensions(node.output(0), infos, graph));
}

ComputeGraphElementPtr createGather(VulkanContext* vulkanContext, const onnx::NodeProto& node, const ValueInfos& infos,
                                    const onnx::GraphProto& graph) {
    if (getTensorElementType(node.input(0), infos, graph) != onnx::TensorProto::FLOAT ||
        getTensorElementType(node.input(1), infos, graph) != onnx::TensorProto::INT64 ||
        getTensorElementType(node.output(0), infos, graph) != onnx::TensorProto::FLOAT) {
        throw std::runtime_error("Gather supports FLOAT data with INT64 indices");
    }
    const auto data = getTensorDimensions(node.input(0), infos, graph);
    return layers::gather(*vulkanContext, data, getTensorDimensions(node.input(1), infos, graph),
                          getAxis(node, 0, data.size()), getTensorDimensions(node.output(0), infos, graph));
}

ComputeGraphElementPtr createSoftmax(VulkanContext* vulkanContext, const onnx::NodeProto& node, const ValueInfos& infos,
                                     const onnx::GraphProto& graph) {
    const auto dimensions = getTensorDimensions(node.input(0), infos, graph);
    if (getAxis(node, -1, dimensions.size()) != dimensions.size() - 1) {
        throw std::runtime_error("Only last-axis Softmax is supported");
    }
    return layers::softmax(*vulkanContext, dimensions);
}

ComputeGraphElementPtr createReduceMean(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                        const ValueInfos& infos, const onnx::GraphProto& graph) {
    if (node.input_size() != 1)
        throw std::runtime_error("ReduceMean supports only the axes attribute (opset <= 17)");
    const auto input = getTensorDimensions(node.input(0), infos, graph);
    std::vector<uint32_t> axes;
    for (const auto& attribute : node.attribute()) {
        if (attribute.name() != "axes")
            continue;
        for (const auto axis : attribute.ints()) {
            const int64_t normalized = axis < 0 ? axis + static_cast<int64_t>(input.size()) : axis;
            if (normalized < 0 || normalized >= static_cast<int64_t>(input.size())) {
                throw std::runtime_error("ReduceMean axis is outside the tensor rank");
            }
            axes.push_back(static_cast<uint32_t>(normalized));
        }
    }
    if (axes.empty()) {
        for (uint32_t axis = 0; axis < input.size(); ++axis)
            axes.push_back(axis);
    }
    std::sort(axes.begin(), axes.end());
    for (size_t i = 1; i < axes.size(); ++i) {
        if (axes[i] != axes[i - 1] + 1)
            throw std::runtime_error("ReduceMean axes must be contiguous");
    }
    return layers::reduceMean(*vulkanContext, input, axes.front(), static_cast<uint32_t>(axes.size()));
}

ComputeGraphElementPtr createSplit(VulkanContext* vulkanContext, const onnx::NodeProto& node, const ValueInfos& infos,
                                   const onnx::GraphProto& graph) {
    if (node.output_size() != 3)
        throw std::runtime_error("Only three-way Split is supported");
    const auto input = getTensorDimensions(node.input(0), infos, graph);
    const uint32_t axis = getAxis(node, 0, input.size());
    return layers::split3(*vulkanContext, input, axis, getTensorDimensions(node.output(0), infos, graph)[axis],
                          getTensorDimensions(node.output(1), infos, graph)[axis]);
}

ComputeGraphElementPtr createResize(VulkanContext* vulkanContext, const onnx::NodeProto& node, const ValueInfos& infos,
                                    const onnx::GraphProto& graph) {
    return layers::resizeNearest(*vulkanContext, getTensorDimensions(node.input(0), infos, graph),
                                 getTensorDimensions(node.output(0), infos, graph));
}

ComputeGraphElementPtr createSlice(VulkanContext* vulkanContext, const onnx::NodeProto& node, const ValueInfos& infos,
                                   const onnx::GraphProto& graph) {
    const auto input = getTensorDimensions(node.input(0), infos, graph);
    const auto output = getTensorDimensions(node.output(0), infos, graph);
    if (input.size() != output.size())
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
    // The one axis the slice changes; the layer finds it again.
    size_t axis = 0;
    while (axis < input.size() && input[axis] == output[axis])
        ++axis;
    const auto starts = getInt64InitializerValues(node.input(1), graph);
    if (starts.size() != 1)
        throw std::runtime_error("Only single-axis Slice is supported");
    int64_t start = starts[0];
    if (start < 0)
        start += input[axis];
    if (start < 0 || start > UINT32_MAX)
        throw std::runtime_error("Unsupported Slice start");
    if (node.input_size() == 5) {
        const auto steps = getInt64InitializerValues(node.input(4), graph);
        if (steps.size() != 1 || steps[0] != 1)
            throw std::runtime_error("Only unit Slice steps are supported");
    }
    const auto inputType = getTensorElementType(node.input(0), infos, graph);
    if (inputType != getTensorElementType(node.output(0), infos, graph)) {
        throw std::runtime_error("Slice supports matching FLOAT or INT64 input and output tensors");
    }
    if (node.input_size() < 3 || node.input_size() > 5)
        throw std::runtime_error("Slice requires 3, 4, or 5 inputs");
    return layers::slice(*vulkanContext, getLayerElementType(inputType, "Slice"), input, output,
                         static_cast<uint32_t>(start), static_cast<layers::SliceParameters>(node.input_size() - 1));
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
        operation = createBinaryBroadcast(vulkanContext, node, name2ValueInfoProto, graph, layers::BinaryOp::Add);
    } else if (operationType == "Sub") {
        operation = createBinaryBroadcast(vulkanContext, node, name2ValueInfoProto, graph, layers::BinaryOp::Sub);
    } else if (operationType == "Mul") {
        operation = createBinaryBroadcast(vulkanContext, node, name2ValueInfoProto, graph, layers::BinaryOp::Mul);
    } else if (operationType == "Div") {
        operation = createBinaryBroadcast(vulkanContext, node, name2ValueInfoProto, graph, layers::BinaryOp::Div);
    } else if (operationType == "Sigmoid") {
        operation = createUnary(vulkanContext, node, name2ValueInfoProto, graph, layers::UnaryOp::Sigmoid);
    } else if (operationType == "Cos") {
        operation = createUnary(vulkanContext, node, name2ValueInfoProto, graph, layers::UnaryOp::Cos);
    } else if (operationType == "Sin") {
        operation = createUnary(vulkanContext, node, name2ValueInfoProto, graph, layers::UnaryOp::Sin);
    } else if (operationType == "Sqrt") {
        operation = createUnary(vulkanContext, node, name2ValueInfoProto, graph, layers::UnaryOp::Sqrt);
    } else if (operationType == "Erf") {
        operation = createUnary(vulkanContext, node, name2ValueInfoProto, graph, layers::UnaryOp::Erf);
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
    } else if (operationType == "ReduceMean") {
        operation = createReduceMean(vulkanContext, node, name2ValueInfoProto, graph);
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
