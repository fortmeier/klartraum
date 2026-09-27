#include "klartraum/computegraph/generalcomputation.hpp"


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
    for (const auto dimension : dimensions) count *= dimension;
    if (count > UINT32_MAX) {
        throw std::runtime_error("Tensor is too large for a Vulkan dispatch");
    }
    return static_cast<uint32_t>(count);
}

void padDimensions(const std::vector<uint32_t>& source, uint32_t (&destination)[4]) {
    if (source.size() > 4) throw std::runtime_error("ONNX tensors with rank above four are unsupported");
    std::fill(std::begin(destination), std::end(destination), 1u);
    const size_t offset = 4 - source.size();
    for (size_t i = 0; i < source.size(); ++i) destination[offset + i] = source[i];
}

int64_t getIntegerAttribute(const onnx::NodeProto& node, const std::string& name, int64_t fallback) {
    for (const auto& attribute : node.attribute()) {
        if (attribute.name() == name && attribute.has_i()) return attribute.i();
    }
    return fallback;
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
    auto biasDim = biasName.empty() ? std::vector<uint32_t>() : getTensorDimensions(biasName, name2ValueInfoProto, graph);

    // Copy dimensions to push constants
    for (size_t i = 0; i < 4; ++i) {
        pushConstants.dimInput[i] = (i < inputDim.size()) ? inputDim[i] : 1;
        pushConstants.dimWeights[i] = (i < weightDim.size()) ? weightDim[i] : 1;
    }
    pushConstants.dimBias[0] = (biasDim.size() > 0) ? biasDim[0] : 1;

    std::string shaderFilename = "shaders/onnx/conv.comp.spv";

    auto operation = vulkanContext->create<GeneralComputation<ConvPushConstants>>(shaderFilename);
    const auto outputDim = getTensorDimensions(node.output(0), name2ValueInfoProto, graph);
    for (size_t i = 0; i < 4; ++i) pushConstants.dimOutput[i] = outputDim[i];
    operation->setPushConstants({pushConstants});
    operation->setGroupCountX((outputDim[3] + 7) / 8);
    operation->setGroupCountY((outputDim[2] + 7) / 8);
    operation->setGroupCountZ(outputDim[1]);

    return operation;
}

ComputeGraphElementPtr createConvTranspose(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                           const std::map<std::string, const onnx::ValueInfoProto*>& name2ValueInfoProto,
                                           const onnx::GraphProto& graph) {
    ConvTransposePushConstants pushConstants;

    // parse attributes
    parseAttributes<uint32_t>(node, "dilations", pushConstants.dilations);
    parseAttributes<uint32_t>(node, "group", pushConstants.groups);
    parseAttributes<uint32_t>(node, "kernel_shape", pushConstants.kernel_shape);
    parseAttributes<uint32_t>(node, "pads", pushConstants.pads);
    parseAttributes<uint32_t>(node, "strides", pushConstants.strides);
    //parseAttributes<uint32_t>(node, "output_padding", pushConstants.output_padding);

    // set dimension constants
    auto inputName = node.input(0);
    auto weightsName = node.input(1);
    auto biasName = (node.input_size() > 2) ? node.input(2) : "";

    // Extract dimensions directly from ONNX graph
    auto inputDim = getTensorDimensions(inputName, name2ValueInfoProto, graph);
    auto weightDim = getTensorDimensions(weightsName, name2ValueInfoProto, graph);
    auto biasDim = biasName.empty() ? std::vector<uint32_t>() : getTensorDimensions(biasName, name2ValueInfoProto, graph);

    // Copy dimensions to push constants
    for (size_t i = 0; i < 4; ++i) {
        pushConstants.dimInput[i] = (i < inputDim.size()) ? inputDim[i] : 1;
        pushConstants.dimWeights[i] = (i < weightDim.size()) ? weightDim[i] : 1;
    }
    pushConstants.dimBias[0] = (biasDim.size() > 0) ? biasDim[0] : 1;

    // Calculate output dimensions for ConvTranspose
    // output_size = (input_size - 1) * stride - 2 * padding + kernel_size + output_padding
    uint32_t output_height = (inputDim[2] - 1) * pushConstants.strides[0] - 2 * pushConstants.pads[0] + pushConstants.kernel_shape[0]; // + pushConstants.output_padding[0];
    uint32_t output_width = (inputDim[3] - 1) * pushConstants.strides[1] - 2 * pushConstants.pads[1] + pushConstants.kernel_shape[1]; // + pushConstants.output_padding[1];
    
    pushConstants.dimOutput[0] = inputDim[0]; // batch size
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
    if (dimensions.size() < 3) throw std::runtime_error("InstanceNormalization requires rank >= 3");
    InstanceNormalizationPushConstants constants{};
    constants.batch = dimensions[0];
    constants.channels = dimensions[1];
    constants.spatialSize = 1;
    for (size_t i = 2; i < dimensions.size(); ++i) constants.spatialSize *= dimensions[i];
    constants.epsilon = 1e-5f;
    for (const auto& attribute : node.attribute()) {
        if (attribute.name() == "epsilon") constants.epsilon = attribute.f();
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

ComputeGraphElementPtr createSoftmax(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                     const std::map<std::string, const onnx::ValueInfoProto*>& infos,
                                     const onnx::GraphProto& graph) {
    const auto dimensions = getTensorDimensions(node.input(0), infos, graph);
    int64_t axis = getIntegerAttribute(node, "axis", -1);
    if (axis < 0) axis += static_cast<int64_t>(dimensions.size());
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
    if (node.output_size() != 3) throw std::runtime_error("Only three-way Split is supported");
    const auto input = getTensorDimensions(node.input(0), infos, graph);
    int64_t axis = getIntegerAttribute(node, "axis", 0);
    if (axis < 0) axis += static_cast<int64_t>(input.size());
    uint32_t outer = 1, inner = 1;
    for (int64_t i = 0; i < axis; ++i) outer *= input[i];
    for (size_t i = static_cast<size_t>(axis) + 1; i < input.size(); ++i) inner *= input[i];
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
    if (input.size() != 4 || output.size() != 4) throw std::runtime_error("Resize requires rank four");
    ResizePushConstants constants{input[0], input[1], input[2], input[3], output[2], output[3]};
    auto operation = vulkanContext->create<GeneralComputation<ResizePushConstants>>(
        "shaders/onnx/resize_nearest.comp.spv");
    operation->setPushConstants({constants});
    operation->setGroupCountX((tensorElementCount(output) + 63) / 64);
    return operation;
}

ComputeGraphElementPtr createSlice(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                   const std::map<std::string, const onnx::ValueInfoProto*>& infos,
                                   const onnx::GraphProto& graph) {
    const auto input = getTensorDimensions(node.input(0), infos, graph);
    const auto output = getTensorDimensions(node.output(0), infos, graph);
    if (input.size() != output.size() || input.size() > 4) throw std::runtime_error("Unsupported Slice rank");
    SlicePushConstants constants{};
    constants.rank = static_cast<uint32_t>(input.size());
    constants.elementCount = tensorElementCount(output);
    constants.start = 0;
    bool foundAxis = false;
    for (size_t i = 0; i < input.size(); ++i) {
        constants.inputDims[i] = input[i];
        constants.outputDims[i] = output[i];
        if (input[i] != output[i]) {
            if (foundAxis) throw std::runtime_error("Only single-axis Slice is supported");
            constants.axis = static_cast<uint32_t>(i);
            foundAxis = true;
        }
    }
    if (!foundAxis) throw std::runtime_error("Slice must change one dimension");
    auto operation = vulkanContext->create<GeneralComputation<SlicePushConstants>>("shaders/onnx/slice.comp.spv");
    operation->setPushConstants({constants});
    operation->setGroupCountX((constants.elementCount + 63) / 64);
    return operation;
}

ComputeGraphElementPtr createTensorOperation(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                             const std::map<std::string, const onnx::ValueInfoProto*>& name2ValueInfoProto,
                                             const onnx::GraphProto& graph) {
    auto output = node.output();
    auto x = output.size();
    // Create a compute operation for each node
    std::string operationType = node.op_type();
    std::cout << "Creating operation for node: " << node.name() << " of type: " << operationType << std::endl;

    ComputeGraphElementPtr operation;

    if (operationType == "Conv") {
        operation = createConv(vulkanContext, node, name2ValueInfoProto, graph);
    } else if (operationType == "Add") {
        operation = createBinaryBroadcast(vulkanContext, node, name2ValueInfoProto, graph, "shaders/onnx/add.comp.spv");
    } else if (operationType == "Mul") {
        operation = createBinaryBroadcast(vulkanContext, node, name2ValueInfoProto, graph, "shaders/onnx/mul.comp.spv");
    } else if (operationType == "Sigmoid") {
        operation = createUnary(vulkanContext, node, name2ValueInfoProto, graph, "shaders/onnx/sigmoid.comp.spv");
    } else if (operationType == "InstanceNormalization") {
        operation = createInstanceNormalization(vulkanContext, node, name2ValueInfoProto, graph);
    } else if (operationType == "MatMul") {
        operation = createMatMul(vulkanContext, node, name2ValueInfoProto, graph);
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
