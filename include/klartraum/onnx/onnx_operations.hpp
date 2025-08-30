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
}

// Helper function to extract tensor dimensions from ONNX ValueInfoProto or initializer
std::vector<uint32_t> getTensorDimensions(const std::string& tensorName,
                                          const std::map<std::string, const onnx::ValueInfoProto*>& name2Value,
                                          const onnx::GraphProto& graph) {
    std::vector<uint32_t> dimensions;

    // First check if it's in the name2Value map (inputs, outputs, value_info)
    auto valueIt = name2Value.find(tensorName);
    if (valueIt != name2Value.end()) {
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
                }
                break;
            }
        }
    }

    return dimensions;
}

ComputeGraphElementPtr createConv(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                  const std::map<std::string, const onnx::ValueInfoProto*>& name2Value,
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
    auto inputDim = getTensorDimensions(inputName, name2Value, graph);
    auto weightDim = getTensorDimensions(weightsName, name2Value, graph);
    auto biasDim = biasName.empty() ? std::vector<uint32_t>() : getTensorDimensions(biasName, name2Value, graph);

    // Copy dimensions to push constants
    for (size_t i = 0; i < 4; ++i) {
        pushConstants.dimInput[i] = (i < inputDim.size()) ? inputDim[i] : 1;
        pushConstants.dimWeights[i] = (i < weightDim.size()) ? weightDim[i] : 1;
    }
    pushConstants.dimBias[0] = (biasDim.size() > 0) ? biasDim[0] : 1;

    std::string shaderFilename = "shaders/onnx/conv.comp.spv";

    auto operation = vulkanContext->create<GeneralComputation<ConvPushConstants>>(shaderFilename);
    operation->setPushConstants({pushConstants});

    return operation;
}

ComputeGraphElementPtr createRelu(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                  const std::map<std::string, const onnx::ValueInfoProto*>& name2Value,
                                  const onnx::GraphProto& graph) {
    TensorOpPushConstants pushConstants;

    auto inputName = node.input(0);
    auto inputDim = getTensorDimensions(inputName, name2Value, graph);

    // set dimension constants
    for (size_t i = 0; i < 4; ++i) {
        pushConstants.dimInput[i] = (i < inputDim.size()) ? inputDim[i] : 1;
        pushConstants.dimOutput[i] = (i < inputDim.size()) ? inputDim[i] : 1; // Output dimensions are the same as input
    }

    std::string shaderFilename = "shaders/onnx/relu.comp.spv";

    auto operation = vulkanContext->create<GeneralComputation<TensorOpPushConstants>>(shaderFilename);
    operation->setPushConstants({pushConstants});

    return operation;
}

ComputeGraphElementPtr createConstant(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                      const std::map<std::string, const onnx::ValueInfoProto*>& name2Value,
                                      const onnx::GraphProto& graph) {
    // Constant operations don't perform computation - they just provide constant data
    // Use NoOp to pass the constant data through without any GPU operations
    auto operation = vulkanContext->create<NoOp>();

    return operation;
}

ComputeGraphElementPtr createReshape(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                     const std::map<std::string, const onnx::ValueInfoProto*>& name2Value,
                                     const onnx::GraphProto& graph) {
    // ReshapePushConstants pushConstants;

    auto inputName = node.input(0);
    auto shapeName = node.input(1);

    auto inputDim = getTensorDimensions(inputName, name2Value, graph);
    auto shapeDim = getTensorDimensions(shapeName, name2Value, graph);

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
                                       const std::map<std::string, const onnx::ValueInfoProto*>& name2Value,
                                       const onnx::GraphProto& graph) {
    TransposePushConstants pushConstants;

    // Parse the perm attribute
    parseAttributes<uint32_t>(node, "perm", pushConstants.dimPerm);

    std::string shaderFilename = "shaders/onnx/transpose.comp.spv";

    auto operation = vulkanContext->create<GeneralComputation<TransposePushConstants>>(shaderFilename);
    operation->setPushConstants({pushConstants});

    return operation;
}

ComputeGraphElementPtr createTensorOperation(VulkanContext* vulkanContext, const onnx::NodeProto& node,
                                             const std::map<std::string, const onnx::ValueInfoProto*>& name2Value,
                                             const onnx::GraphProto& graph) {
    auto output = node.output();
    auto x = output.size();
    // Create a compute operation for each node
    std::string operationType = node.op_type();
    std::cout << "Creating operation for node: " << node.name() << " of type: " << operationType << std::endl;

    ComputeGraphElementPtr operation;

    if (operationType == "Conv") {
        operation = createConv(vulkanContext, node, name2Value, graph);
    } else if (operationType == "Relu") {
        operation = createRelu(vulkanContext, node, name2Value, graph);
    } else if (operationType == "Constant") {
        operation = createConstant(vulkanContext, node, name2Value, graph);
    } else if (operationType == "Reshape") {
        operation = createReshape(vulkanContext, node, name2Value, graph);
    } else if (operationType == "Transpose") {
        operation = createTranspose(vulkanContext, node, name2Value, graph);
    } else {
        throw std::runtime_error("Unsupported operation type: " + operationType);
    }

    return operation;
}

} // namespace klartraum
