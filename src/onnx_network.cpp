#include "klartraum/onnx_network.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>

#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/noop.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "onnx.pb.h"

namespace klartraum {

OnnxNetwork::OnnxNetwork(VulkanContext& vulkanContext, const std::string& modelPath)
    : vulkanContext(&vulkanContext), modelPath(modelPath) {

    std::cout << "OnnxNetwork: Initializing with model: " << modelPath << std::endl;

    loadModel(modelPath);
    createComputeGraph();
}

OnnxNetwork::~OnnxNetwork() {
}

bool OnnxNetwork::loadModel(const std::string& modelPath) {
    std::cout << "OnnxNetwork: Loading model from " << modelPath << std::endl;

    // Open and read the ONNX file
    std::ifstream input(modelPath, std::ios::binary);
    if (!input.is_open()) {
        std::cerr << "OnnxNetwork: Error - Could not open file " << modelPath << std::endl;
        return false;
    }

    // Create model instance
    model = std::make_unique<onnx::ModelProto>();

    // Parse the protobuf from the file
    if (!model->ParseFromIstream(&input)) {
        std::cerr << "OnnxNetwork: Error - Failed to parse ONNX model from " << modelPath << std::endl;
        model.reset();
        return false;
    }

    input.close();

    std::cout << "OnnxNetwork: Model loaded successfully" << std::endl;

    return true;
}

void OnnxNetwork::printModelInfo() const {
    if (!model) {
        std::cout << "OnnxNetwork: No model loaded" << std::endl;
        return;
    }

    std::cout << "\n=== ONNX Model Information ===" << std::endl;

    // IR version
    if (model->has_ir_version()) {
        std::cout << "IR Version: " << model->ir_version() << std::endl;
    }

    // Producer name and version
    if (model->has_producer_name()) {
        std::cout << "Producer: " << model->producer_name();
        if (model->has_producer_version()) {
            std::cout << " v" << model->producer_version();
        }
        std::cout << std::endl;
    }

    // Model version
    if (model->has_model_version()) {
        std::cout << "Model Version: " << model->model_version() << std::endl;
    }

    // Domain
    if (model->has_domain()) {
        std::cout << "Domain: " << model->domain() << std::endl;
    }

    // Graph information
    if (model->has_graph()) {
        const onnx::GraphProto& graph = model->graph();
        std::cout << "Graph name: " << graph.name() << std::endl;
        std::cout << "Inputs: " << graph.input_size() << std::endl;
        std::cout << "Outputs: " << graph.output_size() << std::endl;
        std::cout << "Nodes: " << graph.node_size() << std::endl;
        std::cout << "Initializers: " << graph.initializer_size() << std::endl;

        // Print input information
        std::cout << "\nInput tensors:" << std::endl;
        for (int i = 0; i < graph.input_size(); ++i) {
            const onnx::ValueInfoProto& input = graph.input(i);
            std::cout << "  " << i << ": " << input.name();
            if (input.has_type() && input.type().has_tensor_type()) {
                const onnx::TypeProto::Tensor& tensor_type = input.type().tensor_type();
                if (tensor_type.has_elem_type()) {
                    std::cout << " (type: " << tensor_type.elem_type() << ")";
                }
                if (tensor_type.has_shape()) {
                    std::cout << " shape: [";
                    for (int j = 0; j < tensor_type.shape().dim_size(); ++j) {
                        if (j > 0)
                            std::cout << ", ";
                        const onnx::TensorShapeProto::Dimension& dim = tensor_type.shape().dim(j);
                        if (dim.has_dim_value()) {
                            std::cout << dim.dim_value();
                        } else if (dim.has_dim_param()) {
                            std::cout << dim.dim_param();
                        } else {
                            std::cout << "?";
                        }
                    }
                    std::cout << "]";
                }
            }
            std::cout << std::endl;
        }

        // Print output information
        std::cout << "\nOutput tensors:" << std::endl;
        for (int i = 0; i < graph.output_size(); ++i) {
            const onnx::ValueInfoProto& output = graph.output(i);
            std::cout << "  " << i << ": " << output.name();
            if (output.has_type() && output.type().has_tensor_type()) {
                const onnx::TypeProto::Tensor& tensor_type = output.type().tensor_type();
                if (tensor_type.has_elem_type()) {
                    std::cout << " (type: " << tensor_type.elem_type() << ")";
                }
                if (tensor_type.has_shape()) {
                    std::cout << " shape: [";
                    for (int j = 0; j < tensor_type.shape().dim_size(); ++j) {
                        if (j > 0)
                            std::cout << ", ";
                        const onnx::TensorShapeProto::Dimension& dim = tensor_type.shape().dim(j);
                        if (dim.has_dim_value()) {
                            std::cout << dim.dim_value();
                        } else if (dim.has_dim_param()) {
                            std::cout << dim.dim_param();
                        } else {
                            std::cout << "?";
                        }
                    }
                    std::cout << "]";
                }
            }
            std::cout << std::endl;
        }

        // Print first few node operations
        std::cout << "\nFirst few operations:" << std::endl;
        int nodes_to_show = std::min(15, graph.node_size());
        for (int i = 0; i < nodes_to_show; ++i) {
            const onnx::NodeProto& node = graph.node(i);
            std::cout << "  " << i << ": " << node.op_type();
            if (node.has_name() && !node.name().empty()) {
                std::cout << " (" << node.name() << ")";
            }
            std::cout << " - inputs: " << node.input_size() << ", outputs: " << node.output_size() << std::endl;
        }
        if (graph.node_size() > nodes_to_show) {
            std::cout << "  ... and " << (graph.node_size() - nodes_to_show) << " more nodes" << std::endl;
        }
    }
}

struct TensorOpPushConstants {
    uint32_t batch_a;
    uint32_t depth_a;
    uint32_t height_a;
    uint32_t width_a;

    uint32_t batch_b;
    uint32_t depth_b;
    uint32_t height_b;
    uint32_t width_b;
};

std::shared_ptr<TensorElementInterface> createTensorWithType(VulkanContext* vulkanContext, const onnx::TensorProto::DataType dataType, std::vector<uint32_t> inputShape) {
    if (dataType == onnx::TensorProto::FLOAT) {
        return vulkanContext->create<TensorElement<float>>(inputShape);
    } else if (dataType == onnx::TensorProto::DOUBLE) {
        return vulkanContext->create<TensorElement<double>>(inputShape);
    } else if (dataType == onnx::TensorProto::INT32) {
        return vulkanContext->create<TensorElement<int32_t>>(inputShape);
    } else if (dataType == onnx::TensorProto::INT64) {
        return vulkanContext->create<TensorElement<int64_t>>(inputShape);
    }

    throw std::runtime_error("Unsupported data type: " + std::to_string(dataType));
}

ComputeGraphElementPtr createConv(VulkanContext* vulkanContext, const onnx::NodeProto& node) {
    TensorOpPushConstants pushConstants;
    pushConstants.depth_a = 1;
    pushConstants.height_a = 1;
    pushConstants.width_a = 1;
    pushConstants.depth_b = 1;
    pushConstants.height_b = 1;
    pushConstants.width_b = 1;

    std::string shaderFilename = "shaders/onnx/conv.comp.spv";

    auto operation = vulkanContext->create<GeneralComputation<TensorOpPushConstants>>(shaderFilename);
    operation->setPushConstants({pushConstants});

    return operation;
}

ComputeGraphElementPtr createRelu(VulkanContext* vulkanContext, const onnx::NodeProto& node) {
    TensorOpPushConstants pushConstants;
    pushConstants.depth_a = 1;
    pushConstants.height_a = 1;
    pushConstants.width_a = 1;
    pushConstants.depth_b = 1;
    pushConstants.height_b = 1;
    pushConstants.width_b = 1;

    std::string shaderFilename = "shaders/onnx/relu.comp.spv";

    auto operation = vulkanContext->create<GeneralComputation<TensorOpPushConstants>>(shaderFilename);
    operation->setPushConstants({pushConstants});

    return operation;
}

ComputeGraphElementPtr createConstant(VulkanContext* vulkanContext, const onnx::NodeProto& node) {
    // Constant operations don't perform computation - they just provide constant data
    // Use NoOp to pass the constant data through without any GPU operations
    auto operation = vulkanContext->create<NoOp>();

    return operation;
}

ComputeGraphElementPtr createReshape(VulkanContext* vulkanContext, const onnx::NodeProto& node) {
    TensorOpPushConstants pushConstants;
    pushConstants.depth_a = 1;
    pushConstants.height_a = 1;
    pushConstants.width_a = 1;
    pushConstants.depth_b = 1;
    pushConstants.height_b = 1;
    pushConstants.width_b = 1;

    std::string shaderFilename = "shaders/onnx/reshape.comp.spv";

    auto operation = vulkanContext->create<GeneralComputation<TensorOpPushConstants>>(shaderFilename);
    operation->setPushConstants({pushConstants});

    return operation;
}

ComputeGraphElementPtr createTranspose(VulkanContext* vulkanContext, const onnx::NodeProto& node) {
    TensorOpPushConstants pushConstants;
    pushConstants.depth_a = 1;
    pushConstants.height_a = 1;
    pushConstants.width_a = 1;
    pushConstants.depth_b = 1;
    pushConstants.height_b = 1;
    pushConstants.width_b = 1;

    std::string shaderFilename = "shaders/onnx/transpose.comp.spv";

    auto operation = vulkanContext->create<GeneralComputation<TensorOpPushConstants>>(shaderFilename);
    operation->setPushConstants({pushConstants});

    return operation;
}

ComputeGraphElementPtr createTensorOperation(VulkanContext* vulkanContext, const onnx::NodeProto& node) {
    auto output = node.output();
    auto x = output.size();
    // Create a compute operation for each node
    std::string operationType = node.op_type();
    std::cout << "Creating operation for node: " << node.name() << " of type: " << operationType << std::endl;

    ComputeGraphElementPtr operation;

    if (operationType == "Conv") {
        operation = createConv(vulkanContext, node);
    } else if (operationType == "Relu") {
        operation = createRelu(vulkanContext, node);
    } else if (operationType == "Constant") {
        operation = createConstant(vulkanContext, node);
    } else if (operationType == "Reshape") {
        operation = createReshape(vulkanContext, node);
    } else if (operationType == "Transpose") {
        operation = createTranspose(vulkanContext, node);
    } else {
        throw std::runtime_error("Unsupported operation type: " + operationType);
    }

    return operation;
}

std::map<std::string, ComputeGraphElementPtr> createTensorOperationOutputs(VulkanContext* vulkanContext, const onnx::NodeProto& node, std::map<std::string, onnx::TensorProto::DataType>& name2Type) {
    auto output = node.output();
    auto x = output.size();
    // Create a compute operation for each node
    std::string operationType = node.op_type();
    std::map<std::string, ComputeGraphElementPtr> outputs;

    if (operationType == "Conv") {
        std::string input0Name = node.input(0);
        auto input0 = name2Type.at(input0Name);
        auto dataType = input0;

        std::vector<uint32_t> inputShape = {1, 1, 1, 1}; // Default shape
        std::shared_ptr<TensorElementInterface> output = createTensorWithType(vulkanContext, dataType, inputShape);
        output->setName(node.output(0));
        outputs[node.output(0)] = output;
        name2Type[node.output(0)] = dataType; // Store the output type for this operation
    } else if (operationType == "Relu" || operationType == "Reshape" || operationType == "Transpose") {
        std::vector<uint32_t> inputShape = {1, 1, 1, 1}; // Default shape
        std::string input0Name = node.input(0);
        auto input0 = name2Type.at(input0Name);
        auto dataType = input0;

        std::shared_ptr<TensorElementInterface> output = createTensorWithType(vulkanContext, dataType, inputShape);
        output->setName(node.output(0));
        outputs[node.output(0)] = output;
        name2Type[node.output(0)] = dataType; // Store the output type for this operation
    } else if (operationType == "Constant") {
        std::vector<uint32_t> inputShape = {1, 1, 1, 1}; // Default shape

        // TODO deterimine dynamically
        auto dataType = onnx::TensorProto::FLOAT;

        std::shared_ptr<TensorElementInterface> output = createTensorWithType(vulkanContext, dataType, inputShape);
        output->setName(node.output(0));
        outputs[node.output(0)] = output;
        name2Type[node.output(0)] = dataType; // Store the output type for this operation
    } else {
        throw std::runtime_error("Unsupported operation type: " + operationType);
    }

    return outputs;
}

void OnnxNetwork::createComputeGraph() {
    std::cout << "OnnxNetwork: Creating compute graph from ONNX model" << std::endl;

    if (!model) {
        std::cerr << "OnnxNetwork: Error - No model loaded for compute graph creation" << std::endl;
        return;
    }

    if (!model->has_graph()) {
        std::cerr << "OnnxNetwork: Error - Model has no graph" << std::endl;
        return;
    }

    const onnx::GraphProto& graph = model->graph();

    std::cout << "OnnxNetwork: Analyzing " << graph.node_size() << " operations for compute graph creation" << std::endl;

    // TODO: Implement actual compute graph creation
    // This would involve:
    // 1. Parse ONNX operations and convert to Vulkan compute operations
    // 2. Create buffer allocations for tensors
    // 3. Set up compute pipeline stages
    // 4. Handle data dependencies between operations

    std::vector<const onnx::ValueInfoProto*> infos;
    for (int i = 0; i < graph.input_size(); ++i) {
        infos.push_back(&graph.input(i));
    }

    for (int i = 0; i < graph.output_size(); ++i) {
        infos.push_back(&graph.output(i));
    }

    for (int i = 0; i < graph.value_info_size(); ++i) {
        infos.push_back(&graph.value_info(i));
    }

    // Print all tensor names
    std::cout << "All tensor names in the graph:" << std::endl;
    for (const auto* info : infos) {
        std::cout << "  - " << info->name() << std::endl;
    }
    std::cout << std::endl;

    std::map<std::string, const onnx::ValueInfoProto*> name2Value;
    std::map<std::string, onnx::TensorProto::DataType> name2Type;
    std::map<std::string, std::pair<ComputeGraphElementPtr, int>> outputName2GraphElementAndSlot;

    std::cout << "Creating input tensors:" << std::endl;

    // create input buffer tensors
    for (const auto& input : infos) {
        std::cout << "Creating input tensor: " << input->name() << std::endl;
        name2Value[input->name()] = input;
        if (input->has_type() && input->type().has_tensor_type()) {
            std::vector<uint32_t> inputShape;
            // auto inputTensor = vulkanContext->create<BufferElement<VulkanBuffer<uint32_t>>>(1);

            const onnx::TypeProto::Tensor& tensor_type = input->type().tensor_type();
            if (tensor_type.has_elem_type()) {
                std::cout << " (type: " << tensor_type.elem_type() << ")";
            }
            if (tensor_type.has_shape()) {
                std::cout << " shape: [";
                for (int j = 0; j < tensor_type.shape().dim_size(); ++j) {
                    const auto& dim = tensor_type.shape().dim(j);
                    if (dim.has_dim_value()) {
                        std::cout << dim.dim_value();
                        inputShape.push_back(dim.dim_value());
                    } else if (dim.has_dim_param()) {
                        // set dim = 1 for dynamic dimensions
                        // TODO this is a placeholder, should handle dynamic dimensions properly
                        inputShape.push_back(1);
                        std::cout << dim.dim_param();
                    } else {
                        std::cout << "?";
                    }
                    std::cout << " ";
                }
                std::cout << "]";
            }
            std::cout << std::endl;
            size_t tensorSize = 1; // Calculate based on input shape
            for (const auto& dim : inputShape) {
                tensorSize *= dim;
            }
            if (true) { // inputShape.size() == 4) {
                onnx::TensorProto::DataType dataType = static_cast<onnx::TensorProto::DataType>(tensor_type.elem_type());
                std::cout << "Data type: " << dataType << " ";
                name2Type[input->name()] = dataType;
                auto tensor = createTensorWithType(vulkanContext, dataType, inputShape);
                tensor->setName(input->name());
                graphDataElements[input->name()] = tensor;

                std::cout << " - size: " << tensorSize << " elements" << std::endl;
            } else {
                std::cout << "Unsupported input shape size: " << inputShape.size() << std::endl;
                throw std::runtime_error("Unsupported input shape size for tensor: " + input->name());
            }
        }
    }

    // // create all initializers
    // for (int i = 0; i < graph.initializer_size(); ++i) {
    //     const auto& init = graph.initializer(i);
    //     std::string name = init.name();
    //     std::cout << "Creating initializer tensor: " << name << std::endl;
    //     std::vector<uint32_t> initShape;
    //     for (int j = 0; j < init.dims_size(); j++)
    //     {
    //         initShape.push_back(init.dims(j));
    //     }
    //     const auto& dataType = static_cast<onnx::TensorProto::DataType>(init.data_type());

    //     graphDataElements[name] = createTensorWithType(vulkanContext, dataType, initShape);
    // }

    // to create operations, first go through all nodes and create their operation elements
    // create all operations
    for (int i = 0; i < graph.node_size(); i++) {
        const onnx::NodeProto& node = graph.node(i);
        auto name = node.name();
        std::cout << "Creating operation for node: " << name << std::endl;
        auto operation = createTensorOperation(vulkanContext, node);
        std::string operationName = node.op_type() + "_" + std::to_string(i) + "_" + name;
        operation->setName(operationName);
        graphOperationElements[i] = operation;

        // TODO attributes need to be handled also
        for (int j = 0; j < node.attribute_size(); j++) {
            const auto& attr = node.attribute(j);
            std::cout << " - Attribute " << j << ": " << attr.name() << " = " << attr.f() << std::endl;
        }
    }

    // create all operation outputs
    for (int i = 0; i < graph.node_size(); i++) {
        const onnx::NodeProto& node = graph.node(i);
        std::map<std::string, ComputeGraphElementPtr> outputs = createTensorOperationOutputs(vulkanContext, node, name2Type);
        std::cout << " - Created operation outputs for node: " << node.name() << std::endl;
        // TODO WARNING BUG? ARE MAPS ALWAYS INSERTION ORDERD???
        int slot = 0;
        for (const auto& [name, output] : outputs) {
            std::cout << "   - Output " << name << ": " << output << std::endl;
            graphDataElements[name] = output;
            outputName2GraphElementAndSlot[name] = std::make_pair(graphOperationElements[i], slot);
            slot++;
        }
    }

    // finally, connect all operation inputs
    for (int i = 0; i < graph.node_size(); i++) {
        const onnx::NodeProto& node = graph.node(i);

        std::string name = node.name();

        ComputeGraphElementPtr operation = graphOperationElements[i];

        std::string op_type = node.op_type();
        std::cout << "connecting operation (" << i << "): \"" << name << "\" [" << op_type << "]" << std::endl;

        // Get inputs for this operation
        std::vector<std::string> inputNames;
        int startIndex = 0;
        for (int j = 0; j < node.input_size(); j++) {
            auto index = node.input(j);
            if (outputName2GraphElementAndSlot.find(index) == outputName2GraphElementAndSlot.end()) {
                // inde
                // we have a direct input, so no compute node but a tensor/buffer
                operation->setInput(graphDataElements.at(index), j);
                std::cout << " - Input from Data: \"" << index << "\" -> \"" << graphDataElements.at(index)->getName() << "\"" << std::endl;
            } else {
                auto [input, slot] = outputName2GraphElementAndSlot.at(index);
                operation->setInput(input, j, slot);
                std::cout << " - Input from Node " << j << ": \"" << index << "\" -> \"" << input->getName() << "\"" << std::endl;
            }
            startIndex = j + 1;
        }

        for (int j = 0; j < node.output_size(); j++) {
            auto index = node.output(j);
            auto output = graphDataElements.at(index);
            if (output == nullptr) {
                throw std::runtime_error("Output tensor is null for operation output " + std::to_string(j));
            }
            std::cout << " - Output " << j << ": \"" << index << "\" -> \"" << output->getName() << "\"" << std::endl;
            operation->setInput(output, j + startIndex);
        }
    }

    // get all output names
    std::vector<std::string> outputNames;
    for (int i = 0; i < graph.output_size(); ++i) {
        auto output = graph.output(i);
        auto name = output.name();
        outputNames.push_back(name);
    }

    // all nodes that have outputs will be added the outputs elements of
    // the compute graph group
    uint32_t outputElementIndex = 0;
    std::cout << "Output elements: " << std::endl;
    for (int i = 0; i < graph.node_size(); i++) {
        const onnx::NodeProto& node = graph.node(i);
        for (int j = 0; j < node.output_size(); j++) {
            auto outputName = node.output(j);
            // check if outputName is in output names
            if (std::find(outputNames.begin(), outputNames.end(), outputName) != outputNames.end()) {
                std::cout << " - Found output name: " << outputName << std::endl;
                outputElements[outputElementIndex] = graphOperationElements.at(i);
                std::cout << " - connected output " << outputName << " to operation " << i << std::endl;
                outputElementIndex++;
            }
        }
    }
}

// ComputeGraphGroup interface implementation
void OnnxNetwork::checkInput(ComputeGraphElementPtr input, int index) {
    // Placeholder implementation
    std::cout << "OnnxNetwork: Checking input " << index << std::endl;
}

void OnnxNetwork::_setup(VulkanContext& vulkanContext, uint32_t numberPaths) {
    this->vulkanContext = &vulkanContext;
    this->numberOfPaths = numberPaths;

    std::cout << "OnnxNetwork: Setting up for " << numberPaths << " paths" << std::endl;
}

void OnnxNetwork::_record(VkCommandBuffer commandBuffer, uint32_t pathId) {
    std::cout << "OnnxNetwork: Recording commands for path " << pathId << std::endl;

    // Record compute shader dispatches (placeholder implementation)
    // In practice, this would record the actual Vulkan commands for neural network execution
}

} // namespace klartraum
