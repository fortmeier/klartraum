#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>

#include "klartraum/onnx_network.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"
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

    // create input buffer tensors
    for (int i = 0; i < graph.input_size(); ++i) {
        const onnx::ValueInfoProto& input = graph.input(i);
        if (input.has_type() && input.type().has_tensor_type()) {
            std::vector<uint32_t> inputShape;
            // auto inputTensor = vulkanContext->create<BufferElement<VulkanBuffer<uint32_t>>>(1);

            const onnx::TypeProto::Tensor& tensor_type = input.type().tensor_type();
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
            if (inputShape.size() == 4) {
                onnx::TensorProto::DataType dataType = static_cast<onnx::TensorProto::DataType>(tensor_type.elem_type());
                std::cout << "Data type: " << dataType << " ";
                if (dataType == onnx::TensorProto::FLOAT) {
                    auto tensorElement = vulkanContext->create<TensorElement<float>>(inputShape);
                    graphInputElements[input.name()] = tensorElement;
                } else if (dataType == onnx::TensorProto::DOUBLE) {
                    auto tensorElement = vulkanContext->create<TensorElement<double>>(inputShape);
                    graphInputElements[input.name()] = tensorElement;
                } else if (dataType == onnx::TensorProto::INT32) {
                    auto tensorElement = vulkanContext->create<TensorElement<int32_t>>(inputShape);
                    graphInputElements[input.name()] = tensorElement;
                } else if (dataType == onnx::TensorProto::INT64) {
                    auto tensorElement = vulkanContext->create<TensorElement<int64_t>>(inputShape);
                    graphInputElements[input.name()] = tensorElement;
                } else {
                    throw std::runtime_error("Unsupported data type: " + std::to_string(dataType));
                }
                std::cout << " - size: " << tensorSize << " elements" << std::endl;
            }
        }
    }

    // create all operations
    for (int i = 0; i < graph.node_size(); i++) {
        const onnx::NodeProto& node = graph.node(i);
        // Create a compute operation for each node
        std::string operationType = node.op_type();
        std::string shaderFilename;
        if (operationType == "Conv") {
            shaderFilename = "shaders/onnx/conv.comp.spv"; // Example shader for convolution
        } else if (operationType == "Relu") {
            shaderFilename = "shaders/onnx/relu.comp.spv"; // Example shader for ReLU
        } else {
            throw std::runtime_error("Unsupported operation type: " + operationType);
        }
        TensorOpPushConstants pushConstants;
        pushConstants.depth_a = 1;
        pushConstants.height_a = 1;
        pushConstants.width_a = 1;
        pushConstants.depth_b = 1;
        pushConstants.height_b = 1;
        pushConstants.width_b = 1;

        auto operation = vulkanContext->create<GeneralComputation<TensorOpPushConstants>>(shaderFilename);
        operation->setPushConstants({pushConstants});

        // Get inputs for this operation
        std::vector<std::string> inputNames;
        for (int j = 0; j < node.input_size(); ++j) {
            auto input = graphInputElements[node.input(j)];
            if (!input) {
                throw std::runtime_error("Input tensor is null for operation input " + std::to_string(j));
            }
            operation->setInput(input, j);
        }

        graphOperationElements.push_back(operation);
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
