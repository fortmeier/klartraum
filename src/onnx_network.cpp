#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>

#include "klartraum/computegraph/copybuffer.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/noop.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/onnx/onnx_network.hpp"
#include "klartraum/onnx/onnx_push_constants.hpp"
#include "klartraum/onnx/onnx_operations.hpp"

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

std::shared_ptr<TensorElementInterface> createTensor(VulkanContext* vulkanContext, const TensorInfo& tensorInfo) {
    // TODO we use VK_BUFFER_USAGE_TRANSFER_SRC_BIT for all buffers for now, might be not optimal
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (tensorInfo.dataType == onnx::TensorProto::FLOAT) {
        return vulkanContext->create<TensorElement<float>>(tensorInfo.shape, usage);
    } else if (tensorInfo.dataType == onnx::TensorProto::DOUBLE) {
        return vulkanContext->create<TensorElement<double>>(tensorInfo.shape, usage);
    } else if (tensorInfo.dataType == onnx::TensorProto::INT32) {
        return vulkanContext->create<TensorElement<int32_t>>(tensorInfo.shape, usage);
    } else if (tensorInfo.dataType == onnx::TensorProto::INT64) {
        return vulkanContext->create<TensorElement<int64_t>>(tensorInfo.shape, usage);
    }

    throw std::runtime_error("Unsupported data type: " + std::to_string(tensorInfo.dataType));
}

std::shared_ptr<TensorElementInterface> createConstantTensor(VulkanContext* vulkanContext, const TensorInfo& tensorInfo) {
    // TODO we use VK_BUFFER_USAGE_TRANSFER_SRC_BIT for all buffers for now, might be not optimal
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    if (tensorInfo.dataType == onnx::TensorProto::FLOAT) {
        return vulkanContext->create<TensorElementSinglePath<float>>(tensorInfo.shape, usage);
    } else if (tensorInfo.dataType == onnx::TensorProto::DOUBLE) {
        return vulkanContext->create<TensorElementSinglePath<double>>(tensorInfo.shape, usage);
    } else if (tensorInfo.dataType == onnx::TensorProto::INT32) {
        return vulkanContext->create<TensorElementSinglePath<int32_t>>(tensorInfo.shape, usage);
    } else if (tensorInfo.dataType == onnx::TensorProto::INT64) {
        return vulkanContext->create<TensorElementSinglePath<int64_t>>(tensorInfo.shape, usage);
    }

    throw std::runtime_error("Unsupported data type: " + std::to_string(tensorInfo.dataType));
}

onnx::TensorProto::DataType getTensorDataType(const onnx::TypeProto::Tensor& tensorType) {
    return static_cast<onnx::TensorProto::DataType>(tensorType.elem_type());
}

std::vector<uint32_t> getTensorShape(const onnx::TypeProto::Tensor& tensorType) {
    std::vector<uint32_t> shape;
    for (const auto& dim : tensorType.shape().dim()) {
        shape.push_back(dim.dim_value());
    }
    return shape;
}

std::map<std::string, ComputeGraphElementPtr> createTensorOperationOutputs(VulkanContext* vulkanContext, const onnx::NodeProto& node, TensorInfoMap& name2TensorInfo) {
    auto output = node.output();
    auto x = output.size();
    // Create a compute operation for each node
    std::string operationType = node.op_type();
    std::map<std::string, ComputeGraphElementPtr> outputs;

    if (operationType == "Conv") {
        std::string output0Name = node.output(0);

        auto tensorInfo = name2TensorInfo.at(output0Name);
        auto dataType = tensorInfo.dataType;

        std::shared_ptr<TensorElementInterface> output = createTensor(vulkanContext, tensorInfo);
        output->setName(output0Name);
        outputs[output0Name] = output;
        name2TensorInfo[output0Name] = tensorInfo;
    } else if (operationType == "ConvTranspose") {
        std::string output0Name = node.output(0);
        auto tensorInfo = name2TensorInfo.at(output0Name);
        auto dataType = tensorInfo.dataType;

        std::shared_ptr<TensorElementInterface> output = createTensor(vulkanContext, tensorInfo);
        output->setName(output0Name);
        outputs[output0Name] = output;
        name2TensorInfo[node.output(0)] = tensorInfo;
    } else if (operationType == "Relu" || operationType == "Reshape" || operationType == "Transpose") {
        std::string output0Name = node.output(0);

        auto tensorInfo = name2TensorInfo.at(output0Name);
        auto dataType = tensorInfo.dataType;

        std::shared_ptr<TensorElementInterface> output = createTensor(vulkanContext, tensorInfo);
        output->setName(output0Name);
        outputs[output0Name] = output;
        name2TensorInfo[output0Name] = tensorInfo; // Store the output type for this operation
    } else if (operationType == "Constant") {
        TensorInfo tensorInfo;
        tensorInfo.shape = {1, 1, 1, 1}; // Default shape

        for (int i = 0; i < node.attribute_size(); i++) {
            const auto& attr = node.attribute(i);
            if (attr.name() == "value" && attr.has_t()) {
                auto data = attr.t();
                tensorInfo.dataType = static_cast<onnx::TensorProto::DataType>(data.data_type());
            }
        }

        std::shared_ptr<TensorElementInterface> output = createConstantTensor(vulkanContext, tensorInfo);
        output->setName(node.output(0));
        outputs[node.output(0)] = output;
        name2TensorInfo[node.output(0)] = tensorInfo; // Store the output type for this operation
    } else {
        throw std::runtime_error("Unsupported operation type: " + operationType);
    }

    return outputs;
}

void OnnxNetwork::createInitializerTensor(const onnx::TensorProto* initializer,
    TensorInfoMap& name2TensorInfo,
    VulkanContext* vulkanContext)
{
    std::cout << "Creating initializer tensor: " << initializer->name() << std::endl;
    std::vector<uint32_t> initShape;
    for (int j = 0; j < initializer->dims_size(); j++) {
        initShape.push_back(initializer->dims(j));
    }
    const auto& dataType = static_cast<onnx::TensorProto::DataType>(initializer->data_type());

    TensorInfo tensorInfo{dataType, initShape};
    name2TensorInfo[initializer->name()] = tensorInfo;

    auto tensor = createConstantTensor(vulkanContext, tensorInfo);
    tensor->setName(initializer->name());
    graphDataElements[initializer->name()] = tensor;

    //std::cout << " - size: " << (initShape.empty() ? 0 : std::accumulate(initShape.begin(), initShape.end(), 1, std::multiplies<uint32_t>())) << " elements" << std::endl;
}

void OnnxNetwork::createInfoTensor(const onnx::ValueInfoProto* input,
    TensorInfoMap& name2TensorInfo,
    VulkanContext* vulkanContext)
{
    std::cout << "Creating input tensor: " << input->name() << std::endl;
    
    if (input->has_type() && input->type().has_tensor_type()) {
        std::vector<uint32_t> inputShape;
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
            onnx::TensorProto::DataType dataType = getTensorDataType(tensor_type);
            std::cout << "Data type: " << dataType << " ";
            TensorInfo tensorInfo{dataType, inputShape};
            name2TensorInfo[input->name()] = tensorInfo;

            auto tensor = createConstantTensor(vulkanContext, tensorInfo);
            tensor->setName(input->name());
            graphDataElements[input->name()] = tensor;

            std::cout << " - size: " << tensorSize << " elements" << std::endl;
        } else {
            std::cout << "Unsupported input shape size: " << inputShape.size() << std::endl;
            throw std::runtime_error("Unsupported input shape size for tensor: " + input->name());
        }
    }
}

void OnnxNetwork::createComputeGraph() {
    /**
     * ONNX models can come in different flavors. By default, the size of tensors
     * is determined by the model's input and output specifications and all intermediate tensor sizes
     * can be inferred from these. When going through the network graph by going through the list
     * of graph nodes (graph->node(i)), there is only a link to intermediate tensors by their names.
     * Thus, either the intermediate tensor sizes need to be explicitly defined in the model, or
     * they need to be inferred during the graph traversal.
     * For starters, we assume that all necessary intermediate tensors, weights, and biases are available
     * in the model's value infos. This has to be made sure during the model export.
     * For the example autoencoder, see scripts\onnx\train_simple_autoencoder.ipynb on how this can be achieved.
     */
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

    // WTF why do we have the output elements here?, they will be generated together with the operations
    for (int i = 0; i < graph.output_size(); ++i) {
        infos.push_back(&graph.output(i));
    }

    // WTF are actually the value infos, is that need?
    for (int i = 0; i < graph.value_info_size(); ++i) {
        infos.push_back(&graph.value_info(i));
    }

    std::vector<const onnx::TensorProto*> initializers;
    for (int i = 0; i < graph.initializer_size(); ++i) {
        const onnx::TensorProto& initializer = graph.initializer(i);
        auto name = initializer.name();
        std::cout << "Initializer tensor: " << name << std::endl;
        initializers.push_back(&initializer);
    }

    // Print all info tensor names
    std::cout << "All tensor names in the graph:" << std::endl;
    for (const auto* info : infos) {
        std::cout << "  - " << info->name() << std::endl;
    }
    std::cout << std::endl;

    // Print all initializer tensor names
    std::cout << "All initializer tensor names in the graph:" << std::endl;
    for (const auto* initializer : initializers) {
        std::cout << "  - " << initializer->name() << std::endl;
    }

    // Start with creating info and initializer tensors
    std::cout << "Creating input tensors:" << std::endl;

    // create info buffer tensors
    for (const auto& input : infos) {
        createInfoTensor(input, name2TensorInfo, vulkanContext);
        name2ValueInfoProto[input->name()] = input;
    }

    // create initializer buffer tensors
    // for (const auto& initializer : initializers) {
    //     createInitializerTensor(initializer, name2TensorInfo, vulkanContext);
    // }

    // to create operations, first go through all ONNX nodes and
    // create their corresponding klartraum graph elements
    // create all operations
    createGraphElementsFromNodes();

    // then, for each output tensor associated to a node,
    // create klartraum graph elements
    createGraphElementsFromOutputTensors();

    // finally, connect all operation inputs and outputs
    // in the klartraum compute graph, both inputs and outputs pass
    // through the compute element in the same fashion
    connectGraphElements();

    // get all output names
    storeComputeGraphGroupOutputElements();
}

void OnnxNetwork::createGraphElementsFromNodes()
{
    const onnx::GraphProto& graph = model->graph();

    for (int i = 0; i < graph.node_size(); i++) {
        const onnx::NodeProto& node = graph.node(i);
        auto name = node.name();
        std::cout << "Creating operation for node: " << name << std::endl;
        auto operation = createTensorOperation(vulkanContext, node, name2ValueInfoProto, graph);
        std::string operationName = node.op_type() + "_" + std::to_string(i) + "_" + name;
        operation->setName(operationName);
        graphOperationElements[i] = operation;
    }
}

void OnnxNetwork::createGraphElementsFromOutputTensors()
{
    const onnx::GraphProto& graph = model->graph();

    for (int i = 0; i < graph.node_size(); i++) {
        const onnx::NodeProto& node = graph.node(i);
        std::map<std::string, ComputeGraphElementPtr> outputs = createTensorOperationOutputs(vulkanContext, node, name2TensorInfo);
        std::cout << " - Created operation outputs for node: " << node.name() << std::endl;
        // TODO WARNING BUG? ARE MAPS ALWAYS INSERTION ORDERD???
        // slots start just one after the slots of the inputs
        int slot = node.input_size();
        for (const auto& [name, output] : outputs) {
            std::cout << "   - Output " << name << ": " << output << std::endl;
            graphDataElements[name] = output;
            outputName2GraphElementAndSlot[name] = std::make_pair(graphOperationElements[i], slot);
            slot++;
        }
    }
}

void OnnxNetwork::connectGraphElements()
{
    /*
     * Go through all operation compute graph elements that have been created from the ONNX graph nodes
     * and connect them to their inputs and outputs.
     * Both inputs and outputs pass through the compute element in the same fashion so
     * first the inputs are set via setInput followed by the outputs.
     */
    const onnx::GraphProto& graph = model->graph();

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
}

void OnnxNetwork::storeComputeGraphGroupOutputElements()
{
    /**
     * OnnxNetwork is a klartraum ComputeGraphGroup and thus can have
     * multiple output elements.
     * This method collects all output elements from the ONNX graph and
     * adds them to the outputs of the compute graph group.
     */
    const onnx::GraphProto& graph = model->graph();

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

    if (!model->has_graph()) {
        std::cerr << "OnnxNetwork: Error - Model has no graph" << std::endl;
        return;
    }

    const onnx::GraphProto& graph = model->graph();

    // create all initializers
    for (int i = 0; i < graph.initializer_size(); ++i) {
        const auto& init = graph.initializer(i);
        std::string name = init.name();
        std::cout << "Creating initializer tensor: " << name << std::endl;
        std::vector<uint32_t> initShape;
        for (int j = 0; j < init.dims_size(); j++) {
            initShape.push_back(init.dims(j));
        }
        const auto& dataType = static_cast<onnx::TensorProto::DataType>(init.data_type());

        auto initData = init.raw_data(); // This is where the actual data would be, if needed
        if (!initData.empty()) {
            switch (dataType) {
            case onnx::TensorProto::FLOAT:
                std::cout << " - Data type: FLOAT" << std::endl;
                // Copy the data into the initializer tensor
                auto element = graphDataElements[name];
                std::shared_ptr<TensorElementSinglePath<float>> elementPtr = std::dynamic_pointer_cast<TensorElementSinglePath<float>>(element);
                TensorElementSinglePath<float>* tensor = elementPtr.get();
                tensor->getDataBuffer().memcopyFrom(initData.data(), initData.size());
                break;
                // TODO support other data types
            }
        }
        else if (init.float_data_size() > 0) {
            std::cout << " - Data type: FLOAT (float_data field)" << std::endl;
            auto element = graphDataElements[name];
            std::shared_ptr<TensorElementSinglePath<float>> elementPtrSingle = std::dynamic_pointer_cast<TensorElementSinglePath<float>>(element);
            std::shared_ptr<TensorElement<float>> elementPtrMulti = std::dynamic_pointer_cast<TensorElement<float>>(element);
            if (elementPtrSingle) {
                TensorElementSinglePath<float>* tensor = elementPtrSingle.get();
                tensor->getDataBuffer().memcopyFrom(init.float_data().data(), init.float_data_size());
            } else if (elementPtrMulti) {
                TensorElement<float>* tensor = elementPtrMulti.get();
                tensor->getDataBuffer(0).memcopyFrom(init.float_data().data(), init.float_data_size());
            } else {
                throw std::runtime_error("Initializer " + name + " has unsupported tensor element type");
            }
        } else {
            throw std::runtime_error("Initializer " + name + " has no data");
        }
    }
    // fill all constant tensors
    for (int i = 0; i < graph.node_size(); i++) {
        const onnx::NodeProto& node = graph.node(i);
        std::string op_type = node.op_type();
        if (op_type == "Constant") {
            for (int i = 0; i < node.attribute_size(); ++i) {
                const auto& attr = node.attribute(i);
                if (attr.name() == "value" && attr.has_t()) {
                    auto data = attr.t();
                    auto type = static_cast<onnx::TensorProto::DataType>(data.data_type());
                    size_t dataSize = data.raw_data().size();
                    const char* dataLocation = data.raw_data().data();
                    if (type == onnx::TensorProto::FLOAT) {
                        std::shared_ptr<TensorElementSinglePath<float>> tensorElement = std::dynamic_pointer_cast<TensorElementSinglePath<float>>(graphDataElements[node.output(0)]);
                        std::cout << "copy values for constant node " << node.name() << " of size " << dataSize << std::endl;
                        tensorElement->getDataBuffer().memcopyFrom(dataLocation, dataSize);
                    } else if (type == onnx::TensorProto::INT32) {
                        std::shared_ptr<TensorElementSinglePath<int32_t>> tensorElement = std::dynamic_pointer_cast<TensorElementSinglePath<int32_t>>(graphDataElements[node.output(0)]);
                        std::cout << "copy values for constant node " << node.name() << " of size " << dataSize << std::endl;
                        tensorElement->getDataBuffer().memcopyFrom(dataLocation, dataSize);
                    } else if (type == onnx::TensorProto::INT64) {
                        std::shared_ptr<TensorElementSinglePath<int64_t>> tensorElement = std::dynamic_pointer_cast<TensorElementSinglePath<int64_t>>(graphDataElements[node.output(0)]);
                        std::cout << "copy values for constant node " << node.name() << " of size " << dataSize << std::endl;
                        tensorElement->getDataBuffer().memcopyFrom(dataLocation, dataSize);
                    } else {
                        std::cerr << "Unsupported constant data type: " << type << std::endl;
                        throw std::runtime_error("Unsupported constant data type: " + std::to_string(type));
                    }
                }
            }
        }
    }
}

void OnnxNetwork::_record(VkCommandBuffer commandBuffer, uint32_t pathId) {
    std::cout << "OnnxNetwork: Recording commands for path " << pathId << std::endl;

    // Record compute shader dispatches (placeholder implementation)
    // In practice, this would record the actual Vulkan commands for neural network execution
}

std::vector<float> OnnxNetwork::getFloatInitializerData(const std::string& name) const {
    /**
     * @brief this method retrieves the data associated to a given (tensor) initializer
     * it looks up the initializer by name in the model's graph initializers
     * and returns the data as a vector of floats
     */
    auto graph = model->graph();
    for (int i = 0; i < graph.initializer_size(); i++) {
        const auto& init = graph.initializer(i);
        std::string initName = init.name();
        if (initName == name) {
            if (init.data_type() != onnx::TensorProto::FLOAT) {
                throw std::runtime_error("Initializer " + name + " is not of type FLOAT");
            }
            if (init.has_raw_data()) {
                return std::vector<float>(reinterpret_cast<const float*>(init.raw_data().data()),
                                           reinterpret_cast<const float*>(init.raw_data().data()) + (init.raw_data().size() / sizeof(float)));
            } else if (init.float_data_size() > 0) {
                // Data is stored in float_data field
                const float* data = init.float_data().data();
                return std::vector<float>(data, data + init.float_data_size());
            } else {
                throw std::runtime_error("Initializer " + name + " has no data");
            }
        }
    }
    throw std::runtime_error("Initializer " + name + " not found");
}

ComputeGraphElementPtr OnnxNetwork::getOutputElement(const std::string& name) const{
    /**
     * this methods returns the compute graph element that is associated with the given output name
     * it looks up the output name in the map outputName2GraphElementAndSlot
     * which contains the graph elements that compute the output tensors
     * and thus the respective input slot has to be used to get the output tensor graph element
     */
    auto it = outputName2GraphElementAndSlot.find(name);
    if (it != outputName2GraphElementAndSlot.end()) {
        auto slot = it->second.second;
        auto element = it->second.first;
        auto outputElement = element->getInputElement(slot);
        return outputElement;
    }
    throw std::runtime_error("Output element " + name + " not found");
}

} // namespace klartraum
