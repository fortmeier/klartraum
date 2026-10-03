#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>

#include "klartraum/computegraph/copybuffer.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/noop.hpp"
#include "klartraum/computegraph/tensor_memory_planner.hpp"
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

    // Attention exported by PyTorch is commonly represented as
    // MatMul(query, key) -> Softmax -> MatMul(probabilities, value). Keeping
    // those nodes separate materializes an O(sequence^2) score tensor. At the
    // 4096-token spatial resolution used by SD1.5 at 512x512, one such tensor
    // occupies 1 GiB and its naive dispatch can trip the Windows GPU watchdog.
    // Collapse the lossless three-node pattern before tensors are allocated.
    VkPhysicalDeviceSubgroupProperties subgroupProperties{};
    subgroupProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 deviceProperties{};
    deviceProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    deviceProperties.pNext = &subgroupProperties;
    vkGetPhysicalDeviceProperties2(vulkanContext->getPhysicalDevice(), &deviceProperties);
    const bool canFuseAttention = subgroupProperties.subgroupSize == 32 &&
        (subgroupProperties.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
        (subgroupProperties.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
    bool softmaxDefaultIsLastAxis = false;
    for (const auto& opset : model->opset_import()) {
        if (opset.domain().empty() || opset.domain() == "ai.onnx") {
            softmaxDefaultIsLastAxis = opset.version() >= 13;
        }
    }

    auto* graph = model->mutable_graph();
    std::map<std::string, size_t> consumerCounts;
    for (const auto& node : graph->node()) {
        for (const auto& name : node.input()) {
            if (!name.empty()) ++consumerCounts[name];
        }
    }
    std::vector<onnx::NodeProto> optimizedNodes;
    optimizedNodes.reserve(graph->node_size());
    size_t fusedAttentionCount = 0;
    for (int index = 0; index < graph->node_size();) {
        if (index + 2 < graph->node_size()) {
            const auto& score = graph->node(index);
            const auto& softmax = graph->node(index + 1);
            const auto& context = graph->node(index + 2);
            bool softmaxUsesLastAxis = softmaxDefaultIsLastAxis;
            for (const auto& attribute : softmax.attribute()) {
                if (attribute.name() == "axis") {
                    softmaxUsesLastAxis = attribute.i() == -1;
                }
            }
            if (canFuseAttention && softmaxUsesLastAxis &&
                score.op_type() == "MatMul" && softmax.op_type() == "Softmax" &&
                context.op_type() == "MatMul" && score.input_size() == 2 &&
                score.output_size() == 1 && softmax.input_size() == 1 &&
                softmax.output_size() == 1 && context.input_size() == 2 &&
                context.output_size() == 1 &&
                softmax.input(0) == score.output(0) &&
                context.input(0) == softmax.output(0) &&
                consumerCounts[score.output(0)] == 1 &&
                consumerCounts[softmax.output(0)] == 1) {
                onnx::NodeProto fused = context;
                fused.set_op_type("FusedAttention");
                fused.set_name(context.name() + "/KlartraumFusedAttention");
                fused.clear_input();
                fused.add_input(score.input(0));
                fused.add_input(score.input(1));
                fused.add_input(context.input(1));
                optimizedNodes.push_back(std::move(fused));
                index += 3;
                ++fusedAttentionCount;
                continue;
            }
        }
        optimizedNodes.push_back(graph->node(index));
        ++index;
    }
    if (fusedAttentionCount > 0) {
        graph->clear_node();
        for (auto& node : optimizedNodes) *graph->add_node() = std::move(node);
        std::cout << "OnnxNetwork: fused " << fusedAttentionCount
                  << " attention score/softmax/value sequences" << std::endl;
    } else if (!canFuseAttention) {
        std::cout << "OnnxNetwork: attention fusion disabled because the device does not expose "
                     "32-wide compute subgroups with arithmetic operations" << std::endl;
    }

    input.close();

    std::cout << "OnnxNetwork: Model loaded successfully" << std::endl;

    return true;
}

std::vector<char> OnnxNetwork::readTensorData(const onnx::TensorProto& tensor) const {
    if (!tensor.raw_data().empty()) {
        return {tensor.raw_data().begin(), tensor.raw_data().end()};
    }
    if (tensor.data_location() != onnx::TensorProto::EXTERNAL) return {};

    std::string location;
    uint64_t offset = 0;
    uint64_t length = 0;
    bool hasLength = false;
    const auto parseUnsigned = [&](const std::string& value, const char* field) {
        size_t parsed = 0;
        uint64_t result = 0;
        try {
            result = std::stoull(value, &parsed);
        } catch (const std::exception&) {
            throw std::runtime_error("Invalid external tensor " + std::string(field) + " for " + tensor.name());
        }
        if (parsed != value.size()) {
            throw std::runtime_error("Invalid external tensor " + std::string(field) + " for " + tensor.name());
        }
        return result;
    };
    for (const auto& entry : tensor.external_data()) {
        if (entry.key() == "location") location = entry.value();
        else if (entry.key() == "offset") offset = parseUnsigned(entry.value(), "offset");
        else if (entry.key() == "length") {
            length = parseUnsigned(entry.value(), "length");
            hasLength = true;
        }
    }
    if (location.empty()) throw std::runtime_error("External tensor " + tensor.name() + " has no location");

    const std::filesystem::path relativePath(location);
    if (relativePath.is_absolute()) {
        throw std::runtime_error("External tensor paths must be relative: " + location);
    }
    const auto modelDirectory = std::filesystem::weakly_canonical(std::filesystem::path(modelPath).parent_path());
    const auto dataPath = std::filesystem::weakly_canonical(modelDirectory / relativePath);
    const auto relativeToModel = dataPath.lexically_relative(modelDirectory);
    if (relativeToModel.empty() || *relativeToModel.begin() == "..") {
        throw std::runtime_error("External tensor path escapes the model directory: " + location);
    }

    std::ifstream input(dataPath, std::ios::binary | std::ios::ate);
    if (!input) throw std::runtime_error("Could not open external tensor data: " + dataPath.string());
    const uint64_t fileSize = static_cast<uint64_t>(input.tellg());
    if (!hasLength) length = fileSize >= offset ? fileSize - offset : 0;
    if (offset > fileSize || length > fileSize - offset || length > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("External tensor range is outside its data file: " + tensor.name());
    }
    input.seekg(static_cast<std::streamoff>(offset));
    std::vector<char> data(static_cast<size_t>(length));
    input.read(data.data(), static_cast<std::streamsize>(data.size()));
    if (!input) throw std::runtime_error("Could not read external tensor data: " + tensor.name());
    return data;
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
    constexpr VkBufferUsageFlags dimensionsUsage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    constexpr VkMemoryPropertyFlags memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    if (tensorInfo.dataType == onnx::TensorProto::FLOAT) {
        return vulkanContext->create<TensorElement<float>>(
            tensorInfo.shape, usage, dimensionsUsage, memoryProperties);
    } else if (tensorInfo.dataType == onnx::TensorProto::DOUBLE) {
        return vulkanContext->create<TensorElement<double>>(
            tensorInfo.shape, usage, dimensionsUsage, memoryProperties);
    } else if (tensorInfo.dataType == onnx::TensorProto::INT32) {
        return vulkanContext->create<TensorElement<int32_t>>(
            tensorInfo.shape, usage, dimensionsUsage, memoryProperties);
    } else if (tensorInfo.dataType == onnx::TensorProto::INT64) {
        return vulkanContext->create<TensorElement<int64_t>>(
            tensorInfo.shape, usage, dimensionsUsage, memoryProperties);
    }

    throw std::runtime_error("Unsupported data type: " + std::to_string(tensorInfo.dataType));
}

std::shared_ptr<TensorElementInterface> createConstantTensor(VulkanContext* vulkanContext, const TensorInfo& tensorInfo) {
    // TODO we use VK_BUFFER_USAGE_TRANSFER_SRC_BIT for all buffers for now, might be not optimal
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    constexpr VkBufferUsageFlags dimensionsUsage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    constexpr VkMemoryPropertyFlags memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    if (tensorInfo.dataType == onnx::TensorProto::FLOAT) {
        return vulkanContext->create<TensorElementSinglePath<float>>(
            tensorInfo.shape, usage, dimensionsUsage, memoryProperties);
    } else if (tensorInfo.dataType == onnx::TensorProto::DOUBLE) {
        return vulkanContext->create<TensorElementSinglePath<double>>(
            tensorInfo.shape, usage, dimensionsUsage, memoryProperties);
    } else if (tensorInfo.dataType == onnx::TensorProto::INT32) {
        return vulkanContext->create<TensorElementSinglePath<int32_t>>(
            tensorInfo.shape, usage, dimensionsUsage, memoryProperties);
    } else if (tensorInfo.dataType == onnx::TensorProto::INT64) {
        return vulkanContext->create<TensorElementSinglePath<int64_t>>(
            tensorInfo.shape, usage, dimensionsUsage, memoryProperties);
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
    std::string operationType = node.op_type();
    std::map<std::string, ComputeGraphElementPtr> outputs;

    if (operationType == "Constant") {
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
        for (const auto& outputName : node.output()) {
            const auto tensorInfo = name2TensorInfo.at(outputName);
            auto output = createTensor(vulkanContext, tensorInfo);
            output->setName(outputName);
            outputs[outputName] = output;
        }
    }

    return outputs;
}

void OnnxNetwork::createInitializerTensor(const onnx::TensorProto* initializer,
    TensorInfoMap& name2TensorInfo,
    VulkanContext* vulkanContext)
{
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
    if (input->has_type() && input->type().has_tensor_type()) {
        std::vector<uint32_t> inputShape;
        const onnx::TypeProto::Tensor& tensor_type = input->type().tensor_type();
        if (tensor_type.has_shape()) {
            for (int j = 0; j < tensor_type.shape().dim_size(); ++j) {
                const auto& dim = tensor_type.shape().dim(j);
                if (dim.has_dim_value()) {
                    inputShape.push_back(dim.dim_value());
                } else if (dim.has_dim_param()) {
                    // set dim = 1 for dynamic dimensions
                    // TODO this is a placeholder, should handle dynamic dimensions properly
                    inputShape.push_back(1);
                }
            }
        }
        if (true) { // inputShape.size() == 4) {
            onnx::TensorProto::DataType dataType = getTensorDataType(tensor_type);
            TensorInfo tensorInfo{dataType, inputShape};
            name2TensorInfo[input->name()] = tensorInfo;

            auto tensor = createConstantTensor(vulkanContext, tensorInfo);
            tensor->setName(input->name());
            graphDataElements[input->name()] = tensor;

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

    std::set<std::string> referencedTensorNames;
    for (const auto& input : graph.input()) referencedTensorNames.insert(input.name());
    for (const auto& output : graph.output()) referencedTensorNames.insert(output.name());
    for (const auto& node : graph.node()) {
        for (const auto& input : node.input()) if (!input.empty()) referencedTensorNames.insert(input);
        for (const auto& output : node.output()) if (!output.empty()) referencedTensorNames.insert(output);
    }

    std::vector<const onnx::ValueInfoProto*> infos;
    for (int i = 0; i < graph.input_size(); ++i) {
        infos.push_back(&graph.input(i));
    }

    // WTF why do we have the output elements here?, they will be generated together with the operations
    for (int i = 0; i < graph.output_size(); ++i) {
        infos.push_back(&graph.output(i));
    }

    for (int i = 0; i < graph.value_info_size(); ++i) {
        if (referencedTensorNames.count(graph.value_info(i).name())) {
            infos.push_back(&graph.value_info(i));
        }
    }

    std::vector<const onnx::TensorProto*> initializers;
    for (int i = 0; i < graph.initializer_size(); ++i) {
        const onnx::TensorProto& initializer = graph.initializer(i);
        initializers.push_back(&initializer);
    }

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

    // Derive transient tensor lifetimes from the ONNX producer/consumer graph
    // and assign non-overlapping tensors to shared physical storage.
    planTransientTensorStorage();

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
        for (const auto& [name, output] : outputs) {
            graphDataElements[name] = output;
            int slot = 0;
            for (const auto& inputName : node.input()) {
                if (!inputName.empty()) ++slot;
            }
            for (int outputIndex = 0; outputIndex < node.output_size(); ++outputIndex) {
                if (node.output(outputIndex) == name) {
                    slot += outputIndex;
                    break;
                }
            }
            outputName2GraphElementAndSlot[name] = std::make_pair(graphOperationElements[i], slot);
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

        // Get inputs for this operation
        std::vector<std::string> inputNames;
        int startIndex = 0;
        for (int j = 0; j < node.input_size(); j++) {
            auto index = node.input(j);
            if (index.empty()) continue;
            if (outputName2GraphElementAndSlot.find(index) == outputName2GraphElementAndSlot.end()) {
                // inde
                // we have a direct input, so no compute node but a tensor/buffer
                operation->setInput(graphDataElements.at(index), startIndex);
            } else {
                auto [input, slot] = outputName2GraphElementAndSlot.at(index);
                operation->setInput(input, startIndex, slot);
            }
            ++startIndex;
        }

        for (int j = 0; j < node.output_size(); j++) {
            auto index = node.output(j);
            auto output = graphDataElements.at(index);
            if (output == nullptr) {
                throw std::runtime_error("Output tensor is null for operation output " + std::to_string(j));
            }
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

    for (uint32_t outputIndex = 0; outputIndex < outputNames.size(); ++outputIndex) {
        const auto& outputName = outputNames[outputIndex];

        const auto tensor = graphDataElements.at(outputName);
        inputs[outputIndex] = tensor;
        srcOutputSlots[outputIndex] = -1;

        // Computed outputs depend on their producing operation. Frozen test
        // fixtures may also expose graph inputs or initializers as outputs;
        // those are direct tensor dependencies with no producing operation.
        const auto producer = outputName2GraphElementAndSlot.find(outputName);
        outputElements[outputIndex] = producer == outputName2GraphElementAndSlot.end()
                                          ? tensor
                                          : producer->second.first;

    }
}

void OnnxNetwork::planTransientTensorStorage()
{
    const onnx::GraphProto& graph = model->graph();
    const size_t graphEnd = static_cast<size_t>(graph.node_size());

    struct Lifetime {
        std::string name;
        size_t producer;
        size_t lastUse;
        std::shared_ptr<TensorElementInterface> tensor;
    };

    std::map<std::string, Lifetime> lifetimes;
    for (int nodeIndex = 0; nodeIndex < graph.node_size(); ++nodeIndex) {
        const auto& node = graph.node(nodeIndex);
        for (const auto& outputName : node.output()) {
            const auto element = graphDataElements.find(outputName);
            if (element == graphDataElements.end()) continue;
            auto tensor = std::dynamic_pointer_cast<TensorElementInterface>(element->second);
            if (!tensor || tensor->isSinglePathStorage()) continue;
            lifetimes.emplace(outputName, Lifetime{
                outputName,
                static_cast<size_t>(nodeIndex),
                static_cast<size_t>(nodeIndex),
                tensor,
            });
        }
    }

    for (int nodeIndex = 0; nodeIndex < graph.node_size(); ++nodeIndex) {
        const auto& node = graph.node(nodeIndex);
        for (const auto& inputName : node.input()) {
            const auto lifetime = lifetimes.find(inputName);
            if (lifetime != lifetimes.end()) {
                lifetime->second.lastUse = std::max(
                    lifetime->second.lastUse, static_cast<size_t>(nodeIndex));
            }
        }
    }
    for (const auto& output : graph.output()) {
        const auto lifetime = lifetimes.find(output.name());
        if (lifetime != lifetimes.end()) lifetime->second.lastUse = graphEnd;
    }

    // Metadata-only operations and shape-preserving Slice nodes share their input storage.
    std::map<std::string, std::string> parent;
    for (const auto& [name, lifetime] : lifetimes) parent[name] = name;
    std::function<std::string(const std::string&)> findRoot = [&](const std::string& name) {
        auto& value = parent.at(name);
        if (value != name) value = findRoot(value);
        return value;
    };

    for (int nodeIndex = 0; nodeIndex < graph.node_size(); ++nodeIndex) {
        const auto& node = graph.node(nodeIndex);
        if ((node.op_type() != "Reshape" && node.op_type() != "Unsqueeze" && node.op_type() != "Cast" &&
             node.op_type() != "Slice") ||
            node.input_size() < 1 || node.output_size() < 1) continue;
        const auto input = lifetimes.find(node.input(0));
        const auto output = lifetimes.find(node.output(0));
        if (input == lifetimes.end() || output == lifetimes.end()) continue;
        if (input->second.tensor->getBufferMemSize() != output->second.tensor->getBufferMemSize() ||
            input->second.tensor->getElementType() != output->second.tensor->getElementType()) continue;

        parent[findRoot(output->first)] = findRoot(input->first);
        auto operation = vulkanContext->create<NoOp>();
        operation->setName(graphOperationElements.at(nodeIndex)->getName());
        graphOperationElements[nodeIndex] = operation;
        for (const auto& outputName : node.output()) {
            auto mapping = outputName2GraphElementAndSlot.find(outputName);
            if (mapping != outputName2GraphElementAndSlot.end()) mapping->second.first = operation;
        }
        ++memoryPlanStats.viewAliasCount;
    }

    struct GroupLifetime {
        std::string name;
        size_t producer = graphEnd;
        size_t lastUse = 0;
        size_t bytes = 0;
        std::type_index elementType = typeid(void);
        std::vector<std::string> members;
    };
    std::map<std::string, GroupLifetime> groups;
    size_t totalLogicalBytes = 0;
    for (const auto& [name, lifetime] : lifetimes) {
        const std::string root = findRoot(name);
        auto [position, inserted] = groups.emplace(root, GroupLifetime{});
        auto& group = position->second;
        if (inserted) {
            group.name = root;
            group.elementType = lifetime.tensor->getElementType();
        }
        group.producer = std::min(group.producer, lifetime.producer);
        group.lastUse = std::max(group.lastUse, lifetime.lastUse);
        group.bytes = std::max(group.bytes, lifetime.tensor->getBufferMemSize());
        group.members.push_back(name);
        totalLogicalBytes += lifetime.tensor->getBufferMemSize();
    }
    for (const auto& [name, group] : groups) {
        if (group.members.size() > 1) {
            for (const auto& member : group.members) tensorViewGroups[member] = group.members;
        }
    }

    std::vector<TensorLifetimeRequest> requests;
    requests.reserve(groups.size());
    for (const auto& [name, group] : groups) {
        requests.push_back({name, group.bytes, group.producer, group.lastUse, group.elementType});
    }

    const TensorMemoryPlan plan = TensorMemoryPlanner::plan(std::move(requests));
    memoryPlanStats.logicalBytes = totalLogicalBytes;
    memoryPlanStats.allocatedBytes = plan.allocatedBytes;
    memoryPlanStats.peakLiveBytes = plan.peakLiveBytes;
    memoryPlanStats.slotCount = plan.slotCapacities.size();
    memoryPlanStats.tensorCount = lifetimes.size();

    std::map<std::string, size_t> groupToSlot;
    std::map<size_t, std::string> slotOwner;
    for (const auto& assignment : plan.assignments) {
        groupToSlot[assignment.name] = assignment.slot;
        for (const auto& member : groups.at(assignment.name).members) {
            const auto owner = slotOwner.find(assignment.slot);
            if (owner == slotOwner.end() ||
                lifetimes.at(member).tensor->getBufferMemSize() >
                    lifetimes.at(owner->second).tensor->getBufferMemSize()) {
                slotOwner[assignment.slot] = member;
            }
        }
    }

    for (const auto& assignment : plan.assignments) {
        const std::string& ownerName = slotOwner.at(assignment.slot);
        for (const auto& member : groups.at(assignment.name).members) {
            if (member != ownerName) {
                lifetimes.at(member).tensor->shareDataStorageWith(
                    *lifetimes.at(ownerName).tensor);
            }
        }
    }

    // Reusing a slot adds a write-after-read dependency between otherwise
    // independent ONNX branches. The dependency affects scheduling only and
    // is deliberately not exposed as a shader descriptor input.
    std::map<size_t, std::vector<const GroupLifetime*>> slotLifetimes;
    for (const auto& [name, group] : groups) {
        slotLifetimes[groupToSlot.at(name)].push_back(&group);
    }
    for (auto& [slot, values] : slotLifetimes) {
        std::sort(values.begin(), values.end(), [](const auto* lhs, const auto* rhs) {
            return lhs->producer < rhs->producer;
        });
        for (size_t index = 1; index < values.size(); ++index) {
            const GroupLifetime& previous = *values[index - 1];
            const GroupLifetime& current = *values[index];
            if (previous.lastUse < current.producer && previous.lastUse < graphEnd) {
                graphOperationElements.at(static_cast<uint32_t>(current.producer))->addDependency(
                    graphOperationElements.at(static_cast<uint32_t>(previous.lastUse)));
            }
        }
    }

    // Tensor lifetimes above are expressed in ONNX node order. Independent
    // branches may otherwise be emitted in a different topological order by
    // ComputeGraph, invalidating those lifetime intervals and overwriting a
    // reused slot too early. Keep execution consistent with the order used by
    // the memory plan. The Vulkan queue is serial already, so this adds the
    // required dependency edges without reducing available device parallelism.
    for (int nodeIndex = 1; nodeIndex < graph.node_size(); ++nodeIndex) {
        graphOperationElements.at(nodeIndex)->addDependency(
            graphOperationElements.at(nodeIndex - 1));
    }

    std::cout << "OnnxNetwork: transient memory plan: "
              << memoryPlanStats.tensorCount << " tensors, "
              << memoryPlanStats.slotCount << " slots, "
              << memoryPlanStats.logicalBytes << " logical bytes -> "
              << memoryPlanStats.allocatedBytes << " allocated bytes"
              << std::endl;
}

void OnnxNetwork::retainTensor(const std::string& name)
{
    const auto element = graphDataElements.find(name);
    if (element == graphDataElements.end()) {
        throw std::runtime_error("ONNX tensor " + name + " not found");
    }
    if (retainedTensorNames.count(name)) return;

    std::vector<std::string> members{name};
    const auto viewGroup = tensorViewGroups.find(name);
    if (viewGroup != tensorViewGroups.end()) members = viewGroup->second;

    std::string ownerName = members.front();
    auto owner = std::dynamic_pointer_cast<TensorElementInterface>(
        graphDataElements.at(ownerName));
    if (!owner || owner->isSinglePathStorage()) {
        retainedTensorNames.insert(name);
        return;
    }
    for (const auto& member : members) {
        auto tensor = std::dynamic_pointer_cast<TensorElementInterface>(graphDataElements.at(member));
        if (tensor->getBufferMemSize() > owner->getBufferMemSize()) {
            ownerName = member;
            owner = tensor;
        }
    }

    owner->makeDataStorageUnique();
    for (const auto& member : members) {
        auto tensor = std::dynamic_pointer_cast<TensorElementInterface>(graphDataElements.at(member));
        if (member != ownerName) tensor->shareDataStorageWith(*owner);
        retainedTensorNames.insert(member);
    }
    memoryPlanStats.allocatedBytes += owner->getStorageCapacityBytes();
}

void OnnxNetwork::setInputTensor(const std::string& name,
                                 ComputeGraphElementPtr producer,
                                 int outputSlot) {
    if (!producer) {
        throw std::runtime_error("ONNX input producer must not be null");
    }

    const auto& graph = model->graph();
    const auto graphInput = std::find_if(
        graph.input().begin(), graph.input().end(),
        [&name](const onnx::ValueInfoProto& value) { return value.name() == name; });
    if (graphInput == graph.input().end()) {
        throw std::runtime_error("ONNX graph input " + name + " not found");
    }

    auto tensor = outputSlot == -1 ? producer : producer->getInputElement(outputSlot);
    auto tensorInterface = std::dynamic_pointer_cast<TensorElementInterface>(tensor);
    if (!tensorInterface) {
        throw std::runtime_error("Producer for ONNX graph input " + name + " is not a tensor");
    }

    auto expected = std::dynamic_pointer_cast<TensorElementInterface>(graphDataElements.at(name));
    if (!expected || tensorInterface->getDimensions() != expected->getDimensions()) {
        throw std::runtime_error("Tensor dimensions do not match ONNX graph input " + name);
    }

    bool connected = false;
    for (int i = 0; i < graph.node_size(); ++i) {
        const auto& node = graph.node(i);
        for (int j = 0; j < node.input_size(); ++j) {
            if (node.input(j) == name) {
                graphOperationElements.at(i)->setInput(producer, j, outputSlot);
                connected = true;
            }
        }
    }
    if (!connected) {
        throw std::runtime_error("ONNX graph input " + name + " is not consumed by a node");
    }

    graphDataElements[name] = tensor;
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

    // The weights and constants are device-local; their staging copies are
    // submitted in a few large batches, on the queue the graph is compiled for.
    BatchedUpload batch(vulkanContext, getSetupQueue());

    // create all initializers
    for (int i = 0; i < graph.initializer_size(); ++i) {
        const auto& init = graph.initializer(i);
        std::string name = init.name();
        if (graphDataElements.find(name) == graphDataElements.end()) {
            continue;
        }
        std::vector<uint32_t> initShape;
        for (int j = 0; j < init.dims_size(); j++) {
            initShape.push_back(init.dims(j));
        }
        const auto& dataType = static_cast<onnx::TensorProto::DataType>(init.data_type());

        const auto initData = readTensorData(init);
        if (!initData.empty()) {
            switch (dataType) {
            case onnx::TensorProto::FLOAT: {
                auto tensor = std::dynamic_pointer_cast<TensorElementSinglePath<float>>(graphDataElements.at(name));
                if (!tensor) throw std::runtime_error("Initializer " + name + " is not a FLOAT tensor");
                tensor->getDataBuffer().memcopyFrom(batch, initData.data(), initData.size());
                break;
            }
            case onnx::TensorProto::INT32: {
                auto tensor = std::dynamic_pointer_cast<TensorElementSinglePath<int32_t>>(graphDataElements.at(name));
                if (!tensor) throw std::runtime_error("Initializer " + name + " is not an INT32 tensor");
                tensor->getDataBuffer().memcopyFrom(batch, initData.data(), initData.size());
                break;
            }
            case onnx::TensorProto::INT64: {
                auto tensor = std::dynamic_pointer_cast<TensorElementSinglePath<int64_t>>(graphDataElements.at(name));
                if (!tensor) throw std::runtime_error("Initializer " + name + " is not an INT64 tensor");
                tensor->getDataBuffer().memcopyFrom(batch, initData.data(), initData.size());
                break;
            }
            default:
                throw std::runtime_error("Initializer " + name + " has unsupported raw data type");
            }
        } else if (dataType == onnx::TensorProto::FLOAT && init.float_data_size() > 0) {
            std::cout << " - Data type: FLOAT (float_data field)" << std::endl;
            auto element = graphDataElements.at(name);
            std::shared_ptr<TensorElementSinglePath<float>> elementPtrSingle = std::dynamic_pointer_cast<TensorElementSinglePath<float>>(element);
            std::shared_ptr<TensorElement<float>> elementPtrMulti = std::dynamic_pointer_cast<TensorElement<float>>(element);
            if (elementPtrSingle) {
                TensorElementSinglePath<float>* tensor = elementPtrSingle.get();
                tensor->getDataBuffer().memcopyFrom(batch, init.float_data().data(), init.float_data_size());
            } else if (elementPtrMulti) {
                TensorElement<float>* tensor = elementPtrMulti.get();
                tensor->getDataBuffer(0).memcopyFrom(batch, init.float_data().data(), init.float_data_size());
            } else {
                throw std::runtime_error("Initializer " + name + " has unsupported tensor element type");
            }
        } else if (dataType == onnx::TensorProto::INT32 && init.int32_data_size() > 0) {
            auto tensor = std::dynamic_pointer_cast<TensorElementSinglePath<int32_t>>(graphDataElements.at(name));
            if (!tensor) throw std::runtime_error("Initializer " + name + " is not an INT32 tensor");
            tensor->getDataBuffer().memcopyFrom(batch, init.int32_data().data(), init.int32_data_size());
        } else if (dataType == onnx::TensorProto::INT64 && init.int64_data_size() > 0) {
            auto tensor = std::dynamic_pointer_cast<TensorElementSinglePath<int64_t>>(graphDataElements.at(name));
            if (!tensor) throw std::runtime_error("Initializer " + name + " is not an INT64 tensor");
            tensor->getDataBuffer().memcopyFrom(batch, init.int64_data().data(), init.int64_data_size());
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
                        tensorElement->getDataBuffer().memcopyFrom(batch, dataLocation, dataSize);
                    } else if (type == onnx::TensorProto::INT32) {
                        std::shared_ptr<TensorElementSinglePath<int32_t>> tensorElement = std::dynamic_pointer_cast<TensorElementSinglePath<int32_t>>(graphDataElements[node.output(0)]);
                        std::cout << "copy values for constant node " << node.name() << " of size " << dataSize << std::endl;
                        tensorElement->getDataBuffer().memcopyFrom(batch, dataLocation, dataSize);
                    } else if (type == onnx::TensorProto::INT64) {
                        std::shared_ptr<TensorElementSinglePath<int64_t>> tensorElement = std::dynamic_pointer_cast<TensorElementSinglePath<int64_t>>(graphDataElements[node.output(0)]);
                        std::cout << "copy values for constant node " << node.name() << " of size " << dataSize << std::endl;
                        tensorElement->getDataBuffer().memcopyFrom(batch, dataLocation, dataSize);
                    } else {
                        std::cerr << "Unsupported constant data type: " << type << std::endl;
                        throw std::runtime_error("Unsupported constant data type: " + std::to_string(type));
                    }
                }
            }
        }
    }
    batch.submit();
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
            const auto rawData = readTensorData(init);
            if (!rawData.empty()) {
                if (rawData.size() % sizeof(float) != 0) {
                    throw std::runtime_error("Initializer " + name + " has an invalid FLOAT byte count");
                }
                std::vector<float> result(rawData.size() / sizeof(float));
                std::memcpy(result.data(), rawData.data(), rawData.size());
                return result;
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
