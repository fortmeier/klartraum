#ifndef ONNX_NETWORK_HPP
#define ONNX_NETWORK_HPP

#include <memory>
#include <string>
#include <vector>

#include "klartraum/computegraph/buffertransformation.hpp"
#include "klartraum/computegraph/computegraphgroup.hpp"
#include "klartraum/computegraph/rendergraphelement.hpp"
#include "klartraum/vulkan_buffer.hpp"
#include "klartraum/vulkan_context.hpp"

// Forward declare ONNX types
namespace onnx {
class ModelProto;
class GraphProto;
class NodeProto;
class ValueInfoProto;
}

namespace klartraum {

enum class OnnxDataType {
    Float32,
    Float16,
    Int32,
    Int64,
    Uint8,
    Unknown
};

struct OnnxTensorInfo {
    std::string name;
    OnnxDataType dataType;
    std::vector<int64_t> shape;
    size_t totalElements;
    size_t sizeInBytes;
};

class OnnxNetwork : virtual public ComputeGraphElement, virtual public ComputeGraphGroup {
    /**
     * @brief ONNX Neural Network Graph Representation
     *
     * This class provides GPU-accelerated execution of ONNX models using Klartraum
     * ComputeGraph Framework.
     * It supports:
     * 1. Loading and parsing ONNX model files
     * 2. Converting ONNX operations to ComputeGraphElement instances
     * 3. Managing GPU memory for tensors and intermediate results by ComputeGraphElement instances
     */
public:
    OnnxNetwork(
        VulkanContext& vulkanContext,
        const std::string& modelPath);
    ~OnnxNetwork();


    // Print detailed model information
    void printModelInfo() const;

    // ComputeGraphGroup interface
    virtual void checkInput(ComputeGraphElementPtr input, int index = 0) override;
    virtual void _setup(VulkanContext& vulkanContext, uint32_t numberPaths) override;
    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) override;

    // RenderGraphElement interface
    virtual const char* getType() const override {
        return "OnnxNetwork";
    }

private:
    // Load ONNX model from file
    bool loadModel(const std::string& modelPath);

    // Model parsing and graph creation
    void createComputeGraph();
    
    // ONNX model data
    std::unique_ptr<onnx::ModelProto> model;
    std::string modelPath;

    // Vulkan resources
    VulkanContext* vulkanContext = nullptr;
    std::vector<ComputeGraphElementPtr> graphElements;
    uint32_t numberOfPaths = 0;

};

} // namespace klartraum

#endif // ONNX_NETWORK_HPP
