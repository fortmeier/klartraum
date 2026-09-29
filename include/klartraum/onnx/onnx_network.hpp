// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_ONNX_ONNX_NETWORK_HPP
#define KLARTRAUM_ONNX_ONNX_NETWORK_HPP

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <set>

#include "klartraum/computegraph/buffertransformation.hpp"
#include "klartraum/computegraph/computegraphgroup.hpp"
#include "klartraum/computegraph/rendergraphelement.hpp"
#include "klartraum/vulkan_buffer.hpp"
#include "klartraum/vulkan_context.hpp"

#include "onnx.pb.h"

// Forward declare ONNX types
namespace onnx {
class ModelProto;
class GraphProto;
class NodeProto;
class ValueInfoProto;
}

namespace klartraum {

enum class OnnxDataType { Float32, Float16, Int32, Int64, Uint8, Unknown };

// struct OnnxTensorInfo {
//     std::string name;
//     OnnxDataType dataType;
//     std::vector<int64_t> shape;
//     size_t totalElements;
//     size_t sizeInBytes;
// };

struct TensorInfo {
    onnx::TensorProto::DataType dataType;
    std::vector<uint32_t> shape;
};

using TensorInfoMap = std::map<std::string, TensorInfo>;

/** @brief Memory-reuse statistics for an ONNX network's transient tensors. */
struct OnnxMemoryPlanStats {
    size_t logicalBytes = 0;   ///< Bytes required if every logical tensor had unique storage.
    size_t allocatedBytes = 0; ///< Bytes allocated across reusable physical slots.
    size_t peakLiveBytes = 0;  ///< Maximum logical bytes simultaneously live.
    size_t slotCount = 0;      ///< Number of reusable physical storage slots.
    size_t tensorCount = 0;    ///< Number of transient logical tensors in the plan.
    size_t viewAliasCount = 0; ///< Number of tensors represented as zero-copy views.
};

/**
 * @brief Loads and executes an ONNX network as a Klartraum compute-graph group.
 *
 * ONNX nodes are translated to Vulkan compute operations. Transient tensor
 * lifetimes are derived from the ONNX graph so non-overlapping tensors can
 * share device-local storage. Graph inputs may also be connected directly to
 * tensors produced by preceding Klartraum graph elements.
 */
class OnnxNetwork : virtual public ComputeGraphElement, virtual public ComputeGraphGroup {
public:
    /**
     * @brief Loads an ONNX model and constructs its Klartraum graph representation.
     * @param vulkanContext Vulkan context used to create tensor and operation resources.
     * @param modelPath Path to the ONNX model file.
     * @throws std::runtime_error If the model cannot be read or contains unsupported data.
     */
    OnnxNetwork(VulkanContext& vulkanContext, const std::string& modelPath);
    ~OnnxNetwork();

    /** @brief Prints detailed model inputs, outputs, initializers, and nodes. */
    void printModelInfo() const;

    // ComputeGraphGroup interface
    virtual void checkInput(ComputeGraphElementPtr input, int index = 0) override;
    virtual void _setup(VulkanContext& vulkanContext, uint32_t numberPaths) override;
    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) override;

    // RenderGraphElement interface
    virtual const char* getType() const override { return "OnnxNetwork"; }

    /**
     * @brief Reads a floating-point model initializer.
     * @param name ONNX initializer name.
     * @return Initializer values as 32-bit floats.
     */
    std::vector<float> getFloatInitializerData(const std::string& name) const;

    /**
     * @brief Returns the graph element that produces a named ONNX tensor.
     * @param name ONNX tensor name.
     * @return Producing compute-graph element.
     * @throws std::runtime_error If no element produces @p name.
     */
    ComputeGraphElementPtr getOutputElement(const std::string& name) const;

    /** @brief Returns transient tensor memory statistics computed during model loading. */
    const OnnxMemoryPlanStats& getMemoryPlanStats() const { return memoryPlanStats; }

    /**
     * @brief Keeps a named intermediate tensor readable after network execution.
     * @param name ONNX tensor name to retain.
     * @note Call this before graph compilation/setup. Retained tensors are excluded
     *       from transient storage reuse so their values are not overwritten.
     */
    void retainTensor(const std::string& name);

    /**
     * @brief Connects an ONNX graph input directly to another graph element's tensor.
     * @param name Declared ONNX graph input name.
     * @param producer Element that produces the replacement tensor.
     * @param outputSlot Producer output slot, or `-1` for its default output.
     * @note With an explicit output slot, the producer is retained as an execution
     *       dependency while that output tensor is bound to the ONNX operation.
     */
    void setInputTensor(const std::string& name, ComputeGraphElementPtr producer, int outputSlot = -1);

private:
    // Load ONNX model from file
    bool loadModel(const std::string& modelPath);
    std::vector<char> readTensorData(const onnx::TensorProto& tensor) const;

    // Model parsing and graph creation
    void createComputeGraph();

    void createGraphElementsFromNodes();
    void createGraphElementsFromOutputTensors();
    void planTransientTensorStorage();
    void connectGraphElements();
    void storeComputeGraphGroupOutputElements();

    void createInfoTensor(const onnx::ValueInfoProto* input, TensorInfoMap& name2TensorInfo,
                          VulkanContext* vulkanContext);

    void createInitializerTensor(const onnx::TensorProto* initializer, TensorInfoMap& name2TensorInfo,
                                 VulkanContext* vulkanContext);

    // ONNX model data
    std::unique_ptr<onnx::ModelProto> model;
    std::string modelPath;

    // helper maps for mapping ONNX names to internal representations

    // name2ValueInfoProto maps ONNX tensor names to their ValueInfoProto
    // since ONNX protobuf format does not support of indexing the ValueInfoProtos
    // directly
    std::map<std::string, const onnx::ValueInfoProto*> name2ValueInfoProto;

    // store tensor information for each ONNX tensor so they are
    // easily accessible by their name
    TensorInfoMap name2TensorInfo;

    std::map<std::string, std::pair<ComputeGraphElementPtr, int>> outputName2GraphElementAndSlot;

    // Vulkan resources
    VulkanContext* vulkanContext = nullptr;
    std::map<std::string, ComputeGraphElementPtr> graphDataElements;
    std::map<uint32_t, ComputeGraphElementPtr> graphOperationElements;
    uint32_t numberOfPaths = 0;
    OnnxMemoryPlanStats memoryPlanStats;
    std::map<std::string, std::vector<std::string>> tensorViewGroups;
    std::set<std::string> retainedTensorNames;
};

} // namespace klartraum

#endif // KLARTRAUM_ONNX_ONNX_NETWORK_HPP
