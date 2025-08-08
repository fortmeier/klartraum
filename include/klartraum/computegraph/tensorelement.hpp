#ifndef KLARTRAUM_TENSORELEMENT_HPP
#define KLARTRAUM_TENSORELEMENT_HPP

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "klartraum/computegraph/computegraphelement.hpp"
#include "klartraum/vulkan_buffer.hpp"

namespace klartraum {

template <typename T>
T prod(const std::vector<T>& vec) {
    T result = 1;
    for (const T& val : vec) {
        result *= val;
    }
    return result;
}


/**
 * @brief TensorElement represents a tensor with separate dimension and data buffers
 *
 * This class manages GPU tensors with:
 * - A dimensions buffer containing 4 uint32_t values [width, height, depth, batch]
 * - A data buffer containing the actual tensor data of the specified type
 *
 * The class follows the ComputeGraphElement pattern and manages multiple paths
 * for different rendering/compute contexts.
 */
template <typename DataType>
class TensorElement : public ComputeGraphElement {
public:
    /**
     * @brief Construct a TensorElement with specified dimensions
     *
     * @param vulkanContext The Vulkan context for buffer creation
     * @param batch Tensor batch dimension
     * @param depth Tensor depth dimension
     * @param height Tensor height dimension
     * @param width Tensor width dimension
     * @param dataUsageFlags Vulkan usage flags for the data buffer
     * @param dimUsageFlags Vulkan usage flags for the dimensions buffer
     */
    TensorElement(
        VulkanContext& vulkanContext,
        uint32_t batch,
        uint32_t depth,
        uint32_t height,
        uint32_t width,
        VkBufferUsageFlags dataUsageFlags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VkBufferUsageFlags dimUsageFlags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT) : vulkanContext(vulkanContext),
                                                                                                                    dimensions({width, height, depth, batch}),
                                                                                                                    dataElements(batch * depth * height * width),
                                                                                                                    dataUsageFlags(dataUsageFlags),
                                                                                                                    dimUsageFlags(dimUsageFlags) {

        validateDimensions(dimensions);
    }

    /**
     * @brief Construct a TensorElement from a dimensions vector
     *
     * @param vulkanContext The Vulkan context for buffer creation
     * @param dimensions Vector containing [width, height, depth, batch]
     * @param dataElements Total number of data elements
     * @param dataUsageFlags Vulkan usage flags for the data buffer
     * @param dimUsageFlags Vulkan usage flags for the dimensions buffer
     */
    TensorElement(
        VulkanContext& vulkanContext,
        const std::vector<uint32_t>& dimensions,
        VkBufferUsageFlags dataUsageFlags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VkBufferUsageFlags dimUsageFlags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)
        : vulkanContext(vulkanContext),
          dimensions(dimensions),
          dataElements(prod(dimensions)),
          dataUsageFlags(dataUsageFlags),
          dimUsageFlags(dimUsageFlags) {

        validateDimensions(this->dimensions);
    }
    virtual ~TensorElement() = default;



    // ComputeGraphElement interface
    virtual void _setup(VulkanContext& vulkanContext, uint32_t numberPaths) override {
        this->numberOfPaths = numberPaths;

        // Reserve space for buffers
        dataBuffers.reserve(numberPaths);

        // Create dimensions buffer (always 4 elements: width, height, depth, batch)
        dimensionBuffer = std::make_unique<VulkanBuffer<uint32_t>>(vulkanContext, 4, dimUsageFlags);
        // Initialize dimensions buffer with the tensor dimensions
        std::vector<uint32_t> dimData = {dimensions[0], dimensions[1], dimensions[2], dimensions[3]};
        dimensionBuffer->memcopyFrom(dimData);

        // Create buffers for each path
        for (uint32_t i = 0; i < numberPaths; ++i) {
            // Create data buffer
            dataBuffers.emplace_back(vulkanContext, dataElements, dataUsageFlags);
        }
    }

    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) override {
        if (pathId >= numberOfPaths) {
            throw std::runtime_error("TensorElement: Invalid pathId in _record");
        }

        // Record buffer zeroing if requested
        if (recordDataToZero) {
            dataBuffers[pathId]._recordZero(commandBuffer);
        }

        if (recordDimensionsToZero) {
            dimensionBuffer->_recordZero(commandBuffer);
        }
    }

    virtual const char* getType() const override {
        return "TensorElement";
    }

    // Tensor-specific accessors

    /**
     * @brief Get the data buffer for a specific path
     */
    VulkanBuffer<DataType>& getDataBuffer(uint32_t pathId) {
        if (pathId >= numberOfPaths) {
            throw std::runtime_error("TensorElement: Invalid pathId for data buffer access");
        }
        return dataBuffers[pathId];
    }

    /**
     * @brief Get the dimensions buffer for a specific path
     */
    VulkanBuffer<uint32_t>& getDimensionsBuffer(uint32_t pathId) {
        if (pathId >= numberOfPaths) {
            throw std::runtime_error("TensorElement: Invalid pathId for dimensions buffer access");
        }
        return *dimensionBuffer;
    }

    /**
     * @brief Get the Vulkan buffer handle for data buffer
     */
    VkBuffer& getDataVkBuffer(uint32_t pathId) {
        return getDataBuffer(pathId).getBuffer();
    }

    /**
     * @brief Get the Vulkan buffer handle for dimensions buffer
     */
    VkBuffer& getDimensionsVkBuffer(uint32_t pathId) {
        return getDimensionsBuffer(pathId).getBuffer();
    }

    /**
     * @brief Get tensor dimensions
     */
    const std::vector<uint32_t>& getDimensions() const { return dimensions; }

    /**
     * @brief Get number of data elements
     */
    uint32_t getDataElementCount() const { return dataElements; }

    /**
     * @brief Get total memory size for data buffers
     */
    size_t getDataBufferMemSize() const {
        if (dataBuffers.empty()) {
            return sizeof(DataType) * dataElements;
        }
        return dataBuffers[0].getBufferMemSize();
    }

    /**
     * @brief Get total memory size for dimensions buffers
     */
    size_t getDimensionsBufferMemSize() const {
        return dimensionBuffer->getBufferMemSize();
    }

    /**
     * @brief Copy data to the data buffer for a specific compute path
     */
    void setData(uint32_t pathId, const std::vector<DataType>& data) {
        if (pathId >= numberOfPaths) {
            throw std::runtime_error("TensorElement: Invalid pathId for data setting");
        }

        if (data.size() != dataElements) {
            throw std::runtime_error("TensorElement: Data size mismatch. Expected " +
                                     std::to_string(dataElements) + " but got " + std::to_string(data.size()));
        }

        dataBuffers[pathId].memcopyFrom(data);
    }

    /**
     * @brief Copy data from the data buffer for a specific path
     */
    void getData(uint32_t pathId, std::vector<DataType>& data) {
        if (pathId >= numberOfPaths) {
            throw std::runtime_error("TensorElement: Invalid pathId for data getting");
        }

        data.resize(dataElements);
        dataBuffers[pathId].memcopyTo(data);
    }

    /**
     * @brief Set whether to zero buffers during recording
     */
    void setRecordToZero(bool recordDataToZero, bool recordDimensionsToZero = false) {
        this->recordDataToZero = recordDataToZero;
        this->recordDimensionsToZero = recordDimensionsToZero;
    }

private:
    VulkanContext& vulkanContext;
    std::vector<uint32_t> dimensions; // [width, height, depth, batch]
    uint32_t dataElements;
    uint32_t numberOfPaths = 0;

    // Buffer usage flags
    VkBufferUsageFlags dataUsageFlags;
    VkBufferUsageFlags dimUsageFlags;

    // Buffers for each path
    std::vector<VulkanBuffer<DataType>> dataBuffers;

    // dimensions buffer is the same for each path
    std::unique_ptr<VulkanBuffer<uint32_t>> dimensionBuffer;

    // Recording control flags
    bool recordDataToZero = false;
    bool recordDimensionsToZero = false;

    void validateDimensions(const std::vector<uint32_t>& dims) {
        if (dims.size() != 4) {
            throw std::runtime_error("TensorElement: Dimensions must contain exactly 4 values [width, height, depth, batch]");
        }

        for (size_t i = 0; i < dims.size(); ++i) {
            if (dims[i] == 0) {
                throw std::runtime_error("TensorElement: Dimension " + std::to_string(i) + " cannot be zero");
            }
        }
    }
};

} // namespace klartraum

#endif // KLARTRAUM_TENSORELEMENT_HPP
