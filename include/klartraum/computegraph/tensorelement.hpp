// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_COMPUTEGRAPH_TENSORELEMENT_HPP
#define KLARTRAUM_COMPUTEGRAPH_TENSORELEMENT_HPP

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <typeindex>
#include <vector>

#include "klartraum/computegraph/bufferelement.hpp"
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
 * @brief Untemplated interface for tensor elements
 *
 */
class TensorElementInterface : public BufferElementInterface {
public:
    // virtual void _setup(VulkanContext& vulkanContext, uint32_t numberPaths) = 0;

    // virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) = 0;

    virtual const char* getType() const { return "TensorElement"; }

    // virtual size_t getBufferMemSize() const = 0;

    virtual VkBuffer& getDataVkBuffer(uint32_t pathId) = 0;

    virtual VkBuffer& getDimensionsVkBuffer() = 0;

    /**
     * @brief Get tensor dimensions
     */
    virtual const std::vector<uint32_t>& getDimensions() const = 0;

    /**
     * @brief Set the dimension data and update the dimensions buffer
     */
    virtual void setDimensions(const std::vector<uint32_t>& newDimensions) = 0;

    /**
     * @brief Makes this logical tensor use another tensor's physical data buffers.
     * @param source Tensor whose storage will be shared.
     * @throws std::logic_error If setup already allocated either tensor's buffers.
     * @throws std::invalid_argument If the element types, storage modes, or capacities
     *         are incompatible.
     * @note The caller must ensure both logical tensor lifetimes do not overlap.
     */
    virtual void shareDataStorageWith(TensorElementInterface& source) = 0;

    /**
     * @brief Detaches this tensor from shared storage before graph setup.
     * @throws std::logic_error If setup already allocated the buffers.
     */
    virtual void makeDataStorageUnique() = 0;

    /** @brief Returns the capacity in bytes of the underlying shared storage. */
    virtual size_t getStorageCapacityBytes() const = 0;

    /** @brief Returns an identity token shared by tensors using the same storage. */
    virtual const void* getStorageIdentity() const = 0;

    /** @brief Returns the C++ element type stored in the data buffers. */
    virtual std::type_index getElementType() const = 0;

    /** @brief Reports whether all compute paths intentionally use one data buffer. */
    virtual bool isSinglePathStorage() const { return false; }

private:
};

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
class TensorElement : public TensorElementInterface {
    struct DataStorage {
        DataStorage(uint32_t capacityElements, VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryProperties)
            : capacityElements(capacityElements),
              usageFlags(usageFlags),
              memoryProperties(memoryProperties) {}

        uint32_t capacityElements;
        VkBufferUsageFlags usageFlags;
        VkMemoryPropertyFlags memoryProperties;
        std::vector<VulkanBuffer<DataType>> buffers;
    };

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
    TensorElement(VulkanContext& vulkanContext, uint32_t batch, uint32_t depth, uint32_t height, uint32_t width,
                  VkBufferUsageFlags dataUsageFlags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VkBufferUsageFlags dimUsageFlags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VkMemoryPropertyFlags dataMemoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        : TensorElementInterface(),
          vulkanContext(vulkanContext),
          dimensions({width, height, depth, batch}),
          dataElements(batch * depth * height * width),
          dataUsageFlags(dataUsageFlags),
          dimUsageFlags(dimUsageFlags),
          dataMemoryProperties(dataMemoryProperties),
          dataStorage(std::make_shared<DataStorage>(dataElements, dataUsageFlags, dataMemoryProperties)) {

        validateDimensions(dimensions);
    }

    /**
     * @brief Construct a TensorElement with size of a dimensions vector
     *
     * @param vulkanContext The Vulkan context for buffer creation
     * @param dimensions Vector containing [width, height, depth, batch]
     * @param dataUsageFlags Vulkan usage flags for the data buffer
     * @param dimUsageFlags Vulkan usage flags for the dimensions buffer
     */
    TensorElement(VulkanContext& vulkanContext, const std::vector<uint32_t>& dimensions,
                  VkBufferUsageFlags dataUsageFlags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VkBufferUsageFlags dimUsageFlags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                  VkMemoryPropertyFlags dataMemoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        : TensorElementInterface(),
          vulkanContext(vulkanContext),
          dimensions(dimensions),
          dataElements(prod(dimensions)),
          dataUsageFlags(dataUsageFlags),
          dimUsageFlags(dimUsageFlags),
          dataMemoryProperties(dataMemoryProperties),
          dataStorage(std::make_shared<DataStorage>(dataElements, dataUsageFlags, dataMemoryProperties)) {

        validateDimensions(this->dimensions);
    }
    virtual ~TensorElement() = default;

    // ComputeGraphElement interface
    virtual void _setup(VulkanContext& vulkanContext, uint32_t numberPaths) override {
        this->numberOfPaths = numberPaths;

        // Create dimensions buffer (always 4 elements: width, height, depth, batch)
        dimensionBuffer = std::make_unique<VulkanBuffer<uint32_t>>(vulkanContext, dimensions.size(), dimUsageFlags);
        // Initialize dimensions buffer with the tensor dimensions
        dimensionBuffer->memcopyFrom(dimensions);

        if (dataStorage->buffers.empty()) {
            dataStorage->buffers.reserve(numberPaths);
            for (uint32_t i = 0; i < numberPaths; ++i) {
                dataStorage->buffers.emplace_back(vulkanContext, dataStorage->capacityElements, dataStorage->usageFlags,
                                                  dataStorage->memoryProperties);
            }
        } else if (dataStorage->buffers.size() != numberPaths) {
            throw std::runtime_error("TensorElement: shared storage path count mismatch");
        }
    }

    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) override {
        // Record buffer zeroing if requested
        if (recordDataToZero) {
            dataStorage->buffers[pathId]._recordZero(commandBuffer);
        }

        if (recordDimensionsToZero) {
            dimensionBuffer->_recordZero(commandBuffer);
        }
    }

    virtual const char* getType() const override { return "TensorElement"; }

    // Tensor-specific accessors

    /**
     * @brief Get the data buffer for a specific path
     */
    virtual VulkanBuffer<DataType>& getDataBuffer(uint32_t pathId) {
        if (pathId >= numberOfPaths) {
            throw std::runtime_error("TensorElement: Invalid pathId for data buffer access");
        }
        return dataStorage->buffers[pathId];
    }

    /**
     * @brief Get the dimensions buffer
     */
    virtual VulkanBuffer<uint32_t>& getDimensionsBuffer() { return *dimensionBuffer; }

    /**
     * @brief Get the Vulkan buffer handle for data buffer
     */
    virtual VkBuffer& getDataVkBuffer(uint32_t pathId) override { return getDataBuffer(pathId).getBuffer(); }

    /**
     * @brief Get the Vulkan buffer handle for dimensions buffer
     */
    virtual VkBuffer& getDimensionsVkBuffer() override { return getDimensionsBuffer().getBuffer(); }

    /**
     * @brief Get tensor dimensions
     */
    const std::vector<uint32_t>& getDimensions() const override { return dimensions; }

    /**
     * @brief Set the dimension data and update the dimensions buffer
     */
    void setDimensions(const std::vector<uint32_t>& newDimensions) override {
        validateDimensions(newDimensions);
        dimensions = newDimensions;
    }

    /**
     * @brief Get number of data elements
     */
    uint32_t getDataElementCount() const { return dataElements; }

    void shareDataStorageWith(TensorElementInterface& source) override {
        auto* typedSource = dynamic_cast<TensorElement<DataType>*>(&source);
        if (!typedSource) {
            throw std::invalid_argument("TensorElement: shared storage data type mismatch");
        }
        if (typedSource->dataStorage->capacityElements < dataElements) {
            throw std::invalid_argument("TensorElement: shared storage is too small");
        }
        if (typedSource->dataStorage->usageFlags != dataUsageFlags) {
            throw std::invalid_argument("TensorElement: shared storage usage flags mismatch");
        }
        if (typedSource->dataStorage->memoryProperties != dataMemoryProperties) {
            throw std::invalid_argument("TensorElement: shared storage memory properties mismatch");
        }
        if (!dataStorage->buffers.empty()) {
            throw std::logic_error("TensorElement: storage cannot be replaced after setup");
        }
        dataStorage = typedSource->dataStorage;
    }

    void makeDataStorageUnique() override {
        if (!dataStorage->buffers.empty()) {
            throw std::logic_error("TensorElement: storage cannot be detached after setup");
        }
        dataStorage = std::make_shared<DataStorage>(dataElements, dataUsageFlags, dataMemoryProperties);
    }

    size_t getStorageCapacityBytes() const override {
        return sizeof(DataType) * static_cast<size_t>(dataStorage->capacityElements);
    }

    const void* getStorageIdentity() const override { return dataStorage.get(); }

    std::type_index getElementType() const override { return typeid(DataType); }

    /**
     * @brief Get total memory size for data buffers
     */
    size_t getDataBufferMemSize() const { return sizeof(DataType) * static_cast<size_t>(dataElements); }

    /**
     * @brief Get total memory size for dimensions buffers
     */
    size_t getDimensionsBufferMemSize() const { return dimensionBuffer->getBufferMemSize(); }

    /**
     * @brief Copy data to the data buffer for a specific compute path
     */
    void setData(uint32_t pathId, const std::vector<DataType>& data) {
        if (pathId >= numberOfPaths) {
            throw std::runtime_error("TensorElement: Invalid pathId for data setting");
        }

        if (data.size() != dataElements) {
            throw std::runtime_error("TensorElement: Data size mismatch. Expected " + std::to_string(dataElements) +
                                     " but got " + std::to_string(data.size()));
        }

        getDataBuffer(pathId).memcopyFrom(data);
    }

    /**
     * @brief Set whether to zero buffers during recording
     */
    void setRecordToZero(bool recordDataToZero, bool recordDimensionsToZero = false) {
        this->recordDataToZero = recordDataToZero;
        this->recordDimensionsToZero = recordDimensionsToZero;
    }

    // overrides for BufferElementInterface
    virtual size_t getBufferMemSize() const override { return getDataBufferMemSize(); }

    virtual VkBuffer& getVkBuffer(uint32_t pathId) override { return getDataVkBuffer(pathId); }

private:
    VulkanContext& vulkanContext;
    std::vector<uint32_t> dimensions;
    uint32_t dataElements;
    uint32_t numberOfPaths = 0;

    // Buffer usage flags
    VkBufferUsageFlags dataUsageFlags;
    VkBufferUsageFlags dimUsageFlags;
    VkMemoryPropertyFlags dataMemoryProperties;

    // Logical tensors can share this physical storage when their graph
    // lifetimes do not overlap.
    std::shared_ptr<DataStorage> dataStorage;

    // dimensions buffer is the same for each path
    std::unique_ptr<VulkanBuffer<uint32_t>> dimensionBuffer;

    // Recording control flags
    bool recordDataToZero = false;
    bool recordDimensionsToZero = false;

    void validateDimensions(const std::vector<uint32_t>& dims) {
        for (size_t i = 0; i < dims.size(); ++i) {
            if (dims[i] == 0) {
                throw std::runtime_error("TensorElement: Dimension " + std::to_string(i) + " cannot be zero");
            }
        }
    }
};

/**
 * @brief TensorElementSinglePath represents a tensor with a single buffer
 *
 * This class is designed for constant/fixed tensors (weights, biases, initializers)
 * that don't change during execution and therefore only need a single buffer
 * instead of multiple buffers for different paths.
 *
 */
template <typename DataType>
class TensorElementSinglePath : public TensorElement<DataType> {
public:
    /**
     * @brief Construct a TensorElementSinglePath with specified dimensions
     *
     * @param vulkanContext The Vulkan context for buffer creation
     * @param batch Tensor batch dimension
     * @param depth Tensor depth dimension
     * @param height Tensor height dimension
     * @param width Tensor width dimension
     * @param dataUsageFlags Vulkan usage flags for the data buffer
     * @param dimUsageFlags Vulkan usage flags for the dimensions buffer
     */
    TensorElementSinglePath(
        VulkanContext& vulkanContext, uint32_t batch, uint32_t depth, uint32_t height, uint32_t width,
        VkBufferUsageFlags dataUsageFlags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VkBufferUsageFlags dimUsageFlags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VkMemoryPropertyFlags dataMemoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                     VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        : TensorElement<DataType>(vulkanContext, batch, depth, height, width, dataUsageFlags, dimUsageFlags,
                                  dataMemoryProperties) {}

    /**
     * @brief Construct a TensorElementSinglePath with specified dimensions
     *
     * @param vulkanContext The Vulkan context for buffer creation
     * @param dimensions Vector containing tensor dimensions
     * @param dataUsageFlags Vulkan usage flags for the data buffer
     * @param dimUsageFlags Vulkan usage flags for the dimensions buffer
     */
    TensorElementSinglePath(VulkanContext& vulkanContext, const std::vector<uint32_t>& dimensions,
                            VkBufferUsageFlags dataUsageFlags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VkBufferUsageFlags dimUsageFlags = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                                               VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VkMemoryPropertyFlags dataMemoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        : TensorElement<DataType>(vulkanContext, dimensions, dataUsageFlags, dimUsageFlags, dataMemoryProperties) {}

    virtual ~TensorElementSinglePath() = default;

    // Override setup to use single buffers instead of creating multiple paths
    virtual void _setup(VulkanContext& vulkanContext, uint32_t numberPaths) override {
        TensorElement<DataType>::_setup(vulkanContext, 1);
    }

    virtual const char* getType() const override { return "TensorElementSinglePath"; }

    bool isSinglePathStorage() const override { return true; }

    VulkanBuffer<DataType>& getDataBuffer(uint32_t pathId = -1) override {
        return TensorElement<DataType>::getDataBuffer(0);
    }

    // Override buffer accessors to return single buffers for any pathId
    virtual VkBuffer& getDataVkBuffer(uint32_t pathId = -1) override {
        return TensorElement<DataType>::getDataVkBuffer(0);
    }

    virtual VkBuffer& getVkBuffer(uint32_t pathId = -1) override { return TensorElement<DataType>::getVkBuffer(0); }

private:
};

} // namespace klartraum

#endif // KLARTRAUM_COMPUTEGRAPH_TENSORELEMENT_HPP
