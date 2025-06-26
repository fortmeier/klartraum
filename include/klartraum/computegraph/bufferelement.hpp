#ifndef KLARTRAUM_BUFFERELEMENT_HPP
#define KLARTRAUM_BUFFERELEMENT_HPP

#include "klartraum/computegraph/computegraphelement.hpp"

namespace klartraum {


class BufferElementInterface : public ComputeGraphElement {
public:
    virtual void _setup(VulkanContext& vulkanContext, uint32_t numberPaths) = 0;

    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) = 0;

    virtual const char* getType() const {
        return "BufferElement";
    }

    virtual size_t getBufferMemSize() const = 0;

    virtual VkBuffer& getVkBuffer(uint32_t pathId) = 0;

private:

};

template<typename BufferType>
class TemplatedBufferElementInterface : public BufferElementInterface {
public:
    virtual BufferType& getBuffer(uint32_t pathId) = 0;
};

template<typename BufferType>
class BufferElement : public TemplatedBufferElementInterface<BufferType> {
public:
    BufferElement(VulkanContext& vulkanContext, uint32_t numberElements):
        vulkanContext(vulkanContext), numberElements(numberElements) {
    }

    BufferElement(VulkanContext& vulkanContext, uint32_t numberElements, VkBufferUsageFlags flags):
        vulkanContext(vulkanContext), numberElements(numberElements), bufferUsageFlags(flags) {
    }

    virtual void _setup(VulkanContext& vulkanContext, uint32_t numberPaths) {
        buffers.reserve(numberPaths);
        for(uint32_t i = 0; i < numberPaths; i++) {
            if (bufferUsageFlags == VK_BUFFER_USAGE_FLAG_BITS_MAX_ENUM)
            {
                buffers.emplace_back(vulkanContext, numberElements);
            }
            else
            {
                buffers.emplace_back(vulkanContext, numberElements, bufferUsageFlags);
            }
            
        }
    };

    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) {
        if (recordToZero) {
            buffers[pathId]._recordZero(commandBuffer);
        }
    };

    virtual const char* getType() const {
        return "BufferElement";
    }

    virtual size_t getBufferMemSize() const {
        return buffers[0].getBufferMemSize();
    }

    virtual BufferType& getBuffer(uint32_t pathId) {
        return buffers[pathId];
    }

    void zero() {
        for (auto& buffer : buffers) {
            buffer.zero();
        }
    }

    void setRecordToZero(bool _setToZero) {
        recordToZero = _setToZero;
    }

    virtual VkBuffer& getVkBuffer(uint32_t pathId) {
        return buffers[pathId].getBuffer();
    };

private:
    VulkanContext& vulkanContext;
    uint32_t numberPaths = 0;
    uint32_t numberElements = 0;
    std::vector<BufferType> buffers;
    bool recordToZero = false;
    VkBufferUsageFlags bufferUsageFlags = VK_BUFFER_USAGE_FLAG_BITS_MAX_ENUM;

};


template<typename BufferType>
class BufferElementSinglePath : public TemplatedBufferElementInterface<BufferType> {
public:
    template<typename... Args>
    BufferElementSinglePath(Args&&... args) : buffer(std::forward<Args>(args)...) {}

    virtual void _setup(VulkanContext& vulkanContext, uint32_t numberPaths) {};

    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) {};

    virtual const char* getType() const {
        return "BufferElementSinglePath";
    }

    virtual size_t getBufferMemSize() const {
        return buffer.getBufferMemSize();
    }

    BufferType& getBuffer(uint32_t pathId = 0) {
        return buffer;
    }

    virtual VkBuffer& getVkBuffer(uint32_t pathId = 0) {
        return buffer.getBuffer();
    };

private:
    BufferType buffer;

};

} // namespace klartraum

#endif // KLARTRAUM_BUFFERELEMENT_HPP