#ifndef KLARTRAUM_BUFFERELEMENT_HPP
#define KLARTRAUM_BUFFERELEMENT_HPP

#include "klartraum/drawgraph/drawgraphelement.hpp"

namespace klartraum {


class BufferElementInterface : public DrawGraphElement {
public:
    virtual void _setup(VulkanKernel& vulkanKernel, uint32_t numberPaths) = 0;

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
    BufferElement(VulkanKernel& vulkanKernel, uint32_t numberElements):
        vulkanKernel(vulkanKernel), numberElements(numberElements) {
    }

    virtual void _setup(VulkanKernel& vulkanKernel, uint32_t numberPaths) {
        buffers.reserve(numberPaths);
        for(uint32_t i = 0; i < numberPaths; i++) {
            buffers.emplace_back(vulkanKernel, numberElements);
        }
    };

    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) {};

    virtual const char* getName() const {
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

    virtual VkBuffer& getVkBuffer(uint32_t pathId) {
        return buffers[pathId].getBuffer();
    };

private:
    VulkanKernel& vulkanKernel;
    uint32_t numberPaths = 0;
    uint32_t numberElements = 0;
    std::vector<BufferType> buffers;

};


template<typename BufferType>
class BufferElementSinglePath : public TemplatedBufferElementInterface<BufferType> {
public:
    template<typename... Args>
    BufferElementSinglePath(Args&&... args) : buffer(std::forward<Args>(args)...) {}

    virtual void _setup(VulkanKernel& vulkanKernel, uint32_t numberPaths) {};

    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) {};

    virtual const char* getName() const {
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