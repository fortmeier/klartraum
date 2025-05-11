#ifndef KLARTRAUM_BUFFERELEMENT_HPP
#define KLARTRAUM_BUFFERELEMENT_HPP

#include "klartraum/drawgraph/drawgraphelement.hpp"

namespace klartraum {


class BufferElementInterface : public DrawGraphElement {
public:
    virtual void _setup(VulkanKernel& vulkanKernel, uint32_t numberPaths) = 0;

    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) = 0;

    virtual const char* getName() const {
        return "BufferElement";
    }

    virtual size_t getBufferMemSize() const = 0;

    virtual VkBuffer& getVkBuffer() = 0;

    // BufferType& getBuffer() {
    //     return buffer;
    // }

private:
    // BufferType buffer;

};

template<typename BufferType>
class BufferElement : public BufferElementInterface {
public:
    template<typename... Args>
    BufferElement(Args&&... args) : buffer(std::forward<Args>(args)...) {}

    virtual void _setup(VulkanKernel& vulkanKernel, uint32_t numberPaths) {};

    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) {};

    virtual const char* getName() const {
        return "BufferElement";
    }

    virtual size_t getBufferMemSize() const {
        return buffer.getBufferMemSize();
    }

    BufferType& getBuffer() {
        return buffer;
    }

    virtual VkBuffer& getVkBuffer() {
        return buffer.getBuffer();
    };

private:
    BufferType buffer;

};



} // namespace klartraum

#endif // KLARTRAUM_BUFFERELEMENT_HPP