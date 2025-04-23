#ifndef KLARTRAUM_BUFFERELEMENT_HPP
#define KLARTRAUM_BUFFERELEMENT_HPP

#include "klartraum/drawgraph/drawgraphelement.hpp"

namespace klartraum {

template<typename BufferType>
class BufferElement : public DrawGraphElement {
public:
    template<typename... Args>
    BufferElement(Args&&... args) : buffer(std::forward<Args>(args)...) {}

    virtual void _setup(VulkanKernel& vulkanKernel, uint32_t numberPaths) {};

    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) {};

    virtual const char* getName() const {
        return "BufferElement";
    }

    BufferType& getBuffer() {
        return buffer;
    }

private:
    BufferType buffer;

};



} // namespace klartraum

#endif // KLARTRAUM_BUFFERELEMENT_HPP