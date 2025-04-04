#ifndef KLARTRAUM_DRAWGRAPHELEMENT_HPP
#define KLARTRAUM_DRAWGRAPHELEMENT_HPP

#include <map>
#include <memory>

#include "klartraum/vulkan_kernel.hpp"

namespace klartraum {

class DrawGraphElement;

typedef std::shared_ptr<DrawGraphElement> DrawGraphElementPtr;

class DrawGraphElement {
public:
    virtual void setInput(DrawGraphElementPtr input, int index = 0) {
        inputs[index] = input;
    }
    //virtual DrawGraphElement& get_output() = 0;

    void setWaitFor(uint32_t pathId, VkSemaphore semaphore) {
        renderWaitSemaphores[pathId] = semaphore;
    }

    virtual void _setup(VulkanKernel& vulkanKernel, uint32_t numberPath) {};

    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) {};

    virtual const char* getName() const = 0;

    std::map<int, DrawGraphElementPtr> inputs;
    std::map<int, VkSemaphore> renderWaitSemaphores;
};



} // namespace klartraum

#endif // KLARTRAUM_DRAWGRAPHELEMENT_HPP