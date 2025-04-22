#ifndef KLARTRAUM_DRAWGRAPHELEMENT_HPP
#define KLARTRAUM_DRAWGRAPHELEMENT_HPP

#include <map>
#include <memory>

#include "klartraum/vulkan_kernel.hpp"

namespace klartraum {

class DrawGraphElement;

typedef std::shared_ptr<DrawGraphElement> DrawGraphElementPtr;

class DrawGraph;

class DrawGraphElement {
public:
    friend class DrawGraph;

    void setInput(DrawGraphElementPtr input, int index = 0) {
        checkInput(input, index);
        inputs[index] = input;
    }

    virtual void checkInput(DrawGraphElementPtr input, int index = 0) {
        throw std::runtime_error("checkInput not implemented for this element");
    }

    //virtual DrawGraphElement& get_output() = 0;

    void setWaitFor(uint32_t pathId, VkSemaphore semaphore) {
        renderWaitSemaphores[pathId] = semaphore;
    }

    virtual void _setup(VulkanKernel& vulkanKernel, uint32_t numberPaths) {};

    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) {};

    virtual const char* getName() const = 0;

    std::map<int, DrawGraphElementPtr> inputs;
    std::map<int, VkSemaphore> renderWaitSemaphores;
    
    private:
    // these are updated by the DrawGraph, do not set them manually
    // it is important to reset them before destroying the graph
    // otherwise we will have dangling pointers in the graph
    std::vector<DrawGraphElementPtr> outputs;

};



} // namespace klartraum

#endif // KLARTRAUM_DRAWGRAPHELEMENT_HPP