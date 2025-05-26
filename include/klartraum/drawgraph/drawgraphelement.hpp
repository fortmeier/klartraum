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

    void setInput(DrawGraphElementPtr input, int index = 0, int slot = -1) {
        if(slot == -1) {
            checkInput(input, index);
        } else {
            checkInput(input->getInputElement(slot), index);
        }
        inputs[index] = input;
        srcOutputSlots[index] = slot;
    }

    virtual void checkInput(DrawGraphElementPtr input, int index = 0) {
        throw std::runtime_error("checkInput not implemented for this element");
    }

    DrawGraphElementPtr getInputElement(int index = 0) {
        // if a slot is set, we need to get the element from
        // the inputs of the input element
        if (srcOutputSlots[index] != -1) {
            return inputs[index]->getInputElement(srcOutputSlots[index]);
        }
        // otherwise we can just return the input at the index
        return inputs[index];
    }


    void setWaitFor(uint32_t pathId, VkSemaphore semaphore) {
        renderWaitSemaphores[pathId] = semaphore;
    }

    virtual void _setup(VulkanKernel& vulkanKernel, uint32_t numberPaths) {
        initialized = true;
    };

    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) {
        if (!initialized) {
            throw std::runtime_error("DrawGraphElement not initialized");
        }
    }

    virtual const char* getName() const = 0;

    std::map<int, DrawGraphElementPtr> inputs; //TODO: really should be private
    std::map<int, VkSemaphore> renderWaitSemaphores; //TODO: really should be private
    std::map<int, int> srcOutputSlots; //TODO: really should be private
    
    protected:
    bool initialized = false;

    private:
    // these are updated by the DrawGraph, do not set them manually
    // it is important to reset them before destroying the graph
    // otherwise we will have dangling pointers in the graph
    std::vector<DrawGraphElementPtr> outputs;

};



} // namespace klartraum

#endif // KLARTRAUM_DRAWGRAPHELEMENT_HPP