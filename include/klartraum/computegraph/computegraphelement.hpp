#ifndef KLARTRAUM_COMPUTEGRAPHELEMENT_HPP
#define KLARTRAUM_COMPUTEGRAPHELEMENT_HPP

#include <map>
#include <memory>

#include "klartraum/vulkan_context.hpp"

namespace klartraum {

class ComputeGraphElement;

typedef std::shared_ptr<ComputeGraphElement> ComputeGraphElementPtr;

class ComputeGraph;

class ComputeGraphElement {
public:
    friend class ComputeGraph;

    void setInput(ComputeGraphElementPtr input, int index = 0, int slot = -1) {
        if(slot == -1) {
            checkInput(input, index);
        } else {
            checkInput(input->getInputElement(slot), index);
        }
        inputs[index] = input;
        srcOutputSlots[index] = slot;
    }

    virtual void checkInput(ComputeGraphElementPtr input, int index = 0) {
        throw std::runtime_error("checkInput not implemented for this element");
    }

    ComputeGraphElementPtr getInputElement(int index = 0) {
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

    virtual void _setup(VulkanContext& vulkanContext, uint32_t numberPaths) {
        initialized = true;
    };

    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) {
        if (!initialized) {
            throw std::runtime_error("ComputeGraphElement not initialized");
        }
    }

    virtual const char* getType() const = 0;

    virtual const char* getName() const {
        return name.c_str();
    }
    
    virtual std::map<int, ComputeGraphElementPtr> getInputs() const {
        return inputs;
    }

    void setName(const std::string& name) {
        this->name = name;
    }

protected:
    std::map<int, ComputeGraphElementPtr> inputs; //TODO: really should be private
    std::map<int, VkSemaphore> renderWaitSemaphores; //TODO: really should be private
    std::map<int, int> srcOutputSlots; //TODO: really should be private
    
    bool initialized = false;

    std::string name;

private:
    // these are updated by the ComputeGraph, do not set them manually
    // it is important to reset them before destroying the graph
    // otherwise we will have dangling pointers in the graph
    std::vector<ComputeGraphElementPtr> outputs;

};



} // namespace klartraum

#endif // KLARTRAUM_COMPUTEGRAPHELEMENT_HPP