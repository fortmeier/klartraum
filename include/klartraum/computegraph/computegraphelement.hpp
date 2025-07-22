#ifndef KLARTRAUM_COMPUTEGRAPHELEMENT_HPP
#define KLARTRAUM_COMPUTEGRAPHELEMENT_HPP

#include <map>
#include <memory>

#include "klartraum/vulkan_context.hpp"


namespace klartraum {

class ComputeGraphElement;

typedef std::shared_ptr<ComputeGraphElement> ComputeGraphElementPtr;

class ComputeGraph;

/**
 * @brief Abstract base class representing an element in a compute graph.
 * 
 * ComputeGraphElement serves as the foundation for building computational graphs
 * in the Klartraum framework. Each element can have multiple inputs and outputs,
 * forming a directed acyclic graph structure for computation pipelines.
 * 
 * 
 * @note This class is designed to work with shared_ptr (ComputeGraphElementPtr)
 *       for automatic memory management and safe reference handling.
 *       Generally, you should not create instances of this class directly,
 *       but rather derive from it to implement specific compute graph elements and
 *       use the VulkanContext::create to instantiate them.
 * 
 * @see ComputeGraph which is used to compile a compute graph from these elements.
 */
class ComputeGraphElement {
public:
    friend class ComputeGraph;

    // maybe order of arguments should be changed to index, input, slot
    // or index should be a template parameter
    [[deprecated("Consider using template version setInput<index>(input, slot) for better type safety and clarity")]]
    void setInput(ComputeGraphElementPtr input, int index = 0, int slot = -1) {
        if(slot == -1) {
            checkInput(input, index);
        } else {
            checkInput(input->getInputElement(slot), index);
        }
        inputs[index] = input;
        srcOutputSlots[index] = slot;
    }

    /**
     * @brief Sets the input element for this ComputeGraphElement at the specified index.
     * 
     * If slot is used, this indicates that the input is not given by input directly,
     * but rather by the slot element of the given element.
     * 
     * If slot is specified, retrieves the input element from the provided input's slot and checks its validity.
     * Updates the internal input and source output slot mappings.
     * 
     * @tparam index The index at which to set the input (default is 0).
     * @param input The ComputeGraphElementPtr to set as input.
     * @param slot The slot of the input element to use (default is -1, meaning no slot). See explanation of slots above.
     */    
    template<int index = 0>
    void setInput(ComputeGraphElementPtr input, int slot = -1) {
        setInput(input, index, slot);
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