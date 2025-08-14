#ifndef KLARTRAUM_NOOP_HPP
#define KLARTRAUM_NOOP_HPP

#include "klartraum/computegraph/computegraphelement.hpp"

namespace klartraum {

/**
 * @brief NoOp (No Operation) element that passes inputs through unchanged
 * 
 * This element performs no computation and simply passes its inputs through
 * to its outputs. It's useful for representing operations like ONNX Constant
 * nodes that provide data without computation, or for debugging/testing
 * purposes.
 */
class NoOp : public ComputeGraphElement {
public:
    /**
     * @brief Construct a new NoOp element
     * @param vulkanContext The Vulkan context for this element
     */
    NoOp(VulkanContext& vulkanContext) {}
    
    virtual ~NoOp() = default;



    // ComputeGraphElement interface
    virtual void checkInput(ComputeGraphElementPtr input, int index = 0) override {
        // NoOp doesn't need to check inputs - it just passes data through
    }

    virtual void _setup(VulkanContext& vulkanContext, uint32_t numberPaths) override {
        // NoOp doesn't need any setup - it just passes data through
    }

    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) override {
        // NoOp doesn't record any commands - it's a pass-through
        // The data flows directly from inputs to outputs without GPU operations
    }

    virtual const char* getType() const override {
        return "NoOp";
    }

private:
};

} // namespace klartraum

#endif // KLARTRAUM_NOOP_HPP
