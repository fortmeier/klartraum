#ifndef KLARTRAUM_COPYBUFFER_HPP
#define KLARTRAUM_COPYBUFFER_HPP

#include "klartraum/computegraph/computegraphelement.hpp"
#include "klartraum/computegraph/bufferelement.hpp"

namespace klartraum {

/**
 * @brief CopyBuffer element that copies data from source buffer to destination buffer
 * 
 * This element performs a direct buffer copy operation using vkCmdCopyBuffer.
 * Its main purpose is to copy a buffer so that consecutive operations on the data
 * do not conflict between different operations.
 */
class CopyBuffer : public ComputeGraphElement {
public:
    /**
     * @brief Construct a new CopyBuffer element
     * @param vulkanContext The Vulkan context for this element
     * @param bufferSize Size in bytes to copy (0 means copy entire buffer)
     */
    CopyBuffer(VulkanContext& vulkanContext) 
        : vulkanContext(vulkanContext) {}

    virtual ~CopyBuffer() = default;



    // ComputeGraphElement interface
    virtual void checkInput(ComputeGraphElementPtr input, int index = 0) override {
        if (index != 0 && index != 1) {
            throw std::runtime_error("CopyBuffer only accepts input at index 0 (source buffer) and index 1 (destination buffer)");
        }
        
        auto bufferInput = std::dynamic_pointer_cast<BufferElementInterface>(input);
        if (!bufferInput) {
            throw std::runtime_error("CopyBuffer input must be a BufferElementInterface");
        }
    }

    virtual void _setup(VulkanContext& vulkanContext, uint32_t numberPaths) override {
        this->numberPaths = numberPaths;
        ComputeGraphElement::_setup(vulkanContext, numberPaths);
    }

    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) override {
        if (!initialized) {
            throw std::runtime_error("CopyBuffer not initialized");
        }

        VkBuffer srcBuffer = getInputElement<BufferElementInterface>(0)->getVkBuffer(pathId);
        VkBuffer dstBuffer = getInputElement<BufferElementInterface>(1)->getVkBuffer(pathId);

        size_t srcSize = getInputElement<BufferElementInterface>(0)->getBufferMemSize();
        size_t dstSize = getInputElement<BufferElementInterface>(1)->getBufferMemSize();

        if (srcSize != dstSize) {
            throw std::runtime_error("Source and destination buffers must have the same size for CopyBuffer");
        }

        // Set up the copy region
        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = 0;
        copyRegion.size = srcSize;

        // Perform the buffer copy
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        // Add memory barrier to ensure copy is complete before subsequent operations
        VkMemoryBarrier memoryBarrier{};
        memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        memoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1, &memoryBarrier,
            0, nullptr,
            0, nullptr
        );
    }

    virtual const char* getType() const override {
        return "CopyBuffer";
    }

private:
    VulkanContext& vulkanContext;
    uint32_t numberPaths = 0;
};

} // namespace klartraum

#endif // KLARTRAUM_COPYBUFFER_HPP
