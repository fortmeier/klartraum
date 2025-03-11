#ifndef KLARTRAUM_VULKAN_OPERATION_HPP
#define KLARTRAUM_VULKAN_OPERATION_HPP

#include <vulkan/vulkan.h>

#include <array>

#include "klartraum/vulkan_helpers.hpp"

namespace klartraum {

class VulkanOperationResult {
public:
    virtual void _record(VkCommandBuffer commandBuffer) = 0;
};

template <typename A, typename B, typename R>
class VulkanOperator : public VulkanOperationResult {
public:
    VulkanOperator(VulkanKernel &vulkanKernel, const std::string &shaderPath)
    {
        this->vulkanKernel = &vulkanKernel;
        createDescriptorPool();
        createComputeDescriptorSetLayout();
        
        createComputePipeline(shaderPath);
    
        //createSyncObjects();
    }

    virtual ~VulkanOperator() {};

    VulkanOperationResult* operator()(A& a, B& b, R& result)
    {
        createComputeDescriptorSets(a, b, result);
        VulkanOperationResult* res = dynamic_cast<VulkanOperationResult*>(this);
        return res;
    }

    virtual void _record(VkCommandBuffer commandBuffer) {
        std::cout << "recording" << std::endl;
        
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSet, 0, 0);
        
        vkCmdDispatch(commandBuffer, 7, 1, 1);

    }

private:
    VkDescriptorPool descriptorPool;
    VkDescriptorSet computeDescriptorSet;
    VkDescriptorSetLayout computeDescriptorSetLayout;
    VkPipelineLayout computePipelineLayout;
    VkPipeline computePipeline;

    VulkanKernel* vulkanKernel;

    void createDescriptorPool() {
        auto& device = vulkanKernel->getDevice();
        auto& config = vulkanKernel->getConfig();
    
        std::array<VkDescriptorPoolSize, 3> poolSizes{};
        // A
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[0].descriptorCount = 3; //static_cast<uint32_t>(swapChainSize);
        
        // B
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[1].descriptorCount = 3; //static_cast<uint32_t>(swapChainSize);
    
        // Result
        poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[2].descriptorCount = 3; //static_cast<uint32_t>(swapChainSize);
    
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = poolSizes.size();
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = 3;
    
        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor pool!");
        }
    }    

    void createComputeDescriptorSetLayout()
    {
        auto& device = vulkanKernel->getDevice();

        std::array<VkDescriptorSetLayoutBinding, 3> layoutBindings{};
    
        // A binding
        layoutBindings[0].binding = 0;
        layoutBindings[0].descriptorCount = 1;
        layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[0].pImmutableSamplers = nullptr;
        layoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        // B binding
        layoutBindings[1].binding = 1;
        layoutBindings[1].descriptorCount = 1;
        layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[1].pImmutableSamplers = nullptr;
        layoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        // Result binding
        layoutBindings[2].binding = 2;
        layoutBindings[2].descriptorCount = 1;
        layoutBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[2].pImmutableSamplers = nullptr;
        layoutBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = layoutBindings.size();
        layoutInfo.pBindings = layoutBindings.data();
        
        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &computeDescriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute descriptor set layout!");
        }
    }

    void createComputeDescriptorSets(A& a, B& b, R& r)
    {
        auto& device = vulkanKernel->getDevice();
        auto& config = vulkanKernel->getConfig();
    
        //VkDescriptorSetLayout layout;
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &computeDescriptorSetLayout;
    
        //computeDescriptorSet;
        VkResult result = vkAllocateDescriptorSets(device, &allocInfo, &computeDescriptorSet);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }
    
        VkDescriptorBufferInfo storageBufferInfoA{};
        storageBufferInfoA.buffer = a.getBuffer();
        storageBufferInfoA.offset = 0;
        storageBufferInfoA.range = a.getBufferMemSize();

        VkDescriptorBufferInfo storageBufferInfoB{};
        storageBufferInfoB.buffer = b.getBuffer();
        storageBufferInfoB.offset = 0;
        storageBufferInfoB.range = b.getBufferMemSize();

        VkDescriptorBufferInfo storageBufferInfoResult{};
        storageBufferInfoResult.buffer = r.getBuffer();
        storageBufferInfoResult.offset = 0;
        storageBufferInfoResult.range = r.getBufferMemSize();

        std::array<VkWriteDescriptorSet, 3> descriptorWrites{};

        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = computeDescriptorSet;
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &storageBufferInfoA;

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = computeDescriptorSet;
        descriptorWrites[1].dstBinding = 0;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pBufferInfo = &storageBufferInfoB;

        descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[2].dstSet = computeDescriptorSet;
        descriptorWrites[2].dstBinding = 0;
        descriptorWrites[2].dstArrayElement = 0;
        descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[2].descriptorCount = 1;
        descriptorWrites[2].pBufferInfo = &storageBufferInfoResult;

        vkUpdateDescriptorSets(device, descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
    }

    void createComputePipeline(const std::string &shaderPath)
    {
        auto& device = vulkanKernel->getDevice();

        auto computeShaderCode = readFile(shaderPath);
    
        VkShaderModule computeShaderModule = createShaderModule(computeShaderCode, vulkanKernel->getDevice());
        
        VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
        computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        computeShaderStageInfo.module = computeShaderModule;
        computeShaderStageInfo.pName = "main";
    
    
        
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 2;
    
        VkDescriptorSetLayout combinedLayouts[] = {computeDescriptorSetLayout, vulkanKernel->getCamera().getDescriptorSetLayout()};
        pipelineLayoutInfo.pSetLayouts = combinedLayouts;
        
        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &computePipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline layout!");
        }
    
    
        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.layout = computePipelineLayout;
        pipelineInfo.stage = computeShaderStageInfo;
        
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &computePipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline!");
        }
    
        vkDestroyShaderModule(device, computeShaderModule, nullptr);        
    }

};

class VulkanOperationContext {
public:
    VulkanOperationContext(VkCommandBuffer commandBuffer, VkDevice& device) : device(device) {
        this->commandBuffer = commandBuffer;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        // fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    
        if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS) {
            throw std::runtime_error("failed to create fence!");
        }
    }
    virtual ~VulkanOperationContext() {
        vkDestroyFence(device, fence, nullptr);        
    }

    void operator()(VulkanOperationResult* result)
    {
        results.push_back(result);
        std::cout << "adding" << std::endl;
    }
    
    void eval(VkQueue graphicsQueue)
    {
        vkResetCommandBuffer(this->commandBuffer, 0);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0; // Optional
        beginInfo.pInheritanceInfo = nullptr; // Optional
    
        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        std::cout << "evaluating" << std::endl;
        for (auto result : results)
        {
            result->_record(commandBuffer);
        }

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to record command buffer!");
        }

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    
        // submitInfo.waitSemaphoreCount = 1;
        // submitInfo.pWaitSemaphores = waitSemaphores;
        // submitInfo.pWaitDstStageMask = waitStages;
    
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
    
        // submitInfo.signalSemaphoreCount = 1;
        // submitInfo.pSignalSemaphores = &renderFinishedSemaphores[currentFrame];

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, fence) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit inFlightFence");
        }

        vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
        return;
    }
private:
    VkDevice& device;

    VkCommandBuffer commandBuffer;

    VkFence fence;

    std::vector<VulkanOperationResult*> results;
};

} // namespace klartraum
#endif // KLARTRAUM_VULKAN_OPERATION_HPP