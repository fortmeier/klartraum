#ifndef KLARTRAUM_TRANSFORMATION_HPP
#define KLARTRAUM_TRANSFORMATION_HPP

#include <vulkan/vulkan.h>

#include <array>

#include "klartraum/drawgraph/drawgraphelement.hpp"
#include "klartraum/vulkan_helpers.hpp"

namespace klartraum {

template <typename A, typename R>
class BufferTransformation : public DrawGraphElement {
public:
    BufferTransformation(VulkanKernel &vulkanKernel, const std::string &shaderPath)
    {
        this->vulkanKernel = &vulkanKernel;
        createDescriptorPool();
        createComputeDescriptorSetLayout();
        
        createComputePipeline(shaderPath);
    
        //createSyncObjects();
    }

    virtual ~BufferTransformation() {
        vkDestroyPipelineLayout(vulkanKernel->getDevice(), computePipelineLayout, nullptr);
        vkDestroyPipeline(vulkanKernel->getDevice(), computePipeline, nullptr);
        vkDestroyDescriptorSetLayout(vulkanKernel->getDevice(), computeDescriptorSetLayout, nullptr);
        vkDestroyDescriptorPool(vulkanKernel->getDevice(), descriptorPool, nullptr);
    };

    virtual void _setup(VulkanKernel& vulkanKernel, uint32_t numberPaths) {
        this->numberPaths = numberPaths;
        this->vulkanKernel = &vulkanKernel;

        
        uint32_t inputSize = getInput().getSize();
        if (inputSize == 0) {
            throw std::runtime_error("input size is 0!");
        }
        groupCountX = inputSize;

        for(uint32_t i = 0; i < numberPaths; i++) {
            outputBuffers.emplace_back(vulkanKernel, inputSize);
            createComputeDescriptorSets(i);
        }
        
    };

    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSet, 0, 0);
        
        vkCmdDispatch(commandBuffer, groupCountX, groupCountY, groupCountZ);        

    };

    virtual void checkInput(DrawGraphElementPtr input, int index = 0) {
        BufferElement<A>* bufferSrc = std::dynamic_pointer_cast<BufferElement<A>>(input).get();
        if (bufferSrc == nullptr) {
            throw std::runtime_error("input is not a fitting BufferElement!");
        }
    }

    A& getInput(uint32_t pathId = 0) {
        return dynamic_cast<BufferElement<A>*>(inputs[0].get())->getBuffer();
    }

    virtual const char* getName() const {
        return "BufferTransformation";
    }    

    // VulkanOperationResult* operator()(A& a, B& b, R& result)
    // {
    //     if(a.getSize() != b.getSize() || a.getSize() != result.getSize())
    //     {
    //         std::stringstream ss;
    //         ss << "BufferTransformation::operator() operand size mismatch: a:" << a.getSize() << " b:" << b.getSize() << " result:" << result.getSize();
    //         throw std::runtime_error(ss.str());
    //     }
    //     groupCountX = a.getSize();
    //     createComputeDescriptorSets(a, b, result);
    //     VulkanOperationResult* res = dynamic_cast<VulkanOperationResult*>(this);
    //     return res;
    // }


    A& getOutputBuffer(uint32_t pathId = 0) {
        return outputBuffers[pathId];
    }

private:
    VkDescriptorPool descriptorPool;
    VkDescriptorSet computeDescriptorSet;
    VkDescriptorSetLayout computeDescriptorSetLayout;
    VkPipelineLayout computePipelineLayout;
    VkPipeline computePipeline;

    VulkanKernel* vulkanKernel;

    uint32_t groupCountX = 1;
    uint32_t groupCountY = 1;
    uint32_t groupCountZ = 1;

    uint32_t numberPaths = 1;

    std::vector<R> outputBuffers;

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

    void createComputeDescriptorSets(uint32_t pathId)
    {
        auto& device = vulkanKernel->getDevice();
        auto& config = vulkanKernel->getConfig();

        A& a = getInput(pathId);
        R& r = outputBuffers[0];
    
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

        VkDescriptorBufferInfo storageBufferInfoResult{};
        storageBufferInfoResult.buffer = r.getBuffer();
        storageBufferInfoResult.offset = 0;
        storageBufferInfoResult.range = r.getBufferMemSize();

        std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = computeDescriptorSet;
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &storageBufferInfoA;

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = computeDescriptorSet;
        descriptorWrites[1].dstBinding = 2;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pBufferInfo = &storageBufferInfoResult;

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

} // namespace klartraum
#endif // KLARTRAUM_TRANSFORMATION_HPP