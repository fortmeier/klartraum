#ifndef KLARTRAUM_TRANSFORMATION_HPP
#define KLARTRAUM_TRANSFORMATION_HPP

#include <vulkan/vulkan.h>

#include <array>

#include "klartraum/drawgraph/drawgraphelement.hpp"
#include "klartraum/vulkan_helpers.hpp"
#include "klartraum/drawgraph/bufferelement.hpp"

namespace klartraum {

template <typename A, typename R, typename U = void, typename P = void>
class BufferTransformation : public DrawGraphElement {
public:
    BufferTransformation(VulkanKernel &vulkanKernel, const std::string &shaderPath) :
        shaderPath(shaderPath) 
    {
        this->vulkanKernel = &vulkanKernel;
        
        if constexpr (!std::is_void<U>::value) {
            ubo = new U();
        }



    }

    virtual ~BufferTransformation() {
        vkDestroyPipelineLayout(vulkanKernel->getDevice(), computePipelineLayout, nullptr);
        vkDestroyPipeline(vulkanKernel->getDevice(), computePipeline, nullptr);
        vkDestroyDescriptorSetLayout(vulkanKernel->getDevice(), computeDescriptorSetLayout, nullptr);
        vkDestroyDescriptorPool(vulkanKernel->getDevice(), descriptorPool, nullptr);

        if constexpr (!std::is_void<U>::value) {
            delete ubo;
        }        
    };

    void setPushConstants(const std::vector<P>& pushConstants) {
        if constexpr (!std::is_void<P>::value) {
            this->pushConstants = pushConstants;
        }
    }

    virtual void _setup(VulkanKernel& vulkanKernel, uint32_t numberPaths) {
        this->numberPaths = numberPaths;
        this->vulkanKernel = &vulkanKernel;

        if constexpr (!std::is_void<U>::value) {
            ubo->_setup(vulkanKernel, numberPaths);
        }

        createDescriptorPool();
        createComputeDescriptorSetLayout();
        createComputePipeline();
        
        uint32_t inputSize = getInput().getSize();
        if (inputSize == 0) {
            throw std::runtime_error("input size is 0!");
        }

        // If groupCountX is 0, use input size, otherwise use groupCountX
        groupCountX = groupCountX > 0 ? groupCountX : inputSize; 

        computeDescriptorSets.resize(numberPaths);
        for(uint32_t i = 0; i < numberPaths; i++) {
            // Use custom output size if set, otherwise use input size
            uint32_t outputSize = customOutputSize > 0 ? customOutputSize : inputSize;
            outputBuffers.emplace_back(vulkanKernel, outputSize);
            createComputeDescriptorSets(i);
            
            if constexpr (!std::is_void<U>::value) {
                ubo->update(i);
            }
        }
        

    };

    void setCustomOutputSize(uint32_t size) {
        customOutputSize = size;
    }

    uint32_t getCustomOutputSize() const {
        return customOutputSize;
    }

    void setGroupCountX(uint32_t count) {
        groupCountX = count;
    }

    uint32_t getGroupCountX() const {
        return groupCountX;
    }

    void setGroupCountY(uint32_t count) {
        groupCountY = count;
    }

    uint32_t getGroupCountY() const {
        return groupCountY;
    }

    void setGroupCountZ(uint32_t count) {
        groupCountZ = count;
    }

    uint32_t getGroupCountZ() const {
        return groupCountZ;
    }

    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) {
        if constexpr (std::is_void<U>::value) {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSets[pathId], 0, 0);
        } else {
            std::vector<VkDescriptorSet> descriptorSets = {computeDescriptorSets[pathId], ubo->getDescriptorSets()[pathId]};
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, (uint32_t)descriptorSets.size(), descriptorSets.data(), 0, 0);
        }

        if constexpr (std::is_void<P>::value) {
            vkCmdDispatch(commandBuffer, groupCountX, groupCountY, groupCountZ);
        } else {
            for (const auto& pushConstant : pushConstants) {
                vkCmdPushConstants(commandBuffer, computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(P), &pushConstant);
                vkCmdDispatch(commandBuffer, groupCountX, groupCountY, groupCountZ);
                VkMemoryBarrier memoryBarrier{};
                memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                memoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

                vkCmdPipelineBarrier(
                    commandBuffer,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    1, &memoryBarrier,
                    0, nullptr,
                    0, nullptr
                );
            }
        }
    };

    virtual void checkInput(DrawGraphElementPtr input, int index = 0) {
        BufferElement<A>* bufferSrc = std::dynamic_pointer_cast<BufferElement<A>>(input).get();
        if (bufferSrc == nullptr) {
            throw std::runtime_error("input is not a fitting BufferElement!");
        }
    }

    A& getInput(uint32_t pathId = 0) {
        // TODO why inputs[0], and not inputs[pathId]?
        return dynamic_cast<BufferElement<A>*>(inputs[0].get())->getBuffer();
    }

    virtual const char* getName() const {
        return "BufferTransformation";
    }    

    R& getOutputBuffer(uint32_t pathId = 0) {
        return outputBuffers[pathId];
    }

    std::conditional_t<!std::is_void<U>::value, U*, void*> getUbo() {
        if constexpr (!std::is_void<U>::value) {
            return ubo;
        } else {
            return nullptr;
        }
    }

    void addScratchBufferElement(std::shared_ptr<BufferElementInterface> bufferElement) {
        otherInputs.push_back(bufferElement);
        //outputBuffers[pathId].addScratchBuffer(size);
    }

private:
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> computeDescriptorSets;
    VkDescriptorSetLayout computeDescriptorSetLayout;
    VkPipelineLayout computePipelineLayout;
    VkPipeline computePipeline;

    VulkanKernel* vulkanKernel;

    const std::string shaderPath;

    uint32_t groupCountX = 0;
    uint32_t groupCountY = 1;
    uint32_t groupCountZ = 1; // 0 means use input size

    uint32_t numberPaths = 1;
    uint32_t customOutputSize = 0;  // 0 means use input size

    std::vector<R> outputBuffers;

    std::conditional_t<!std::is_void<U>::value, U*, void*> ubo = nullptr;

    std::conditional_t<!std::is_void<P>::value, std::vector<P>, void*> pushConstants;

    std::vector<std::shared_ptr<BufferElementInterface>> otherInputs;
    /*

    */
    void createDescriptorPool() {
        auto& device = vulkanKernel->getDevice();
        auto& config = vulkanKernel->getConfig();

        
        std::vector<VkDescriptorPoolSize> poolSizes(3 + otherInputs.size());
        // A
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        // TODO use numberPaths instead of hardcoded 3 ...
        poolSizes[0].descriptorCount = 3; //static_cast<uint32_t>(swapChainSize);
        
        // B
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        // TODO use numberPaths instead of hardcoded 3 ...
        poolSizes[1].descriptorCount = 3; //static_cast<uint32_t>(swapChainSize);
        
        // Result
        poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        // TODO use numberPaths instead of hardcoded 3 ...
        poolSizes[2].descriptorCount = 3; //static_cast<uint32_t>(swapChainSize);

        // Other inputs
        for (size_t i = 0; i < otherInputs.size(); i++) {
            poolSizes[3 + i].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            poolSizes[3 + i].descriptorCount = 3; //static_cast<uint32_t>(swapChainSize);
        }
        
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = (uint32_t)poolSizes.size();
        poolInfo.pPoolSizes = poolSizes.data();
        // TODO use numberPaths instead of hardcoded 3 ... or whatever is needed
        poolInfo.maxSets = 3;
    
        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor pool!");
        }
    }    

    void createComputeDescriptorSetLayout()
    {
        auto& device = vulkanKernel->getDevice();

        std::vector<VkDescriptorSetLayoutBinding> layoutBindings(3 + otherInputs.size());
    
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

        // Other inputs
        for (size_t i = 0; i < otherInputs.size(); i++) {
            layoutBindings[3 + i].binding = 3 + i;
            layoutBindings[3 + i].descriptorCount = 1;
            layoutBindings[3 + i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            layoutBindings[3 + i].pImmutableSamplers = nullptr;
            layoutBindings[3 + i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = (uint32_t)layoutBindings.size();
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
        R& r = outputBuffers[pathId];
    
        //VkDescriptorSetLayout layout;
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &computeDescriptorSetLayout;
    
        VkResult result = vkAllocateDescriptorSets(device, &allocInfo, &computeDescriptorSets[pathId]);
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

        std::vector<VkWriteDescriptorSet> descriptorWrites{2+otherInputs.size()};

        // A
        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = computeDescriptorSets[pathId];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &storageBufferInfoA;

        // R
        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = computeDescriptorSets[pathId];
        descriptorWrites[1].dstBinding = 2;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pBufferInfo = &storageBufferInfoResult;

        // Other inputs
        std::vector<VkDescriptorBufferInfo> storageBufferInfoOthers(otherInputs.size());
        for (size_t i = 0; i < otherInputs.size(); i++) {
            auto& otherInput = otherInputs[i];
            VkDescriptorBufferInfo& storageBufferInfoOther = storageBufferInfoOthers[i];
            storageBufferInfoOther.buffer = otherInput->getVkBuffer();
            storageBufferInfoOther.offset = 0;
            storageBufferInfoOther.range = otherInput->getBufferMemSize();

            descriptorWrites[2 + i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[2 + i].dstSet = computeDescriptorSets[pathId];
            descriptorWrites[2 + i].dstBinding = 3 + i;
            descriptorWrites[2 + i].dstArrayElement = 0;
            descriptorWrites[2 + i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[2 + i].descriptorCount = 1;
            descriptorWrites[2 + i].pBufferInfo = &storageBufferInfoOther;
        }

        vkUpdateDescriptorSets(device, (uint32_t)descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
    }

    void createComputePipeline()
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
        
        if constexpr (!std::is_void<U>::value) {
            pipelineLayoutInfo.setLayoutCount = 2;
            VkDescriptorSetLayout combinedLayouts[] = {computeDescriptorSetLayout, ubo->getDescriptorSetLayout()};
            pipelineLayoutInfo.pSetLayouts = combinedLayouts;
        } else {
            pipelineLayoutInfo.setLayoutCount = 1;
            VkDescriptorSetLayout combinedLayouts[] = {computeDescriptorSetLayout};
            pipelineLayoutInfo.pSetLayouts = combinedLayouts;
        }

        if constexpr (!std::is_void<P>::value) {
            VkPushConstantRange pushConstantRange{};
            pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pushConstantRange.offset = 0;
            pushConstantRange.size = sizeof(P);
            pipelineLayoutInfo.pushConstantRangeCount = 1;
            pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
        }
        
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