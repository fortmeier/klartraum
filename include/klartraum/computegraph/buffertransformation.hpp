#ifndef KLARTRAUM_TRANSFORMATION_HPP
#define KLARTRAUM_TRANSFORMATION_HPP

#include <vulkan/vulkan.h>

#include <array>

#include "klartraum/computegraph/computegraphelement.hpp"
#include "klartraum/vulkan_helpers.hpp"
#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/vulkan_buffer.hpp"

namespace klartraum {

using DispatchIndirectCommandBufferElement = BufferElement<VulkanBuffer<VkDispatchIndirectCommand>>;


template <typename A, typename R, typename U = void, typename P = void>
class BufferTransformation : public TemplatedBufferElementInterface<R> {
public:
    BufferTransformation(VulkanContext &vulkanContext, const std::string &shaderPath) :
        shaderPaths({shaderPath}) 
    {
        this->vulkanContext = &vulkanContext;
        
        if constexpr (!std::is_void<U>::value) {
            uboPtr = std::make_shared<U>();
        }



    }

    BufferTransformation(VulkanContext &vulkanContext, const std::vector<std::string> &shaderPaths) :
        shaderPaths(shaderPaths) 
    {
        this->vulkanContext = &vulkanContext;
        
        if constexpr (!std::is_void<U>::value) {
            uboPtr = std::make_shared<U>();
        }



    }

    virtual ~BufferTransformation() {
        if (this->initialized) {
            vkDestroyPipelineLayout(vulkanContext->getDevice(), computePipelineLayout, nullptr);
            for(auto& computePipeline : computePipelines) {
                vkDestroyPipeline(vulkanContext->getDevice(), computePipeline, nullptr);
            }
            vkDestroyDescriptorSetLayout(vulkanContext->getDevice(), computeDescriptorSetLayout, nullptr);
            vkDestroyDescriptorPool(vulkanContext->getDevice(), descriptorPool, nullptr);

            if constexpr (!std::is_void<U>::value) {
                uboPtr.reset();
            }
        }
    };

    void setPushConstants(const std::vector<P>& pushConstants) {
        if constexpr (!std::is_void<P>::value) {
            this->pushConstants = pushConstants;
        }
    }

    virtual void _setup(VulkanContext& vulkanContext, uint32_t numberPaths) {
        this->numberPaths = numberPaths;
        this->vulkanContext = &vulkanContext;

        if constexpr (!std::is_void<U>::value) {
            uboPtr->_setup(vulkanContext, numberPaths);
        }

        for (auto& other : otherInputs) {
            other->_setup(vulkanContext, numberPaths);
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
            outputBuffers.emplace_back(vulkanContext, outputSize);
            createComputeDescriptorSets(i);
            
            if constexpr (!std::is_void<U>::value) {
                uboPtr->update(i);
            }
        }
        this->initialized = true;

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

    void setDynamicGroupDispatchParams(const std::shared_ptr<DispatchIndirectCommandBufferElement>& params) {
        this->dynamicGroupDispatchParams = params;
    }

    void bind(VkCommandBuffer commandBuffer, uint32_t pathId, VkPipeline computePipeline) {
        if constexpr (std::is_void<U>::value) {
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSets[pathId], 0, 0);
        } else {
            std::vector<VkDescriptorSet> descriptorSets = {computeDescriptorSets[pathId], uboPtr->getDescriptorSets()[pathId]};
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
            vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, (uint32_t)descriptorSets.size(), descriptorSets.data(), 0, 0);
        }
    }

    void dispatch(VkCommandBuffer commandBuffer, uint32_t pathId, VkPipeline computePipeline) {
        if (dynamicGroupDispatchParams == nullptr)
        {
            vkCmdDispatch(commandBuffer, groupCountX, groupCountY, groupCountZ);
        }
        else
        {
            vkCmdDispatchIndirect(commandBuffer, dynamicGroupDispatchParams->getVkBuffer(pathId), 0);
        }
    }

    void dispatch(VkCommandBuffer commandBuffer, uint32_t pathId, VkPipeline computePipeline, P pushConstant) {
        vkCmdPushConstants(commandBuffer, computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(P), &pushConstant);
        if (dynamicGroupDispatchParams == nullptr)
        {
            vkCmdDispatch(commandBuffer, groupCountX, groupCountY, groupCountZ);
        }
        else
        {
            vkCmdDispatchIndirect(commandBuffer, dynamicGroupDispatchParams->getVkBuffer(0), 0);
        }
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

    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) {
        if (!this->initialized) {
            throw std::runtime_error("BufferTransformation not initialized");
        }

        
        if constexpr (std::is_void<P>::value) {
            recordScratchToZero(commandBuffer, pathId);
            for(VkPipeline computePipeline : computePipelines) {
                bind(commandBuffer, pathId, computePipeline);
                dispatch(commandBuffer, pathId, computePipeline);
            }
        } else {
            if (pushConstants.empty()) {
                throw std::runtime_error("push constants are empty!");
            }
            for (const auto& pushConstant : pushConstants) {
                recordScratchToZero(commandBuffer, pathId);
                for(VkPipeline computePipeline : computePipelines) {
                    bind(commandBuffer, pathId, computePipeline);
                    dispatch(commandBuffer, pathId, computePipeline, pushConstant);
                }
            }
        }
    };

    virtual void checkInput(ComputeGraphElementPtr input, int index = 0) {
        BufferElementInterface* bufferSrc = std::dynamic_pointer_cast<BufferElementInterface>(input).get();
        if (bufferSrc == nullptr) {
            throw std::runtime_error("input is not a fitting BufferElementInterface!");
        }
    }

    A& getInput(uint32_t pathId = 0) {
        // TODO why inputs[0], and not inputs[pathId]?
        auto bufferPtr = dynamic_cast<TemplatedBufferElementInterface<A>*>(this->getInputElement(0).get());
        if (bufferPtr == nullptr) {
            throw std::runtime_error("input is not a fitting BufferElement!");
        }
        
        return bufferPtr->getBuffer(pathId);
    }

    virtual const char* getType() const {
        return "BufferTransformation";
    }    

    R& getOutputBuffer(uint32_t pathId = 0) {
        return outputBuffers[pathId];
    }

    std::conditional_t<!std::is_void<U>::value, std::shared_ptr<U>, void*> getUbo() {
        if constexpr (!std::is_void<U>::value) {
            return uboPtr;
        } else {
            return nullptr;
        }
    }

    void setUbo(std::conditional_t<!std::is_void<U>::value, std::shared_ptr<U>, void*> ubo) {
        if constexpr (!std::is_void<U>::value) {
            uboPtr = ubo;
        }
    }

    void addScratchBufferElement(std::shared_ptr<BufferElementInterface> bufferElement, bool recordSetToZero = true) {
        otherInputs.push_back(bufferElement);
        otherInputsSetToZero.push_back(recordSetToZero);
    }


    virtual size_t getBufferMemSize() const override {
        return outputBuffers[0].getBufferMemSize();
    }

    virtual VkBuffer& getVkBuffer(uint32_t pathId) override {
        return outputBuffers[pathId].getBuffer();
    }


    virtual R& getBuffer(uint32_t pathId) override {
        return outputBuffers[pathId];
    }

private:
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> computeDescriptorSets;
    VkDescriptorSetLayout computeDescriptorSetLayout;
    VkPipelineLayout computePipelineLayout;
    std::vector<VkPipeline> computePipelines;

    VulkanContext* vulkanContext;

    const std::vector<std::string> shaderPaths;

    uint32_t groupCountX = 0;
    uint32_t groupCountY = 1;
    uint32_t groupCountZ = 1; // 0 means use input size

    uint32_t numberPaths = 1;
    uint32_t customOutputSize = 0;  // 0 means use input size

    std::vector<R> outputBuffers;

    std::conditional_t<!std::is_void<U>::value, std::shared_ptr<U>, void*> uboPtr = nullptr;

    std::conditional_t<!std::is_void<P>::value, std::vector<P>, void*> pushConstants;

    std::shared_ptr<DispatchIndirectCommandBufferElement> dynamicGroupDispatchParams;


    std::vector<std::shared_ptr<BufferElementInterface>> otherInputs;
    std::vector<bool> otherInputsSetToZero; // whether to record the scratch buffers to zero
    /*

    */
    void createDescriptorPool() {
        auto& device = vulkanContext->getDevice();
        auto& config = vulkanContext->getConfig();

        
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
        auto& device = vulkanContext->getDevice();

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
        for (int i = 0; i < otherInputs.size(); i++) {
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
        auto& device = vulkanContext->getDevice();
        auto& config = vulkanContext->getConfig();

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
        for (int i = 0; i < otherInputs.size(); i++) {
            auto& otherInput = otherInputs[i];
            VkDescriptorBufferInfo& storageBufferInfoOther = storageBufferInfoOthers[i];
            storageBufferInfoOther.buffer = otherInput->getVkBuffer(pathId);
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
        auto& device = vulkanContext->getDevice();

        std::vector<VkShaderModule> computeShaderModules(shaderPaths.size());
        std::vector<VkPipelineShaderStageCreateInfo> computeShaderStages(shaderPaths.size()); 

        for (size_t i = 0; i < shaderPaths.size(); i++) {
            auto computeShaderCode = readFile(shaderPaths[i]);

            computeShaderModules[i] = createShaderModule(computeShaderCode, vulkanContext->getDevice());

            computeShaderStages[i].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            computeShaderStages[i].stage = VK_SHADER_STAGE_COMPUTE_BIT;
            computeShaderStages[i].module = computeShaderModules[i];
            computeShaderStages[i].pName = "main";
        }

    
        
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        
        if constexpr (!std::is_void<U>::value) {
            pipelineLayoutInfo.setLayoutCount = 2;
            VkDescriptorSetLayout combinedLayouts[] = {computeDescriptorSetLayout, uboPtr->getDescriptorSetLayout()};
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
    
    
        std::vector<VkComputePipelineCreateInfo> pipelineInfos(computeShaderStages.size());
        for (size_t i = 0; i < computeShaderStages.size(); i++) {
            VkComputePipelineCreateInfo& pipelineInfo = pipelineInfos[i];
            pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipelineInfo.layout = computePipelineLayout;
            pipelineInfo.stage = computeShaderStages[i];
        }

        computePipelines.resize(computeShaderStages.size());
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, computeShaderStages.size(), pipelineInfos.data(), nullptr, computePipelines.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline!");
        }

        for (size_t i = 0; i < computeShaderModules.size(); i++) {
            vkDestroyShaderModule(device, computeShaderModules[i], nullptr);
        }
    }

    void recordScratchToZero(VkCommandBuffer commandBuffer, uint32_t pathId) {
        for (size_t i = 0; i < otherInputs.size(); i++) {
            auto& scratch = otherInputs[i];
            if (otherInputsSetToZero[i]) {
                VkBuffer scratchBuffer = scratch->getVkBuffer(pathId);
                size_t memsize = scratch->getBufferMemSize();
                vkCmdFillBuffer(commandBuffer, scratchBuffer, 0, memsize, 0);
            }
        }
    }
};

} // namespace klartraum
#endif // KLARTRAUM_TRANSFORMATION_HPP