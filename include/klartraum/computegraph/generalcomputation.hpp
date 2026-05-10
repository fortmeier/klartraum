#ifndef KLARTRAUM_GENERALCOMPUTATION_HPP
#define KLARTRAUM_GENERALCOMPUTATION_HPP

#include <array>
#include <string>
#include <vector>

#include "klartraum/computegraph/computegraphelement.hpp"
#include "klartraum/vulkan_helpers.hpp"
#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/computegraph/tensorelement.hpp"

namespace klartraum {

using DispatchIndirectCommandBufferElement = BufferElement<VulkanBuffer<VkDispatchIndirectCommand>>;

template <typename P = void>
class GeneralComputation : public ComputeGraphElement {
public:
    GeneralComputation(VulkanContext &vulkanContext, const std::string &shaderPath) :
        shaderPaths({shaderPath}) 
    {
        this->vulkanContext = &vulkanContext;
    }

    GeneralComputation(VulkanContext &vulkanContext, const std::vector<std::string> &shaderPaths) :
        shaderPaths(shaderPaths) 
    {
        this->vulkanContext = &vulkanContext;
    }

    virtual ~GeneralComputation() {
        if(initialized) {
            vkDestroyPipelineLayout(vulkanContext->getDevice(), computePipelineLayout, nullptr);
            for(auto& computePipeline : computePipelines) {
                vkDestroyPipeline(vulkanContext->getDevice(), computePipeline, nullptr);
            }
            vkDestroyDescriptorSetLayout(vulkanContext->getDevice(), computeDescriptorSetLayout, nullptr);
            vkDestroyDescriptorPool(vulkanContext->getDevice(), descriptorPool, nullptr);
        }
    }

    virtual void _setup(VulkanContext& vulkanContext, uint32_t numberPaths) {
        this->numberPaths = numberPaths;
        this->vulkanContext = &vulkanContext;
        // TODO CHECK, suggetest by copilot and in BT, but I think it is not necessary
        // ADDEDUM: we setup them, since other is missleading, it should be named scratchBuffers
        // these are not part of the compute graph 
        for (auto& other : otherInputs) {
            other->_setup(vulkanContext, numberPaths);
        }

        createDescriptorPool();
        createComputeDescriptorSetLayout();
        createComputePipeline();
        
        uint32_t inputSize = (uint32_t)inputs.size();
        if (inputSize == 0 && otherInputs.empty()) {
            throw std::runtime_error("no inputs provided!");
        }

        computeDescriptorSets.resize(numberPaths);
        for(uint32_t i = 0; i < numberPaths; i++) {
            createComputeDescriptorSets(i);
        }

        initialized = true;
    }

    void setGroupCount(uint32_t countX, uint32_t countY, uint32_t countZ) {
        groupCountX = countX;
        groupCountY = countY;
        groupCountZ = countZ;
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

    void setPushConstants(const std::vector<P>& pushConstants) {
        if constexpr (!std::is_void<P>::value) {
            this->pushConstants = pushConstants;
        }
    }

    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) {
        if (!initialized) {
            throw std::runtime_error("GeneralComputation not initialized");
        }

        // Add memory barriers for all ImageViewSrc inputs
        for (int i = 0; i < inputs.size(); i++) {
            ComputeGraphElementPtr input = getInputElement(i);
            ImageViewSrc* imageElement = dynamic_cast<ImageViewSrc*>(input.get());
            if (imageElement) {
                VkImageMemoryBarrier imageBarrier{};
                imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                imageBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
                imageBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                imageBarrier.image = imageElement->getImage(pathId);
                imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                imageBarrier.subresourceRange.baseMipLevel = 0;
                imageBarrier.subresourceRange.levelCount = 1;
                imageBarrier.subresourceRange.baseArrayLayer = 0;
                imageBarrier.subresourceRange.layerCount = 1;
                imageBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                imageBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

                vkCmdPipelineBarrier(
                    commandBuffer,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    0, nullptr,
                    0, nullptr,
                    1, &imageBarrier
                );
            }
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
                    dispatch(commandBuffer, pathId, computePipeline, &pushConstant);
                }
            }
        }
    }

    virtual void checkInput(ComputeGraphElementPtr input, int index = 0) {
        BufferElementInterface* bufferSrc = std::dynamic_pointer_cast<BufferElementInterface>(input).get();
        ImageViewSrc* imageSrc = std::dynamic_pointer_cast<ImageViewSrc>(input).get();
        UniformBufferObjectInterface* uboSrc = std::dynamic_pointer_cast<UniformBufferObjectInterface>(input).get();
        TensorElementInterface* tensorSrc = std::dynamic_pointer_cast<TensorElementInterface>(input).get();

        if (bufferSrc == nullptr && imageSrc == nullptr && uboSrc == nullptr && tensorSrc == nullptr) {
            throw std::runtime_error("input is not a BufferElementInterface or ImageViewSrc or UniformBufferObject or TensorElementInterface!");
        }
    }

    // BufferElementInterface& getInput(uint32_t pathId = 0) {
    //     return *dynamic_cast<BufferElementInterface*>(inputs[0].get());
    // }

    virtual const char* getType() const {
        return "GeneralComputation";
    }    

    void addScratchBufferElement(std::shared_ptr<BufferElementInterface> bufferElement, bool recordSetToZero = true) {
        otherInputs.push_back(bufferElement);
        otherInputsSetToZero.push_back(recordSetToZero);
    }

private:
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> computeDescriptorSets;
    VkDescriptorSetLayout computeDescriptorSetLayout;
    VkPipelineLayout computePipelineLayout;
    std::vector<VkPipeline> computePipelines;

    VulkanContext* vulkanContext;

    // inputs that are also outputs of the shader
    std::map<int, ComputeGraphElementPtr> inputOutputs;

    bool setToZero = false; // whether to set the scratch buffers to zero before dispatching


    const std::vector<std::string> shaderPaths;

    uint32_t groupCountX = 1;
    uint32_t groupCountY = 1;
    uint32_t groupCountZ = 1;

    uint32_t numberPaths = 1;

    std::conditional_t<!std::is_void<P>::value, std::vector<P>, void*> pushConstants;
    std::shared_ptr<DispatchIndirectCommandBufferElement> dynamicGroupDispatchParams;

    std::vector<std::shared_ptr<BufferElementInterface>> otherInputs;
    std::vector<bool> otherInputsSetToZero; // whether to record the scratch buffers to zero

    static VkDescriptorType getDescriptorType(const ComputeGraphElementPtr& input) {
        if (dynamic_cast<ImageViewSrc*>(input.get())) {
            return VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        } else if (dynamic_cast<BufferElementInterface*>(input.get())) {
            return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        } else if (dynamic_cast<UniformBufferObjectInterface*>(input.get())) {
            return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        }
        return VK_DESCRIPTOR_TYPE_MAX_ENUM; // Invalid type
    }

    void createDescriptorPool() {
        auto& device = vulkanContext->getDevice();
        
        std::vector<VkDescriptorPoolSize> poolSizes(inputs.size() + otherInputs.size());
        
        // Regular inputs
        for (size_t i = 0; i < inputs.size(); i++) {
            ComputeGraphElementPtr inputPtr = getInputElement(i);
            poolSizes[i].type = getDescriptorType(inputPtr);
            poolSizes[i].descriptorCount = numberPaths;
        }
        
        // Other inputs
        for (size_t i = 0; i < otherInputs.size(); i++) {
            poolSizes[inputs.size() + i].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            poolSizes[inputs.size() + i].descriptorCount = numberPaths;
        }
        
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = (uint32_t)poolSizes.size();
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = numberPaths;
    
        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor pool!");
        }
    }    

    void createComputeDescriptorSetLayout() {
        auto& device = vulkanContext->getDevice();

        std::vector<VkDescriptorSetLayoutBinding> layoutBindings(inputs.size() + otherInputs.size());
    
        // Regular inputs
        for (int i = 0; i < inputs.size(); i++) {
            ComputeGraphElementPtr inputPtr = getInputElement(i);

            VkDescriptorType descriptorType = getDescriptorType(inputPtr);

            if (descriptorType == VK_DESCRIPTOR_TYPE_MAX_ENUM) {
                throw std::runtime_error("Input type not supported for descriptor set layout");
            }

            layoutBindings[i].binding = i;
            layoutBindings[i].descriptorCount = 1;
            layoutBindings[i].descriptorType = descriptorType;
            layoutBindings[i].pImmutableSamplers = nullptr;
            layoutBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        
        // Other inputs
        for (int i = 0; i < otherInputs.size(); i++) {
            layoutBindings[inputs.size() + i].binding = inputs.size() + i;
            layoutBindings[inputs.size() + i].descriptorCount = 1;
            layoutBindings[inputs.size() + i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            layoutBindings[inputs.size() + i].pImmutableSamplers = nullptr;
            layoutBindings[inputs.size() + i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = (uint32_t)layoutBindings.size();
        layoutInfo.pBindings = layoutBindings.data();
        
        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &computeDescriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute descriptor set layout!");
        }
    }

    void createComputeDescriptorSets(uint32_t pathId) {
        auto& device = vulkanContext->getDevice();

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &computeDescriptorSetLayout;
    
        if (vkAllocateDescriptorSets(device, &allocInfo, &computeDescriptorSets[pathId]) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

        // creates memory for the descriptor sets 
        // that is alive until the method goes out of scope
        // (past the vkUpdateDescriptorSets call)
        std::vector<VkDescriptorImageInfo> imageInfos;
        imageInfos.reserve(inputs.size());
        std::vector<VkDescriptorBufferInfo> bufferInfos;
        bufferInfos.reserve(inputs.size());

        std::vector<VkWriteDescriptorSet> descriptorWrites{inputs.size() + otherInputs.size()};
        
        // Regular inputs
        for(int i = 0; i < inputs.size(); i++) {
            ComputeGraphElementPtr inputPtr = getInputElement(i);

            // Cast input to BufferElement to access underlying buffer
            BufferElementInterface* bufferElement = dynamic_cast<BufferElementInterface*>(inputPtr.get());
            ImageViewSrc* imageElement = dynamic_cast<ImageViewSrc*>(inputPtr.get());
            UniformBufferObjectInterface* uboElement = dynamic_cast<UniformBufferObjectInterface*>(inputPtr.get());

            
            if (imageElement) {
                VkDescriptorImageInfo& imageInfo = imageInfos.emplace_back();
                
                imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                imageInfo.imageView = imageElement->getImageView(pathId);
                // imageInfo.sampler = imageElement->getSampler();
                
                descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrites[i].dstSet = computeDescriptorSets[pathId];
                descriptorWrites[i].dstBinding = i;
                descriptorWrites[i].dstArrayElement = 0;
                descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                descriptorWrites[i].descriptorCount = 1;
                descriptorWrites[i].pImageInfo = &imageInfo;
            } else if (bufferElement) {
                bufferInfos.emplace_back();
                VkDescriptorBufferInfo& inputBufferInfo = bufferInfos.back();

                inputBufferInfo.buffer = bufferElement->getVkBuffer(pathId);
                inputBufferInfo.offset = 0;
                inputBufferInfo.range = bufferElement->getBufferMemSize();

                descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrites[i].dstSet = computeDescriptorSets[pathId];
                descriptorWrites[i].dstBinding = i;
                descriptorWrites[i].dstArrayElement = 0;
                descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                descriptorWrites[i].descriptorCount = 1;
                descriptorWrites[i].pBufferInfo = &inputBufferInfo;
            } else if (uboElement) {
                bufferInfos.emplace_back();
                VkDescriptorBufferInfo& inputBufferInfo = bufferInfos.back();

                inputBufferInfo.buffer = uboElement->getVkBuffer(pathId);
                inputBufferInfo.offset = 0;
                inputBufferInfo.range = uboElement->getBufferMemSize();

                descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                descriptorWrites[i].dstSet = computeDescriptorSets[pathId];
                descriptorWrites[i].dstBinding = i;
                descriptorWrites[i].dstArrayElement = 0;
                descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                descriptorWrites[i].descriptorCount = 1;
                descriptorWrites[i].pBufferInfo = &inputBufferInfo;
            }
            else {
                throw std::runtime_error("not implemented yet");
            }
        }
        
        // Other inputs
        std::vector<VkDescriptorBufferInfo> storageBufferInfoOthers(otherInputs.size());
        for (int i = 0; i < otherInputs.size(); i++) {
            auto& otherInput = otherInputs[i];
            VkDescriptorBufferInfo& storageBufferInfoOther = storageBufferInfoOthers[i];
            storageBufferInfoOther.buffer = otherInput->getVkBuffer(pathId);
            storageBufferInfoOther.offset = 0;
            storageBufferInfoOther.range = otherInput->getBufferMemSize();

            descriptorWrites[inputs.size() + i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[inputs.size() + i].dstSet = computeDescriptorSets[pathId];
            descriptorWrites[inputs.size() + i].dstBinding = inputs.size() + i;
            descriptorWrites[inputs.size() + i].dstArrayElement = 0;
            descriptorWrites[inputs.size() + i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[inputs.size() + i].descriptorCount = 1;
            descriptorWrites[inputs.size() + i].pBufferInfo = &storageBufferInfoOther;
        }

        vkUpdateDescriptorSets(device, (uint32_t)descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
    }

    void createComputePipeline() {
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
    
        // pushConstantRange must outlive vkCreatePipelineLayout — declare outside the
        // if constexpr block to avoid a dangling pointer in pPushConstantRanges.
        VkPushConstantRange pushConstantRange{};
        if constexpr (!std::is_void<P>::value) {
            pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            pushConstantRange.offset = 0;
            pushConstantRange.size = sizeof(P);
        }

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &computeDescriptorSetLayout;
        if constexpr (!std::is_void<P>::value) {
            pipelineLayoutInfo.pushConstantRangeCount = 1;
            pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
        }

        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &computePipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute pipeline layout!");
        }

        std::vector<VkComputePipelineCreateInfo> pipelineInfos(computeShaderStages.size());
        for (size_t i = 0; i < computeShaderStages.size(); i++) {
            VkComputePipelineCreateInfo& pipelineInfo = pipelineInfos[i];
            pipelineInfo.sType              = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            pipelineInfo.layout             = computePipelineLayout;
            pipelineInfo.stage              = computeShaderStages[i];
            pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
            pipelineInfo.basePipelineIndex  = -1;
        }

        computePipelines.resize(computeShaderStages.size());
        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, (uint32_t)computeShaderStages.size(), pipelineInfos.data(), nullptr, computePipelines.data()) != VK_SUCCESS) {
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

    void bind(VkCommandBuffer commandBuffer, uint32_t pathId, VkPipeline computePipeline) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &computeDescriptorSets[pathId], 0, 0);
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

    template<typename PushConstantType = P>
    typename std::enable_if<!std::is_void<PushConstantType>::value, void>::type
    dispatch(VkCommandBuffer commandBuffer, uint32_t pathId, VkPipeline computePipeline, const PushConstantType* pushConstant) {
        vkCmdPushConstants(commandBuffer, computePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstantType), pushConstant);
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
};

} // namespace klartraum
#endif // KLARTRAUM_GENERALCOMPUTATION_HPP 