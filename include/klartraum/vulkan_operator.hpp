#ifndef KLARTRAUM_VULKAN_OPERATION_HPP
#define KLARTRAUM_VULKAN_OPERATION_HPP

#include <vulkan/vulkan.h>

#include <array>

#include "klartraum/vulkan_helpers.hpp"

namespace klartraum {

template <typename A, typename B, typename R>
class VulkanOperator {
public:
    VulkanOperator(VulkanKernel &vulkanKernel, const std::string &shaderPath)
    {
        this->vulkanKernel = &vulkanKernel;
        // createDescriptorPool();
        createComputeDescriptorSetLayout();
        createComputeDescriptorSets();
        createComputePipeline(shaderPath);
    
        //createSyncObjects();
    }

    virtual ~VulkanOperator() {};

    void operator()(const A& a, const B& b, R& result)
    {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);

        std::array<VkDescriptorSet, 2> combinedDescriptorSets = {computeDescriptorSets[imageIndex], descriptorSets[currentFrame]};
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 2, combinedDescriptorSets.data(), 0, 0);
        
        uint32_t num_groups_z = number_of_gaussians / 16;

        vkCmdDispatch(commandBuffer, 64, 64, num_groups_z);
        return;
    }

private:
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> computeDescriptorSets;
    VkDescriptorSetLayout computeDescriptorSetLayout;
    VkPipelineLayout computePipelineLayout;
    VkPipeline computePipeline;

    VulkanKernel* vulkanKernel;

    void createComputeDescriptorSetLayout()
    {
        auto& device = vulkanKernel->getDevice();

        std::array<VkDescriptorSetLayoutBinding, 2> layoutBindings{};
    
        // gaussians binding
        layoutBindings[0].binding = 0;
        layoutBindings[0].descriptorCount = 1;
        layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[0].pImmutableSamplers = nullptr;
        layoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        // output frame binding
        layoutBindings[1].binding = 1;
        layoutBindings[1].descriptorCount = 1;
        layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layoutBindings[1].pImmutableSamplers = nullptr;
        layoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = layoutBindings.size();
        layoutInfo.pBindings = layoutBindings.data();
        
        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &computeDescriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create compute descriptor set layout!");
        }
    }

    void createComputeDescriptorSets()
    {
        /*
        auto& device = vulkanKernel->getDevice();
        auto& config = vulkanKernel->getConfig();
    
        uint32_t swapChainSize = 3; // TODO
    
        std::vector<VkDescriptorSetLayout> layouts(swapChainSize, computeDescriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(swapChainSize);
        allocInfo.pSetLayouts = layouts.data();
    
        computeDescriptorSets.resize(swapChainSize);
        VkResult result = vkAllocateDescriptorSets(device, &allocInfo, computeDescriptorSets.data());
        if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }
    
        for (size_t i = 0; i < swapChainSize; i++) {
            VkDescriptorBufferInfo storageBufferInfo{};
            storageBufferInfo.buffer = vertexBuffer;
            storageBufferInfo.offset = 0;
            storageBufferInfo.range = sizeof(Gaussian) * number_of_gaussians;
    
    
            std::array<VkWriteDescriptorSet, 2> descriptorWrites{};
            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = computeDescriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &storageBufferInfo;
    
    
            VkDescriptorImageInfo imageInfo = {};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL; // TODO ????
            imageInfo.imageView = vulkanKernel->getImageView(i);
    
            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = computeDescriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pImageInfo = &imageInfo;
    
            vkUpdateDescriptorSets(device, descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
        }
        */
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
#endif // KLARTRAUM_VULKAN_OPERATION_HPP