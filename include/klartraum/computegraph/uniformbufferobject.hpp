#ifndef KLARTRAUM_UNIFORMBUFFEROBJECT_HPP
#define KLARTRAUM_UNIFORMBUFFEROBJECT_HPP

#include <vector>

#include <vulkan/vulkan.h>

#include "klartraum/computegraph/computegraphelement.hpp"

namespace klartraum {

class VulkanContext;

class UniformBufferObjectInterface : public ComputeGraphElement {
public:
    virtual size_t getBufferMemSize() const = 0;

    virtual VkBuffer& getVkBuffer(uint32_t pathId) = 0;
};

template<typename UniformBufferObjectType>
class UniformBufferObject : public UniformBufferObjectInterface {
public:
    virtual void _setup(VulkanContext& vulkanContext, uint32_t numberPaths)
    {
        this->vulkanContext = &vulkanContext;
        numberOfPaths = numberPaths;
        
        createDescriptorSetLayout();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        initialized = true;
    }

    virtual ~UniformBufferObject()
    {
        if(initialized)
        {
            auto config = vulkanContext->getConfig();
            auto device = vulkanContext->getDevice();
        
            for (size_t i = 0; i < numberOfPaths; i++) {
                vkDestroyBuffer(device, uniformBuffers[i], nullptr);
                vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
            }
        
            vkDestroyDescriptorPool(device, descriptorPool, nullptr);
            vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        }
    }

    virtual const char* getType() const {
        return "UniformBufferObject";
    }    

    VkDescriptorSetLayout& getDescriptorSetLayout()
    {
        return descriptorSetLayout;
    }

    std::vector<VkDescriptorSet>& getDescriptorSets()
    {
        return descriptorSets;
    }

    void update(uint32_t pathId)
    {
        memcpy(uniformBuffersMapped[pathId], &ubo, sizeof(ubo));
    }

    UniformBufferObjectType ubo;

    virtual size_t getBufferMemSize() const override
    {
        return sizeof(UniformBufferObjectType);
    }

    virtual VkBuffer& getVkBuffer(uint32_t pathId) override
    {
        return uniformBuffers[pathId];
    }

private:
    VulkanContext* vulkanContext = nullptr;

    void createDescriptorSetLayout()
    {
        auto device = vulkanContext->getDevice();

        VkDescriptorSetLayoutBinding uboLayoutBinding{};
        uboLayoutBinding.binding = 0;
        uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboLayoutBinding.descriptorCount = 1;
    
        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
    
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = 1;
        layoutInfo.pBindings = &uboLayoutBinding;
    
        if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor set layout!");
        }
    }

    void createUniformBuffers()
    {
        auto config = vulkanContext->getConfig();
        auto device = vulkanContext->getDevice();
    
        VkDeviceSize bufferSize = sizeof(UniformBufferObjectType);
    
        uniformBuffers.resize(numberOfPaths);
        uniformBuffersMemory.resize(numberOfPaths);
        uniformBuffersMapped.resize(numberOfPaths);
    
        for (size_t i = 0; i < numberOfPaths; i++) {
            vulkanContext->createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);
    
            vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
        }
    }

    void createDescriptorPool()
    {
        auto device = vulkanContext->getDevice();
        auto config = vulkanContext->getConfig();
    
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSize.descriptorCount = static_cast<uint32_t>(numberOfPaths);    
    
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
    
        poolInfo.maxSets = static_cast<uint32_t>(numberOfPaths);
    
        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor pool!");
        }
    
    }

    void createDescriptorSets()

    {
        auto device = vulkanContext->getDevice();
        auto config = vulkanContext->getConfig();
    
        std::vector<VkDescriptorSetLayout> layouts(numberOfPaths, descriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(numberOfPaths);
        allocInfo.pSetLayouts = layouts.data();
    
        descriptorSets.resize(numberOfPaths);
        if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }
    
        for (size_t i = 0; i < numberOfPaths; i++) {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObjectType);
    
            VkWriteDescriptorSet descriptorWrite{};
            descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrite.dstSet = descriptorSets[i];
            descriptorWrite.dstBinding = 0;
            descriptorWrite.dstArrayElement = 0;        
    
            descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrite.descriptorCount = 1;
    
            descriptorWrite.pBufferInfo = &bufferInfo;
            descriptorWrite.pImageInfo = nullptr; // Optional
            descriptorWrite.pTexelBufferView = nullptr; // Optional
    
            vkUpdateDescriptorSets(device, 1, &descriptorWrite, 0, nullptr);
        }
    }

    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;

    uint32_t numberOfPaths = 0;

};

} // namespace klartraum

#endif // KLARTRAUM_UNIFORMBUFFEROBJECT_HPP