#include <glm/glm.hpp>
#include <array>
#include <stdexcept>

#include "load-spz.h"

#include "klartraum/vulkan_gaussian_splatting.hpp"
#include "klartraum/vulkan_helpers.hpp"
#include "klartraum/drawgraph/imageviewsrc.hpp"

namespace klartraum {

struct Gaussian {
    spz::UnpackedGaussian gaussian;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Gaussian);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }
    static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(spz::UnpackedGaussian, position);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(spz::UnpackedGaussian, color);        

        return attributeDescriptions;
    }
};

VulkanGaussianSplatting::VulkanGaussianSplatting(std::string path, GaussianSplattingRenderingType type) {
    loadSPZModel(path);
}

VulkanGaussianSplatting::~VulkanGaussianSplatting() {
    if(vulkanKernel != nullptr)
    {
        auto device = vulkanKernel->getDevice();

        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);
    }

}

void VulkanGaussianSplatting::checkInput(DrawGraphElementPtr input, int index) {
    ImageViewSrc* imageViewSrc = std::dynamic_pointer_cast<ImageViewSrc>(input).get();
    if (index == 0 && imageViewSrc == nullptr) {
        throw std::runtime_error("input is not an ImageViewSrc!");
    }
    CameraUboType* cameraUbo = std::dynamic_pointer_cast<CameraUboType>(input).get();
    if (index == 1 && cameraUbo == nullptr) {
        throw std::runtime_error("input is not a CameraUboType!");
    }
    if (index > 1) {
        throw std::runtime_error("input index out of range!");
    }

}

void VulkanGaussianSplatting::_setup(VulkanKernel& vulkanKernel, uint32_t numberPaths)
{
    numberOfPaths = numberPaths;
    this->vulkanKernel = &vulkanKernel;

    createDescriptorPool();
    createComputeDescriptorSetLayout();
    createVertexBuffer();
    createComputeDescriptorSets();
    createComputePipeline();

    createSyncObjects();

    projectedGaussians = std::make_unique<VulkanBuffer<Gaussian2D>>(vulkanKernel, number_of_gaussians);
}

void VulkanGaussianSplatting::_record(VkCommandBuffer commandBuffer, uint32_t pathId)
{
    auto& device = vulkanKernel->getDevice();
    auto& swapChain = vulkanKernel->getSwapChain();
    auto& graphicsQueue = vulkanKernel->getGraphicsQueue();


    auto& swapChainExtent = vulkanKernel->getSwapChainExtent();

    if(inputs.size() == 0) {
        throw std::runtime_error("no input!");
    }
    ImageViewSrc* imageViewSrc = std::dynamic_pointer_cast<ImageViewSrc>(inputs[0]).get();
    if (imageViewSrc == nullptr) {
        throw std::runtime_error("input is not an ImageViewSrc!");
    }
    VkImage image = imageViewSrc->getImage(pathId);


    // Ensure compute writes are visible to graphics
    VkImageMemoryBarrier imageBarrier = {};
    imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imageBarrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;  // Graphics writes
    imageBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT; // Compute reads/writes
    imageBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; //VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;  // Layout used by graphics rendering
    imageBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;  // Layout used by compute shader
    imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;  // Assume single queue
    imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    // TODO : use the correct image
    
    imageBarrier.image = image;  // The image used as framebuffer and compute input
    imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imageBarrier.subresourceRange.baseMipLevel = 0;
    imageBarrier.subresourceRange.levelCount = 1;
    imageBarrier.subresourceRange.baseArrayLayer = 0;
    imageBarrier.subresourceRange.layerCount = 1;
    
    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 
        0,  // No dependency flags
        0, nullptr,  // No global memory barriers
        0, nullptr,  // No buffer memory barriers
        1, &imageBarrier // Image memory barrier
    );
    
    // issue the compute pipeline for gaussian splatting
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);

    auto& cameraUBO = this->getCameraUBO();
    auto& descriptorSets = cameraUBO->getDescriptorSets();
    std::array<VkDescriptorSet, 2> combinedDescriptorSets = {computeDescriptorSets[pathId], descriptorSets[pathId]};
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 2, combinedDescriptorSets.data(), 0, 0);
    
    uint32_t num_groups_z = number_of_gaussians / 16;

    vkCmdDispatch(commandBuffer, 64, 64, num_groups_z);

    VkImageMemoryBarrier barrierBack = {};
    barrierBack.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrierBack.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrierBack.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrierBack.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrierBack.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrierBack.image = image;
    barrierBack.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrierBack.subresourceRange.baseMipLevel = 0;
    barrierBack.subresourceRange.levelCount = 1;
    barrierBack.subresourceRange.baseArrayLayer = 0;
    barrierBack.subresourceRange.layerCount = 1;
    
    barrierBack.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrierBack.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;
    
    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrierBack);
}

void VulkanGaussianSplatting::initialize(VulkanKernel &vulkanKernel, VkRenderPass& renderPass) {
    this->vulkanKernel = &vulkanKernel;


}

void VulkanGaussianSplatting::createComputeDescriptorSetLayout() {
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
    layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    ;
    layoutBindings[1].pImmutableSamplers = nullptr;
    layoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = (uint32_t)layoutBindings.size();
    layoutInfo.pBindings = layoutBindings.data();
    
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &computeDescriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create compute descriptor set layout!");
    }
}

void VulkanGaussianSplatting::createDescriptorPool() {
    auto& device = vulkanKernel->getDevice();
    auto& config = vulkanKernel->getConfig();

    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    // gaussians binding
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[0].descriptorCount = static_cast<uint32_t>(numberOfPaths);
    
    // output frame binding
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[1].descriptorCount = static_cast<uint32_t>(numberOfPaths);

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = (uint32_t)poolSizes.size();
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = static_cast<uint32_t>(numberOfPaths);

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor pool!");
    }
}

void VulkanGaussianSplatting::createComputeDescriptorSets() {
    auto& device = vulkanKernel->getDevice();
    auto& config = vulkanKernel->getConfig();

    std::vector<VkDescriptorSetLayout> layouts(numberOfPaths, computeDescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(numberOfPaths);
    allocInfo.pSetLayouts = layouts.data();

    computeDescriptorSets.resize(numberOfPaths);
    VkResult result = vkAllocateDescriptorSets(device, &allocInfo, computeDescriptorSets.data());
    if (result != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    if(inputs.size() == 0) {
        throw std::runtime_error("no input!");
    }
    ImageViewSrc* imageViewSrc = std::dynamic_pointer_cast<ImageViewSrc>(inputs[0]).get();
    if (imageViewSrc == nullptr) {
        throw std::runtime_error("input is not an ImageViewSrc!");
    }
    
    for (uint32_t i = 0; i < numberOfPaths; i++) {
        VkImageView imageView = imageViewSrc->getImageView(i);

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
        imageInfo.imageView = imageView;

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = computeDescriptorSets[i];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pImageInfo = &imageInfo;

        vkUpdateDescriptorSets(device, (uint32_t)descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
    }
}


void VulkanGaussianSplatting::createComputePipeline() {
    auto& device = vulkanKernel->getDevice();

    auto computeShaderCode = readFile("shaders/gaussian_splatting.comp.spv");

    VkShaderModule computeShaderModule = createShaderModule(computeShaderCode, vulkanKernel->getDevice());
    
    VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
    computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    computeShaderStageInfo.module = computeShaderModule;
    computeShaderStageInfo.pName = "main";


    
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 2;

    auto& cameraUBO = this->getCameraUBO();

    VkDescriptorSetLayout combinedLayouts[] = {computeDescriptorSetLayout, cameraUBO->getDescriptorSetLayout()};
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

void VulkanGaussianSplatting::createVertexBuffer() {
    auto device = vulkanKernel->getDevice();

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = sizeof(Gaussian) * number_of_gaussians;
    bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT ;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &vertexBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create vertex buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, vertexBuffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = vulkanKernel->findMemoryType(memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &vertexBufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate vertex buffer memory!");
    }

    vkBindBufferMemory(device, vertexBuffer, vertexBufferMemory, 0);

    void* gpu_data;
    vkMapMemory(device, vertexBufferMemory, 0, bufferInfo.size, 0, &gpu_data);
    memcpy(gpu_data, data.data(), (size_t) bufferInfo.size);
    vkUnmapMemory(device, vertexBufferMemory);
}



void VulkanGaussianSplatting::createSyncObjects()
{
    auto device = vulkanKernel->getDevice();


}

void VulkanGaussianSplatting::recordCommandBuffer(VkCommandBuffer commandBuffer, VkFramebuffer framebuffer, uint32_t pathId)
{


}

void VulkanGaussianSplatting::loadSPZModel(std::string path)
{
    spz::PackedGaussians packed = spz::loadSpzPacked(path);
    std::vector<spz::UnpackedGaussian> gaussians;
    //for (int i = 70000; i < 70020; /*packed.numPoints*/ i++) {
    for (int i = 0; i < packed.numPoints; i++) {
        spz::UnpackedGaussian gaussian = packed.unpack(i);
        gaussians.push_back(gaussian);
    }

    number_of_gaussians = (uint32_t)gaussians.size();
    data.resize(gaussians.size() * sizeof(spz::UnpackedGaussian));
    memcpy(data.data(), gaussians.data(), data.size());


}

} // namespace klartraum
