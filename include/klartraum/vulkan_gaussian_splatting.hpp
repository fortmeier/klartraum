#ifndef VULKAN_GAUSSIAN_SPLATTING_HPP
#define VULKAN_GAUSSIAN_SPLATTING_HPP

#include <vector>
#include <string>

#include "klartraum/drawgraph/drawgraphelement.hpp"

#include "klartraum/vulkan_buffer.hpp"
#include "klartraum/vulkan_gaussian_splatting_types.hpp"

namespace klartraum {


enum class GaussianSplattingRenderingType {
    PointCloud,
    // GaussianSplatting, not implemented yet
};    

class VulkanGaussianSplatting : public DrawGraphElement {
public:
    VulkanGaussianSplatting(std::string path, GaussianSplattingRenderingType type=GaussianSplattingRenderingType::PointCloud);
    ~VulkanGaussianSplatting();

    virtual void _setup(VulkanKernel& vulkanKernel, uint32_t numberPaths);

    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId);

    virtual const char* getName() const
    {
        return "GaussianSplatting";
    }
    
    virtual void initialize(VulkanKernel& vulkanKernel, VkRenderPass& renderpass);
    
    virtual void recordCommandBuffer(VkCommandBuffer commandBuffer, VkFramebuffer framebuffer, uint32_t pathId);

private:
    void createComputeDescriptorSetLayout();
    void createDescriptorPool();
    void createComputeDescriptorSets();
    void createComputePipeline();
    void createSyncObjects();

    void createVertexBuffer();

    void loadSPZModel(std::string path);

    VulkanKernel* vulkanKernel = nullptr;

    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;

    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> computeDescriptorSets;
    VkDescriptorSetLayout computeDescriptorSetLayout;
    VkPipelineLayout computePipelineLayout;
    VkPipeline computePipeline;

    std::vector<uint8_t> data;
    size_t number_of_gaussians;

    std::unique_ptr<VulkanBuffer<Gaussian2D>> projectedGaussians;

    uint32_t numberOfPaths = 0;
    
};

} // namespace klartraum

#endif // VULKAN_GAUSSIAN_SPLATTING_HPP