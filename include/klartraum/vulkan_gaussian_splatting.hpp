#ifndef VULKAN_GAUSSIAN_SPLATTING_HPP
#define VULKAN_GAUSSIAN_SPLATTING_HPP

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vector>
#include <string>

#include "klartraum/glfw_frontend.hpp"

namespace klartraum {

enum class GaussianSplattingRenderingType {
    PointCloud,
    // GaussianSplatting, not implemented yet
};    

class VulkanGaussianSplatting : public DrawComponent {
public:
    VulkanGaussianSplatting(std::string path, GaussianSplattingRenderingType type=GaussianSplattingRenderingType::PointCloud);
    ~VulkanGaussianSplatting();

    virtual void draw(uint32_t currentFrame, VkCommandBuffer& commandBuffer, VkFramebuffer& framebuffer, VkSemaphore& imageAvailableSemaphore, uint32_t imageIndex) override;

    virtual void initialize(VulkanKernel& vulkanKernel) override;


private:
    void createComputeDescriptorSetLayout();
    void createDescriptorPool();
    void createComputeDescriptorSets();
    void createComputePipeline();
    void createGraphicsPipeline();
    void createSyncObjects();

    void createVertexBuffer();

    void recordCommandBuffer(uint32_t currentFrame, VkCommandBuffer commandBuffer, VkFramebuffer framebuffer, uint32_t imageIndex);

    void loadSPZModel(std::string path);

    VulkanKernel* vulkanKernel;

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
 
    
};

} // namespace klartraum

#endif // VULKAN_GAUSSIAN_SPLATTING_HPP