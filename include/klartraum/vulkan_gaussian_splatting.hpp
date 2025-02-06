#ifndef VULKAN_GAUSSIAN_SPLATTING_HPP
#define VULKAN_GAUSSIAN_SPLATTING_HPP

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vector>
#include <string>

#include "klartraum/glfw_frontend.hpp"

namespace klartraum {

class VulkanGaussianSplatting : public DrawComponent {
public:
    VulkanGaussianSplatting(std::string path);
    ~VulkanGaussianSplatting();

    virtual void draw(uint32_t currentFrame, VkCommandBuffer& commandBuffer, VkFramebuffer& framebuffer, VkSemaphore& imageAvailableSemaphore) override;

    virtual void initialize(VulkanKernel& vulkanKernel) override;


private:
    void createGraphicsPipeline();
    void createSyncObjects();

    void createVertexBuffer();

    void recordCommandBuffer(uint32_t currentFrame, VkCommandBuffer commandBuffer, VkFramebuffer framebuffer);

    void loadSPZModel(std::string path);

    VulkanKernel* vulkanKernel;

    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;

    std::vector<uint8_t> data;
    size_t number_of_gaussians;
 
    
};

} // namespace klartraum

#endif // VULKAN_GAUSSIAN_SPLATTING_HPP