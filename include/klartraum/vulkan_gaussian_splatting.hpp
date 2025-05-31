#ifndef VULKAN_GAUSSIAN_SPLATTING_HPP
#define VULKAN_GAUSSIAN_SPLATTING_HPP

#include <vector>
#include <string>
#include <glm/gtc/matrix_transform.hpp>

#include "klartraum/drawgraph/rendergraphelement.hpp"

#include "klartraum/vulkan_buffer.hpp"
#include "klartraum/vulkan_gaussian_splatting_types.hpp"

#include "klartraum/drawgraph/drawgraphgroup.hpp"
#include "klartraum/drawgraph/buffertransformation.hpp"

namespace klartraum {


enum class GaussianSplattingRenderingType {
    PointCloud,
    // GaussianSplatting, not implemented yet
};    

class VulkanGaussianSplatting : virtual public RenderGraphElement, virtual public DrawGraphGroup {
/**
 * @brief 
 * 
 * Gaussian Splatting consists of these steps:
 * 1. project the 3D Gaussian to 2D
 * 2. sort the 2D Gaussians by depth using radix sort
 * 3. distribute the 2D Gaussians to 4x4 subtiles
 * 4. splat the 2D Gaussians to each subtile of the image
 * 
 * Here, this is solved by a drawgraph consisting of
 * 1. shader drawgraphelent (projection shader, 3D gaussians in, 2D gaussians out)
 * 2. sorting shader (2D gaussians in, sorted 2D gaussians out)
 */
public:
    VulkanGaussianSplatting(std::string path, GaussianSplattingRenderingType type=GaussianSplattingRenderingType::PointCloud);
    ~VulkanGaussianSplatting();

    virtual void checkInput(DrawGraphElementPtr input, int index = 0);

    virtual void _setup(VulkanKernel& vulkanKernel, uint32_t numberPaths);

    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId);

    virtual const char* getType() const
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
    uint32_t number_of_gaussians;

    uint32_t numberOfPaths = 0;

    // 
    std::unique_ptr<VulkanBuffer<Gaussian2D>> projectedGaussians;

    std::unique_ptr<GaussianProjection> project3Dto2D;

    std::unique_ptr<GaussianSort> sort2DGaussians;

    std::unique_ptr<GaussianBinning> bin;

    std::unique_ptr<GaussianComputeBounds> computeBounds;

    std::unique_ptr<GaussianSplatting> splat;
   
};

} // namespace klartraum

#endif // VULKAN_GAUSSIAN_SPLATTING_HPP