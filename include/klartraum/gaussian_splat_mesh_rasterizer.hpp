#ifndef KLARTRAUM_GAUSSIAN_SPLAT_MESH_RASTERIZER_HPP
#define KLARTRAUM_GAUSSIAN_SPLAT_MESH_RASTERIZER_HPP

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vector>
#include <memory>

#include "klartraum/draw_component.hpp"
#include "klartraum/computegraph/bufferelement.hpp"

namespace klartraum {

// Mesh-shader draw component for the raster backend (perf plan R5), an additive,
// device-gated alternative to GaussianSplatRasterizer's vertex+fragment path.
// One mesh workgroup emits a batch of quads, reading the precomputed Splat2D
// records (R1) dereferenced through the sorted-index permutation plus the
// visible count (so the last partial workgroup emits the right number). It is
// selected only when VK_EXT_mesh_shader is supported; otherwise the raster
// backend uses the vertex path. Outputs match the vertex shader's interface, so
// gsplat_raster.frag is reused. Draws via vkCmdDrawMeshTasksIndirectEXT with
// groupCount filled by gsplat_mesh_args.comp from the visible count.
class GaussianSplatMeshRasterizer : public DrawComponent {
public:
    // splatBuffers (set 1): Splat2D (binding 0), sorted indices (binding 1),
    //   visible-count buffer (binding 2).
    // meshArgsBuffer: VkDrawMeshTasksIndirectCommandEXT buffer (groupCountX/Y/Z).
    GaussianSplatMeshRasterizer(std::vector<std::shared_ptr<BufferElementInterface>> splatBuffers,
                                std::shared_ptr<BufferElementInterface> meshArgsBuffer);
    ~GaussianSplatMeshRasterizer();

    virtual void initialize(VulkanContext& vulkanContext, VkRenderPass& renderPass, std::shared_ptr<CameraUboType> cameraUBO) override;

    void recordCommandBuffer(VkCommandBuffer commandBuffer, VkFramebuffer framebuffer, uint32_t pathId) override;

private:
    void createSplatDescriptorSetLayout();
    void createDescriptorPool();
    void createDescriptorSets();
    void createGraphicsPipeline();

    std::vector<std::shared_ptr<BufferElementInterface>> splatBuffers;
    std::shared_ptr<BufferElementInterface> meshArgsBuffer;

    VkDescriptorSetLayout splatDescriptorSetLayout;
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> splatDescriptorSets;

    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;
};

} // namespace klartraum

#endif // KLARTRAUM_GAUSSIAN_SPLAT_MESH_RASTERIZER_HPP
