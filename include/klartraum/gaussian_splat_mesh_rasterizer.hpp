// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_GAUSSIAN_SPLAT_MESH_RASTERIZER_HPP
#define KLARTRAUM_GAUSSIAN_SPLAT_MESH_RASTERIZER_HPP

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vector>
#include <memory>

#include "klartraum/draw_component.hpp"
#include "klartraum/computegraph/bufferelement.hpp"

namespace klartraum {

/**
 * @brief Mesh-shader draw component of the raster Gaussian splatting backend, an alternative to
 * GaussianSplatRasterizer's vertex shader path.
 *
 * One mesh workgroup emits a batch of quads from the precomputed 2D splat
 * records, read through the sorted-index permutation, and from the visible count
 * (so the last, partial workgroup emits the right number). The backend uses it
 * only when VK_EXT_mesh_shader is supported, otherwise the vertex path. Its outputs
 * match the vertex shader's, so the same fragment shader (`gsplat_raster.frag`)
 * is used. It draws with `vkCmdDrawMeshTasksIndirectEXT`; `gsplat_mesh_args.comp`
 * computes the group count from the visible count.
 */
class GaussianSplatMeshRasterizer : public DrawComponent {
public:
    // splatBuffers (set 1): Splat2D (binding 0), sorted indices (binding 1),
    //   visible-count buffer (binding 2).
    // meshArgsBuffer: VkDrawMeshTasksIndirectCommandEXT buffer (groupCountX/Y/Z).
    GaussianSplatMeshRasterizer(std::vector<std::shared_ptr<BufferElementInterface>> splatBuffers,
                                std::shared_ptr<BufferElementInterface> meshArgsBuffer);
    ~GaussianSplatMeshRasterizer();

    virtual void initialize(VulkanContext& vulkanContext, VkRenderPass& renderPass,
                            std::shared_ptr<CameraUboType> cameraUBO) override;

    void recordCommandBuffer(VkCommandBuffer commandBuffer, VkFramebuffer framebuffer, uint32_t pathId) override;

    // The extent of the render target this draws into. Must be set to the
    // viewport/offscreen extent; if left zero the swapchain extent is used.
    void setTargetExtent(VkExtent2D extent) { targetExtent = extent; }

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

    VkExtent2D targetExtent{};
};

} // namespace klartraum

#endif // KLARTRAUM_GAUSSIAN_SPLAT_MESH_RASTERIZER_HPP
