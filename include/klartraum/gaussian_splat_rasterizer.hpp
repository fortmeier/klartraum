// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_GAUSSIAN_SPLAT_RASTERIZER_HPP
#define KLARTRAUM_GAUSSIAN_SPLAT_RASTERIZER_HPP

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <vector>
#include <memory>

#include "klartraum/draw_component.hpp"
#include "klartraum/computegraph/bufferelement.hpp"

namespace klartraum {

// Layout must match the GLSL `push_constant` block in gsplat_raster.vert
// (std430 scalar rules: vec2/vec2/float/int pack tightly, no padding).
struct GaussianSplatRasterPushConstants {
    glm::vec2 resolution;
    glm::vec2 focal;
    float splatScale;
    int32_t shDegree;
    uint32_t numSplats;
};

/**
 * @brief Draw component of the raster Gaussian splatting backend (vertex and fragment shader path).
 *
 * It binds
 * - set 0: the camera uniform buffer;
 * - set 1: the per-splat storage buffers and the sorted index buffer, in the
 *   order passed to the constructor (binding = vector index);
 *
 * generates the quad corners from `gl_VertexIndex` (no vertex buffer), and issues
 * one `vkCmdDrawIndirect` that reads the instance count from the draw-arguments
 * buffer. That buffer is filled by the compute culling and sorting, which
 * BufferToGraphicsBarrier orders before this render pass. Splats are blended with
 * premultiplied "over", so they composite correctly back to front.
 */
class GaussianSplatRasterizer : public DrawComponent {
public:
    // splatBuffers: per-splat SoA storage buffers (e.g. positions, colorsAlpha)
    //   followed by the sorted-index buffer — bound as set 1, binding == index.
    // drawArgsBuffer: VkDrawIndirectCommand buffer written by the compute stage
    //   (vertexCount = 4, instanceCount = visible splat count after sort/cull).
    GaussianSplatRasterizer(std::vector<std::shared_ptr<BufferElementInterface>> splatBuffers,
                            std::shared_ptr<BufferElementInterface> drawArgsBuffer);
    ~GaussianSplatRasterizer();

    virtual void initialize(VulkanContext& vulkanContext, VkRenderPass& renderPass,
                            std::shared_ptr<CameraUboType> cameraUBO) override;

    void recordCommandBuffer(VkCommandBuffer commandBuffer, VkFramebuffer framebuffer, uint32_t pathId) override;

    void setPushConstants(const GaussianSplatRasterPushConstants& pushConstants) {
        this->pushConstants = pushConstants;
    }

private:
    void createSplatDescriptorSetLayout();
    void createDescriptorPool();
    void createDescriptorSets();
    void createGraphicsPipeline();

    std::vector<std::shared_ptr<BufferElementInterface>> splatBuffers;
    std::shared_ptr<BufferElementInterface> drawArgsBuffer;

    VkDescriptorSetLayout splatDescriptorSetLayout;
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> splatDescriptorSets;

    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    GaussianSplatRasterPushConstants pushConstants{};
};

} // namespace klartraum

#endif // KLARTRAUM_GAUSSIAN_SPLAT_RASTERIZER_HPP
