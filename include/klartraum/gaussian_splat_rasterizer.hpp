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

// DrawComponent for the sort-once + hardware-rasterization Gaussian-splatting
// backend (klartraum_rasterized_gs_backend_guide.md §3.4 / Stage C). Unlike
// DrawBasics (one descriptor set, no push constants, a vertex buffer, direct
// draw), this component binds:
//   - set 0: the camera UBO (as DrawBasics does)
//   - set 1: the per-splat SoA storage buffers plus the sorted-index buffer,
//     in the order supplied to the constructor (binding == vector index)
// generates its quad corners from gl_VertexIndex (no vertex buffer), and
// issues a single vkCmdDrawIndirect reading instanceCount from drawArgsBuffer
// (filled by the compute culling/sort stage and ordered ahead of this render
// pass via BufferToGraphicsBarrier). The color attachment is blended with
// premultiplied "over" so splats composite correctly back-to-front.
class GaussianSplatRasterizer : public DrawComponent {
public:
    // splatBuffers: per-splat SoA storage buffers (e.g. positions, colorsAlpha)
    //   followed by the sorted-index buffer — bound as set 1, binding == index.
    // drawArgsBuffer: VkDrawIndirectCommand buffer written by the compute stage
    //   (vertexCount = 4, instanceCount = visible splat count after sort/cull).
    GaussianSplatRasterizer(std::vector<std::shared_ptr<BufferElementInterface>> splatBuffers,
                            std::shared_ptr<BufferElementInterface> drawArgsBuffer);
    ~GaussianSplatRasterizer();

    virtual void initialize(VulkanContext& vulkanContext, VkRenderPass& renderPass, std::shared_ptr<CameraUboType> cameraUBO) override;

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
