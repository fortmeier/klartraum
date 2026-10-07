// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_VULKAN_GAUSSIAN_SPLATTING_RASTER_HPP
#define KLARTRAUM_VULKAN_GAUSSIAN_SPLATTING_RASTER_HPP

#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "klartraum/computegraph/computegraphgroup.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/computegraph/rendergraphelement.hpp"
#include "klartraum/computegraph/renderpass.hpp"
#include "klartraum/computegraph/buffertographicsbarrier.hpp"
#include "klartraum/vulkan_buffer.hpp"
#include "klartraum/vulkan_gaussian_splatting_types.hpp"
#include "klartraum/gaussian_splat_rasterizer.hpp"
#include "klartraum/gaussian_splat_mesh_rasterizer.hpp"

namespace klartraum {

/**
 * @brief Gaussian splatting backend that sorts the Gaussians once and draws them with hardware rasterization.
 *
 * The counterpart to VulkanGaussianSplatting, which rasterizes in compute shaders.
 * It reads the same Gaussian data and connects:
 * - GaussianDist: culling, depth keys and compaction (`gsplat_dist.comp`);
 * - a radix sort of the depth keys;
 * - BufferToGraphicsBarrier, which orders the indirect draw after the compute work;
 * - a RenderPass with a GaussianSplatRasterizer (or, where mesh shaders are
 *   supported, a GaussianSplatMeshRasterizer) that draws instanced quads with
 *   premultiplied "over" blending.
 *
 * The internal RenderPass is the group's output element: the single
 * image-producing element through which the compute graph reaches all others.
 */
class VulkanGaussianSplattingRaster : public RenderGraphElement, public ComputeGraphGroup {
public:
    VulkanGaussianSplattingRaster(VulkanContext& vulkanContext, std::shared_ptr<ImageViewSrc> imageViewSrc,
                                  std::shared_ptr<CameraUboType> cameraUBO, GaussianSoABuffers buffers,
                                  GsplatConfig config = GsplatConfig{});

    ~VulkanGaussianSplattingRaster();

    virtual void checkInput(ComputeGraphElementPtr input, int index = 0) override;
    virtual void _setup(VulkanContext& vulkanContext, uint32_t numberPaths) override;

    virtual const char* getType() const override { return "GaussianSplattingRaster"; }

    void setPushConstants(const GaussianSplatRasterPushConstants& pushConstants) {
        rasterizer->setPushConstants(pushConstants);
    }

private:
    void initialize(VulkanContext& vulkanContext, std::shared_ptr<ImageViewSrc> imageViewSrc,
                    std::shared_ptr<CameraUboType> cameraUBO, GsplatConfig config);

    GsplatConfig config;

    VulkanContext* vulkanContext = nullptr;

    // Static SoA input buffers (position, rotation, scale, colour+alpha, SH
    // R/G/B) + splat count, injected by the caller. Held as shared handles so a
    // single upload can be shared across backends/viewports without reloading.
    GaussianSoABuffers buffers;

    // Stage A: cull + depth-key + compaction
    std::shared_ptr<GaussianDist> dist;

    // Per-splat 2D attribute precompute (project once per splat, not 4x per
    // vertex): writes the Splat2D buffer the vertex shader reads verbatim.
    std::shared_ptr<GaussianRasterProject> project;
    std::shared_ptr<BufferElement<VulkanBuffer<float>>> splat2D;

    // Stage B: ping-pong key/index buffers (A holds dist's output and, after
    // an even pass count, the final sorted result the rasterizer reads) plus
    // the existing radix sort's scratch buffers.
    std::shared_ptr<BufferElement<VulkanBuffer<uint32_t>>> keysA;
    std::shared_ptr<BufferElement<VulkanBuffer<uint32_t>>> indicesA;
    std::shared_ptr<BufferElement<VulkanBuffer<uint32_t>>> keysB;
    std::shared_ptr<BufferElement<VulkanBuffer<uint32_t>>> indicesB;
    std::shared_ptr<BufferElement<VulkanBuffer<uint32_t>>> scratchHist;
    std::shared_ptr<BufferElement<VulkanBuffer<uint32_t>>> scratchCounts;
    std::shared_ptr<BufferElement<VulkanBuffer<uint32_t>>> scratchOffsets;
    std::shared_ptr<BufferElement<VulkanBuffer<uint32_t>>> totalCount;
    std::shared_ptr<RadixSort> sortOp;

    std::shared_ptr<DrawIndirectCommandBufferElement> drawArgs;

    std::shared_ptr<BufferToGraphicsBarrier> barrier;
    std::shared_ptr<RenderPass> renderPass;
    std::shared_ptr<GaussianSplatRasterizer> rasterizer;

    // Optional VK_EXT_mesh_shader draw path (perf plan R5), selected when
    // config.useMeshShader && the device supports mesh shaders; otherwise the
    // vertex `rasterizer` above is used. meshArgs holds the
    // VkDrawMeshTasksIndirectCommandEXT filled by meshArgsOp from the visible
    // count.
    std::shared_ptr<BufferElement<VulkanBuffer<VkDrawMeshTasksIndirectCommandEXT>>> meshArgs;
    std::shared_ptr<MeshArgsFill> meshArgsOp;
    std::shared_ptr<GaussianSplatMeshRasterizer> meshRasterizer;
};

} // namespace klartraum

#endif // KLARTRAUM_VULKAN_GAUSSIAN_SPLATTING_RASTER_HPP
