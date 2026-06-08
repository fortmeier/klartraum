#ifndef VULKAN_GAUSSIAN_SPLATTING_RASTER_HPP
#define VULKAN_GAUSSIAN_SPLATTING_RASTER_HPP

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

namespace klartraum {

// Sort-once + hardware-rasterization Gaussian-splatting backend
// (klartraum_rasterized_gs_backend_guide.md §5/§7 step 5 "Composite"), the
// counterpart to VulkanGaussianSplatting's compute-tile rasterizer. Loads the
// same SPZ/Gaussian3D model into SoA buffers, then wires:
//   GaussianDist (cull + depth-key + compaction, dist.comp)
//     -> sentinel-filled keys/indices ping-pong -> RadixSort (reused, unmodified)
//     -> BufferToGraphicsBarrier (compute-write -> indirect-draw/vertex-read ordering)
//     -> RenderPass{ GaussianSplatRasterizer } (instanced indirect quad draw,
//        premultiplied "over" blending)
// outputElements[0] is the internal RenderPass — the single image-producing
// element the ComputeGraph traversal needs to reach everything else.
class VulkanGaussianSplattingRaster : public RenderGraphElement,
                                      public ComputeGraphGroup {
public:
    VulkanGaussianSplattingRaster(
        VulkanContext& vulkanContext,
        std::shared_ptr<ImageViewSrc> imageViewSrc,
        std::shared_ptr<CameraUboType> cameraUBO,
        std::string path,
        GsplatConfig config = GsplatConfig{});

    VulkanGaussianSplattingRaster(
        VulkanContext& vulkanContext,
        std::shared_ptr<ImageViewSrc> imageViewSrc,
        std::shared_ptr<CameraUboType> cameraUBO,
        std::vector<Gaussian3D> gaussians,
        GsplatConfig config = GsplatConfig{});

    ~VulkanGaussianSplattingRaster();

    virtual void checkInput(ComputeGraphElementPtr input, int index = 0) override;
    virtual void _setup(VulkanContext& vulkanContext, uint32_t numberPaths) override;

    virtual const char* getType() const override { return "GaussianSplattingRaster"; }

    void setPushConstants(const GaussianSplatRasterPushConstants& pushConstants) {
        rasterizer->setPushConstants(pushConstants);
    }

private:
    void loadSPZModel(std::string path);
    void initialize(VulkanContext& vulkanContext,
                    std::shared_ptr<ImageViewSrc> imageViewSrc,
                    std::shared_ptr<CameraUboType> cameraUBO,
                    GsplatConfig config);

    GsplatConfig config_;

    VulkanContext* vulkanContext = nullptr;
    uint32_t number_of_gaussians = 0;

    std::vector<Gaussian3D> gaussians3DData;

    // SoA 3D input buffers (single-path: static data), uploaded once.
    std::shared_ptr<BufferElementSinglePath<VulkanBuffer<glm::vec3>>> buf3DPos;
    std::shared_ptr<BufferElementSinglePath<VulkanBuffer<glm::vec4>>> buf3DRot;
    std::shared_ptr<BufferElementSinglePath<VulkanBuffer<glm::vec3>>> buf3DScale;
    std::shared_ptr<BufferElementSinglePath<VulkanBuffer<glm::vec4>>> buf3DColAlpha;
    std::shared_ptr<BufferElementSinglePath<VulkanBuffer<float>>>     buf3DShR;
    std::shared_ptr<BufferElementSinglePath<VulkanBuffer<float>>>     buf3DShG;
    std::shared_ptr<BufferElementSinglePath<VulkanBuffer<float>>>     buf3DShB;

    // Stage A: cull + depth-key + compaction
    std::shared_ptr<GaussianDist> dist;

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
};

} // namespace klartraum

#endif // VULKAN_GAUSSIAN_SPLATTING_RASTER_HPP
