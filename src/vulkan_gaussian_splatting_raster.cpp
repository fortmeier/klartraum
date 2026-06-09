#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include <glm/glm.hpp>

#include "load-spz.h"

#include "klartraum/vulkan_gaussian_splatting_raster.hpp"
#include "klartraum/draw_basics.hpp"

namespace klartraum {

static float sigmoidRaster(float x) { return 1.0f / (1.0f + std::exp(-x)); }

VulkanGaussianSplattingRaster::VulkanGaussianSplattingRaster(
    VulkanContext& vulkanContext,
    std::shared_ptr<ImageViewSrc> imageViewSrc,
    std::shared_ptr<CameraUboType> cameraUBO,
    std::string path,
    GsplatConfig config)
{
    loadSPZModel(path);
    initialize(vulkanContext, imageViewSrc, cameraUBO, config);
}

VulkanGaussianSplattingRaster::VulkanGaussianSplattingRaster(
    VulkanContext& vulkanContext,
    std::shared_ptr<ImageViewSrc> imageViewSrc,
    std::shared_ptr<CameraUboType> cameraUBO,
    std::vector<Gaussian3D> gaussians,
    GsplatConfig config)
{
    gaussians3DData     = std::move(gaussians);
    number_of_gaussians = static_cast<uint32_t>(gaussians3DData.size());
    initialize(vulkanContext, imageViewSrc, cameraUBO, config);
}

VulkanGaussianSplattingRaster::~VulkanGaussianSplattingRaster() {}

void VulkanGaussianSplattingRaster::_setup(VulkanContext& vulkanContext, uint32_t numberPaths) {
    ComputeGraphElement::_setup(vulkanContext, numberPaths);  // marks this element initialized

    // drawArgs's per-path buffers only exist after this point (graph compile
    // allocates them). vertexCount=4 (the rasterizer's triangle-strip quad) is
    // constant across frames, so seed it once here rather than every _record —
    // setRecordToZeroRange above only resets instanceCount each frame, leaving
    // this byte range untouched.
    VkCommandBufferAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool        = vulkanContext.getCommandPool();
    ai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(vulkanContext.getDevice(), &ai, &cmd);

    VkCommandBufferBeginInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &bi);

    for (uint32_t pathId = 0; pathId < numberPaths; pathId++) {
        vkCmdFillBuffer(cmd, drawArgs->getVkBuffer(pathId),
                        offsetof(VkDrawIndirectCommand, vertexCount), sizeof(uint32_t), 4u);
    }

    vkEndCommandBuffer(cmd);
    VkSubmitInfo si{};
    si.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &cmd;
    vkQueueSubmit(vulkanContext.getGraphicsQueue(), 1, &si, VK_NULL_HANDLE);
    vkQueueWaitIdle(vulkanContext.getGraphicsQueue());
    vkFreeCommandBuffers(vulkanContext.getDevice(), vulkanContext.getCommandPool(), 1, &cmd);
}

void VulkanGaussianSplattingRaster::checkInput(ComputeGraphElementPtr input, int index) {
    if (index == 0 && !std::dynamic_pointer_cast<ImageViewSrc>(input))
        throw std::runtime_error("VulkanGaussianSplattingRaster: input 0 must be ImageViewSrc");
    if (index == 1 && !std::dynamic_pointer_cast<CameraUboType>(input))
        throw std::runtime_error("VulkanGaussianSplattingRaster: input 1 must be CameraUboType");
    if (index > 1)
        throw std::runtime_error("VulkanGaussianSplattingRaster: input index out of range");
}

void VulkanGaussianSplattingRaster::initialize(
    VulkanContext& vulkanContext,
    std::shared_ptr<ImageViewSrc> imageViewSrc,
    std::shared_ptr<CameraUboType> cameraUBO,
    GsplatConfig config)
{
    this->vulkanContext = &vulkanContext;
    this->config_       = config;
    this->setInput(imageViewSrc, 0);
    this->setInput(cameraUBO,    1);

    const uint32_t N = number_of_gaussians;

    // Convert AoS -> SoA and upload to GPU (single-path, static), exactly as
    // the compute-tile backend does — both backends draw from the same model.
    std::vector<glm::vec3> pos3d(N), scale3d(N);
    std::vector<glm::vec4> rot3d(N), colAlpha3d(N);
    std::vector<float>     shR(15*N), shG(15*N), shB(15*N);

    for (uint32_t i = 0; i < N; i++) {
        const auto& g = gaussians3DData[i];
        pos3d[i]      = {g.position[0], g.position[1], g.position[2]};
        rot3d[i]      = {g.rotation[0], g.rotation[1], g.rotation[2], g.rotation[3]};
        scale3d[i]    = {g.scale[0],    g.scale[1],    g.scale[2]};
        colAlpha3d[i] = {g.color[0],    g.color[1],    g.color[2],    g.alpha};
        for (int b = 0; b < 15; b++) {
            shR[b * N + i] = g.shR[b];
            shG[b * N + i] = g.shG[b];
            shB[b * N + i] = g.shB[b];
        }
    }

    buf3DPos      = std::make_shared<BufferElementSinglePath<VulkanBuffer<glm::vec3>>>(vulkanContext, N);
    buf3DRot      = std::make_shared<BufferElementSinglePath<VulkanBuffer<glm::vec4>>>(vulkanContext, N);
    buf3DScale    = std::make_shared<BufferElementSinglePath<VulkanBuffer<glm::vec3>>>(vulkanContext, N);
    buf3DColAlpha = std::make_shared<BufferElementSinglePath<VulkanBuffer<glm::vec4>>>(vulkanContext, N);
    buf3DShR      = std::make_shared<BufferElementSinglePath<VulkanBuffer<float>>>(vulkanContext, 15*N);
    buf3DShG      = std::make_shared<BufferElementSinglePath<VulkanBuffer<float>>>(vulkanContext, 15*N);
    buf3DShB      = std::make_shared<BufferElementSinglePath<VulkanBuffer<float>>>(vulkanContext, 15*N);

    buf3DPos->setName("RasterPos3D");
    buf3DRot->setName("RasterRot3D");
    buf3DScale->setName("RasterScale3D");
    buf3DColAlpha->setName("RasterColAlpha3D");
    buf3DShR->setName("RasterShR");
    buf3DShG->setName("RasterShG");
    buf3DShB->setName("RasterShB");

    buf3DPos->getBuffer().memcopyFrom(pos3d);
    buf3DRot->getBuffer().memcopyFrom(rot3d);
    buf3DScale->getBuffer().memcopyFrom(scale3d);
    buf3DColAlpha->getBuffer().memcopyFrom(colAlpha3d);
    buf3DShR->getBuffer().memcopyFrom(shR);
    buf3DShG->getBuffer().memcopyFrom(shG);
    buf3DShB->getBuffer().memcopyFrom(shB);

    // --- Stage A: cull + depth-key + compaction (gsplat_dist.comp) ---
    const VkBufferUsageFlags storageDst =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    keysA    = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, N, storageDst);
    indicesA = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, N, storageDst);
    keysB    = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, N, storageDst);
    indicesB = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, N, storageDst);
    keysA->setName("RasterKeysA");
    indicesA->setName("RasterIndicesA");
    keysB->setName("RasterKeysB");
    indicesB->setName("RasterIndicesB");

    // The sort below runs only over the visible count dist compacts into the
    // front of these buffers (slots [0, instanceCount)), so no sentinel fill of
    // the tail is needed — the sort never reads past instanceCount.

    drawArgs = std::make_shared<DrawIndirectCommandBufferElement>(vulkanContext, 1,
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | storageDst);
    drawArgs->setName("RasterDrawArgs");
    drawArgs->setRecordToZeroRange(offsetof(VkDrawIndirectCommand, instanceCount), sizeof(uint32_t));

    // dist runs two sub-shaders: gsplat_dist.comp (cull + depth-key + compaction)
    // then gsplat_dist_count.comp, which copies the finalized instanceCount into
    // the sort's count buffer (binding 5) so the sort processes only the visible
    // range (perf plan R3). The count buffer (totalCount) is wired below once it
    // is created. GeneralComputation's inter-pipeline barrier orders the two.
    dist = vulkanContext.create<GaussianDist>(std::vector<std::string>{
        "shaders/gsplat/gsplat_dist.comp.spv",
        "shaders/gsplat/gsplat_dist_count.comp.spv"});
    dist->setName("GaussianDist");
    dist->setInput(buf3DPos,  0);
    dist->setInput(cameraUBO, 1);
    dist->setInput(keysA,     2);
    dist->setInput(indicesA,  3);
    dist->setInput(drawArgs,  4);
    dist->setGroupCountX((N + 255) / 256);
    dist->setPushConstants({{N, 0.1f}});

    // --- Stage A2: per-splat 2D attribute precompute (perf plan R1+R2) ---
    // Projects covariance, eigendecomposes, and evaluates SH once per splat into
    // the Splat2D buffer the vertex shader reads verbatim (16 floats / splat,
    // 64-byte stride matching gsplat_raster_project.comp's scalar struct). Keyed
    // by splat id; the vertex shader dereferences it through the sorted index.
    splat2D = std::make_shared<BufferElement<VulkanBuffer<float>>>(
        vulkanContext, 16 * N, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    splat2D->setName("RasterSplat2D");

    project = vulkanContext.create<GaussianRasterProject>(
        "shaders/gsplat/gsplat_raster_project.comp.spv");
    project->setName("RasterProject");
    project->setInput(buf3DPos,      0);
    project->setInput(buf3DRot,      1);
    project->setInput(buf3DScale,    2);
    project->setInput(buf3DColAlpha, 3);
    project->setInput(buf3DShR,      4);
    project->setInput(buf3DShG,      5);
    project->setInput(buf3DShB,      6);
    project->setInput(cameraUBO,     7);
    project->setInput(splat2D,       8);
    project->setGroupCountX((N + 255) / 256);
    project->setPushConstants({{N,
        (float)imageViewSrc->getImageExtent(0).width,
        (float)imageViewSrc->getImageExtent(0).height,
        3.0f, config.shDegree}});

    // --- Stage B: radix sort over the visible count (perf plan R3). The reused
    // sort operates on plain (uint key, uint index) pairs; 8 passes x 4 bits
    // sorts the full 32-bit key, and an even pass count returns the result to
    // the A buffers dist wrote into. The dispatch group count stays fixed at
    // numSortWGs (so the histogram stride is consistent), but the per-workgroup
    // item count is driven by the visible count read from the count buffer. ---
    const uint32_t numBins    = 16;
    const uint32_t tpg        = 256;
    const uint32_t numSortWGs = std::max(1u, std::min(config.numSortWGsCap, N / tpg + 1));

    scratchHist    = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, numBins * numSortWGs);
    scratchCounts  = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, numBins);
    scratchOffsets = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, numSortWGs + 1);
    totalCount     = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, 1);
    scratchHist->setRecordToZero(true);
    scratchCounts->setRecordToZero(true);
    scratchOffsets->setRecordToZero(true);
    // totalCount holds the visible count, fully written each frame by dist's
    // count sub-shader — no per-frame zeroing.
    totalCount->setRecordToZero(false);
    scratchHist->setName("RasterSortScratchHist");
    scratchCounts->setName("RasterSortScratchCounts");
    scratchOffsets->setName("RasterSortScratchOffsets");
    totalCount->setName("RasterSortTotalCount");

    // Wire the count buffer as dist's binding 5 so gsplat_dist_count.comp writes
    // the visible count into it; the existing dist -> sort edge then makes that
    // write visible to the sort, which reads it as binding 6 (inputBuffer2).
    dist->setInput(totalCount, 5);

    sortOp = std::make_shared<RadixSort>(vulkanContext, std::vector<std::string>{
        "shaders/gsplat/gsplat_radix_sort_histogram.comp.spv",
        "shaders/gsplat/gsplat_radix_sort_hist_prefix_sum.comp.spv",
        "shaders/gsplat/gsplat_radix_sort_hist_scatter.comp.spv"
    });
    sortOp->setName("RasterRadixSort");
    sortOp->setInput(dist, 0, 2);  // keysA    (dist's output index 2)
    sortOp->setInput(dist, 1, 3);  // indicesA (dist's output index 3)
    sortOp->setInput(keysB,    2);
    sortOp->setInput(indicesB, 3);
    sortOp->addScratchBufferElement(scratchCounts,  true);
    sortOp->addScratchBufferElement(scratchOffsets, true);
    sortOp->addScratchBufferElement(totalCount,     false);
    sortOp->addScratchBufferElement(scratchHist,    true);
    sortOp->setGroupCountX(numSortWGs);
    {
        // useCountBuffer = 1: the sort reads the active count from totalCount
        // (the visible count dist wrote) instead of N, so it sorts only the
        // visible range. The group count stays numSortWGs, keeping the
        // histogram's bin-major stride consistent across the three passes.
        std::vector<SortPushConstants> pcs;
        for (uint32_t i = 0; i < 32 / 4; ++i)
            pcs.push_back({i, N, numBins, 1u});
        sortOp->setPushConstants(pcs);
    }

    // --- Order compute writes (sorted indices + draw args) before the render
    // pass reads them indirectly/in the vertex shader (guide §3.1). ---
    barrier = std::make_shared<BufferToGraphicsBarrier>();
    barrier->setName("RasterBarrier");
    barrier->addBuffer(indicesA);
    barrier->addBuffer(drawArgs);
    barrier->addBuffer(splat2D);
    barrier->setInput(sortOp,   0, 1);  // indicesA, sorted back into A by the even pass count
    barrier->setInput(dist,     1, 4);  // drawArgs (instanceCount written by dist's atomicAdd)
    barrier->setInput(project,  2, 8);  // splat2D (per-splat attributes), output binding 8

    // --- Stage C: instanced indirect draw, hardware rasterized ---
    auto extent = imageViewSrc->getImageExtent(0);
    // The vertex shader reads only the precomputed Splat2D buffer (binding 0)
    // dereferenced through the sorted-index permutation (binding 1) — all the
    // per-splat SoA inputs are now consumed by the project pass instead.
    rasterizer = std::make_shared<GaussianSplatRasterizer>(
        std::vector<std::shared_ptr<BufferElementInterface>>{ splat2D, indicesA },
        drawArgs);
    GaussianSplatRasterPushConstants pushConstants{};
    pushConstants.resolution = glm::vec2((float)extent.width, (float)extent.height);
    pushConstants.focal      = glm::vec2(1000.0f, 1000.0f);
    // Sigma multiplier for the EWA-covariance quad extent — 3.0 covers ~99.7%
    // of each Gaussian (guide §5C), the fragment shader's per-pixel conic
    // evaluation handles the exact falloff within that quad.
    pushConstants.splatScale = 3.0f;
    pushConstants.shDegree   = 0;
    pushConstants.numSplats  = N;
    rasterizer->setPushConstants(pushConstants);

    renderPass = std::make_shared<RenderPass>(vulkanContext.getSwapChainImageFormat(), extent);
    renderPass->setName("RasterRenderPass");
    renderPass->setInput(imageViewSrc, 0);
    renderPass->setInput(cameraUBO,    1);
    renderPass->addDrawComponent(rasterizer);
    renderPass->addComputeDependency(barrier);

    outputElements[0] = renderPass;
}

void VulkanGaussianSplattingRaster::loadSPZModel(std::string path) {
    spz::PackedGaussians packed = spz::loadSpzPacked(path);
    gaussians3DData.clear();
    gaussians3DData.reserve(packed.numPoints);
    spz::CoordinateConverter conv;

    for (int i = 0; i < packed.numPoints; i++) {
        spz::UnpackedGaussian ug = packed.unpack(i, conv);
        Gaussian3D g;
        memcpy(&g, &ug, sizeof(spz::UnpackedGaussian));
        g.alpha    = sigmoidRaster(ug.alpha);
        g.scale[0] = std::exp(ug.scale[0]);
        g.scale[1] = std::exp(ug.scale[1]);
        g.scale[2] = std::exp(ug.scale[2]);
        gaussians3DData.push_back(g);
    }
    number_of_gaussians = static_cast<uint32_t>(gaussians3DData.size());
    std::cout << "Loaded " << number_of_gaussians << " gaussians from " << path << "\n";
}

} // namespace klartraum
