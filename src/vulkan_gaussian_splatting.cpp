#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <glm/glm.hpp>

#include "load-spz.h"

#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/vulkan_gaussian_splatting.hpp"

namespace klartraum {

static float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

VulkanGaussianSplatting::VulkanGaussianSplatting(
    VulkanContext& vulkanContext,
    std::shared_ptr<ImageViewSrc> _imageViewSrc,
    std::shared_ptr<CameraUboType> _cameraUBO,
    std::string path)
{
    loadSPZModel(path);
    initialize(vulkanContext, _imageViewSrc, _cameraUBO);
}

VulkanGaussianSplatting::VulkanGaussianSplatting(
    VulkanContext& vulkanContext,
    std::shared_ptr<ImageViewSrc> _imageViewSrc,
    std::shared_ptr<CameraUboType> _cameraUBO,
    std::vector<Gaussian3D> gaussians)
{
    gaussians3DData     = std::move(gaussians);
    number_of_gaussians = static_cast<uint32_t>(gaussians3DData.size());
    initialize(vulkanContext, _imageViewSrc, _cameraUBO);
}

void VulkanGaussianSplatting::initialize(
    VulkanContext& vulkanContext,
    std::shared_ptr<ImageViewSrc> _imageViewSrc,
    std::shared_ptr<CameraUboType> _cameraUBO)
{
    this->vulkanContext = &vulkanContext;
    this->setInput(_imageViewSrc, 0);
    this->setInput(_cameraUBO,    1);

    auto imageViewSrc = std::dynamic_pointer_cast<ImageViewSrc>(getInputElement(0));
    if (!imageViewSrc) throw std::runtime_error("VulkanGaussianSplatting: input 0 is not ImageViewSrc");

    VkExtent2D ext = imageViewSrc->getImageExtent(0);
    const float W  = static_cast<float>(ext.width);
    const float H  = static_cast<float>(ext.height);
    const uint32_t N          = number_of_gaussians;
    const uint32_t gridSize   = 4;
    const uint32_t numBins    = gridSize * gridSize;
    const uint32_t tpg        = 128;
    const uint32_t maxMod     = 2;
    const uint32_t maxBinned  = N * maxMod;
    const uint32_t numBinWGs  = N / tpg + 1;
    const uint32_t numSortWGs = std::max(1u, std::min(320u, maxBinned / tpg + 1));

    // Convert AoS → SoA and upload to GPU (single-path, static)
    std::vector<glm::vec3> pos3d(N), scale3d(N);
    std::vector<glm::vec4> rot3d(N), colAlpha3d(N);
    std::vector<float>     shR(15*N), shG(15*N), shB(15*N);

    for (uint32_t i = 0; i < N; i++) {
        const auto& g = gaussians3DData[i];
        pos3d[i]     = {g.position[0], g.position[1], g.position[2]};
        rot3d[i]     = {g.rotation[0], g.rotation[1], g.rotation[2], g.rotation[3]};
        scale3d[i]   = {g.scale[0],    g.scale[1],    g.scale[2]};
        colAlpha3d[i]= {g.color[0],    g.color[1],    g.color[2],    g.alpha};
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

    buf3DPos->setName("Pos3D");
    buf3DRot->setName("Rot3D");
    buf3DScale->setName("Scale3D");
    buf3DColAlpha->setName("ColAlpha3D");
    buf3DShR->setName("ShR");
    buf3DShG->setName("ShG");
    buf3DShB->setName("ShB");

    buf3DPos->getBuffer().memcopyFrom(pos3d);
    buf3DRot->getBuffer().memcopyFrom(rot3d);
    buf3DScale->getBuffer().memcopyFrom(scale3d);
    buf3DColAlpha->getBuffer().memcopyFrom(colAlpha3d);
    buf3DShR->getBuffer().memcopyFrom(shR);
    buf3DShG->getBuffer().memcopyFrom(shG);
    buf3DShB->getBuffer().memcopyFrom(shB);

    // Projected 2D outputs (per-path)
    auto proj2DPos2D   = std::make_shared<BufferElement<VulkanBuffer<glm::vec2>>>(vulkanContext, N);
    auto proj2DZ       = std::make_shared<BufferElement<VulkanBuffer<float>>>(vulkanContext, N);
    auto proj2DBinMask = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, N);
    auto proj2DCovInv  = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(vulkanContext, N);
    auto proj2DColAlpha= std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(vulkanContext, N);
    proj2DPos2D->setName("ProjPos2D");
    proj2DZ->setName("ProjZ");
    proj2DBinMask->setName("ProjBinMask");
    proj2DCovInv->setName("ProjCovInv");
    proj2DColAlpha->setName("ProjColAlpha");

    // Stage 1: projection
    project3Dto2D = vulkanContext.create<GaussianProjection>(
        "shaders/gsplat/gsplat_projection.comp.spv");
    project3Dto2D->setName("GaussianProjection");
    project3Dto2D->setInput(buf3DPos,       0);
    project3Dto2D->setInput(buf3DRot,       1);
    project3Dto2D->setInput(buf3DScale,     2);
    project3Dto2D->setInput(buf3DColAlpha,  3);
    project3Dto2D->setInput(buf3DShR,       4);
    project3Dto2D->setInput(buf3DShG,       5);
    project3Dto2D->setInput(buf3DShB,       6);
    project3Dto2D->setInput(_cameraUBO,     7);
    project3Dto2D->setInput(proj2DPos2D,    8);
    project3Dto2D->setInput(proj2DZ,        9);
    project3Dto2D->setInput(proj2DBinMask,  10);
    project3Dto2D->setInput(proj2DCovInv,   11);
    project3Dto2D->setInput(proj2DColAlpha, 12);
    project3Dto2D->setGroupCountX(N / tpg + 1);
    project3Dto2D->setPushConstants({{N, gridSize, W, H}});

    // Binning buffers
    auto binHistogram = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, numBins * numBinWGs);
    binHistogram->setName("BinHistogram"); binHistogram->setRecordToZero(true);
    auto binOffsets = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, numBinWGs + 1);
    binOffsets->setName("BinOffsets");   binOffsets->setRecordToZero(true);
    auto totalCount = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, 1);
    totalCount->setName("TotalCount"); totalCount->setRecordToZero(true);

    // Stage 2: binning count
    binCount = std::make_shared<GaussianBinningCount>(
        vulkanContext, "shaders/gsplat/gsplat_binning_count.comp.spv");
    binCount->setName("GaussianBinningCount");
    binCount->setInput(project3Dto2D, 0, 8);   // proj2DPos2D
    binCount->setInput(project3Dto2D, 1, 9);   // proj2DZ
    binCount->setInput(project3Dto2D, 2, 11);  // proj2DCovInv
    binCount->setInput(binHistogram,  3);
    binCount->setInput(binOffsets,    4);
    binCount->setGroupCountX(numBinWGs);
    binCount->setPushConstants({{N, gridSize, W, H}});

    // Stage 3: binning prefix sum
    binPrefixSum = std::make_shared<GeneralComputation<>>(
        vulkanContext, "shaders/gsplat/gsplat_binning_prefix_sum.comp.spv");
    binPrefixSum->setName("GaussianBinningPrefixSum");
    binPrefixSum->setInput(binCount, 0, 3);  // binHistogram
    binPrefixSum->setInput(binCount, 1, 4);  // binOffsets
    binPrefixSum->setGroupCountX(numBinWGs);

    // Binned buffers
    auto binPos2D   = std::make_shared<BufferElement<VulkanBuffer<glm::vec2>>>(vulkanContext, maxBinned);
    auto binZ       = std::make_shared<BufferElement<VulkanBuffer<float>>>(vulkanContext, maxBinned);
    auto binBinMask = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, maxBinned);
    auto binCovInv  = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(vulkanContext, maxBinned);
    auto binColAlpha= std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(vulkanContext, maxBinned);
    binPos2D->setName("BinPos2D");    binPos2D->setRecordToZero(false);
    binZ->setName("BinZ");            binZ->setRecordToZero(false);
    binBinMask->setName("BinMask");   binBinMask->setRecordToZero(false);
    binCovInv->setName("BinCovInv");  binCovInv->setRecordToZero(false);
    binColAlpha->setName("BinCA");    binColAlpha->setRecordToZero(false);

    // Stage 4: binning scatter
    BinningScatterPushConstants scatterPC{N, gridSize, W, H, maxBinned};
    binScatter = std::make_shared<GaussianBinningScatter>(
        vulkanContext, "shaders/gsplat/gsplat_binning_scatter.comp.spv");
    binScatter->setName("GaussianBinningScatter");
    binScatter->setInput(project3Dto2D, 0, 8);   // proj2DPos2D
    binScatter->setInput(project3Dto2D, 1, 9);   // proj2DZ
    binScatter->setInput(project3Dto2D, 2, 11);  // proj2DCovInv
    binScatter->setInput(project3Dto2D, 3, 12);  // proj2DColAlpha
    binScatter->setInput(binPrefixSum,  4, 0);   // prefix sums
    binScatter->setInput(totalCount,    5);
    binScatter->setInput(binPos2D,      6);
    binScatter->setInput(binZ,          7);
    binScatter->setInput(binBinMask,    8);
    binScatter->setInput(binCovInv,     9);
    binScatter->setInput(binColAlpha,   10);
    binScatter->setGroupCountX(numBinWGs);
    binScatter->setPushConstants({scatterPC});

    // Sort ping-pong buffers
    sortRadixValA = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, maxBinned);
    sortRadixValA->setName("SortValA");
    sortRadixValB = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, maxBinned);
    sortRadixValB->setName("SortValB");
    auto sortRadixIdxA = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, maxBinned);
    sortRadixIdxA->setName("SortIdxA");
    auto sortRadixIdxB = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, maxBinned);
    sortRadixIdxB->setName("SortIdxB");

    auto scratchHist    = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, numBins * numSortWGs);
    scratchHist->setName("ScratchHist");    scratchHist->setRecordToZero(true);
    auto scratchCounts  = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, numBins);
    scratchCounts->setName("ScratchCounts");scratchCounts->setRecordToZero(true);
    auto scratchOffsets = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, numSortWGs + 1);
    scratchOffsets->setName("ScratchOff"); scratchOffsets->setRecordToZero(true);

    // Stage 5: extract sort keys
    extractSortKeys = std::make_shared<GeneralComputation<>>(
        vulkanContext, "shaders/gsplat/gsplat_extract_sort_keys.comp.spv");
    extractSortKeys->setName("ExtractSortKeys");
    extractSortKeys->setInput(binScatter, 0, 8);  // binnedBinMask
    extractSortKeys->setInput(binScatter, 1, 7);  // binnedZ
    extractSortKeys->setInput(binScatter, 2, 5);  // totalCount
    extractSortKeys->setInput(sortRadixValA, 3);
    extractSortKeys->setInput(sortRadixIdxA, 4);
    const uint32_t numCoverWGs = std::max(1u, std::min(maxBinned / tpg + 1, 14534u));
    extractSortKeys->setGroupCountX(numCoverWGs);

    // Stage 6: radix sort
    sortOp = std::make_shared<RadixSort>(vulkanContext, std::vector<std::string>{
        "shaders/gsplat/gsplat_radix_sort_histogram.comp.spv",
        "shaders/gsplat/gsplat_radix_sort_hist_prefix_sum.comp.spv",
        "shaders/gsplat/gsplat_radix_sort_hist_scatter.comp.spv"
    });
    sortOp->setName("RadixSort");
    sortOp->setInput(extractSortKeys, 0, 3);  // sortRadixValA
    sortOp->setInput(extractSortKeys, 1, 4);  // sortRadixIdxA
    sortOp->setInput(sortRadixValB, 2);
    sortOp->setInput(sortRadixIdxB, 3);
    sortOp->addScratchBufferElement(scratchCounts,  true);
    sortOp->addScratchBufferElement(scratchOffsets, true);
    sortOp->addScratchBufferElement(totalCount,     false);
    sortOp->addScratchBufferElement(scratchHist,    true);
    sortOp->setGroupCountX(numSortWGs);
    {
        std::vector<SortPushConstants> pcs;
        for (uint32_t i = 0; i < 32 / 4; ++i)
            pcs.push_back({i, maxBinned, numBins});
        sortOp->setPushConstants(pcs);
    }

    // Sorted buffers
    auto sortedPos2D   = std::make_shared<BufferElement<VulkanBuffer<glm::vec2>>>(vulkanContext, maxBinned);
    auto sortedZ       = std::make_shared<BufferElement<VulkanBuffer<float>>>(vulkanContext, maxBinned);
    auto sortedBinMask = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, maxBinned);
    auto sortedCovInv  = std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(vulkanContext, maxBinned);
    auto sortedColAlpha= std::make_shared<BufferElement<VulkanBuffer<glm::vec4>>>(vulkanContext, maxBinned);
    sortedPos2D->setName("SortedPos2D");
    sortedZ->setName("SortedZ");
    sortedBinMask->setName("SortedBinMask");
    sortedCovInv->setName("SortedCovInv");
    sortedColAlpha->setName("SortedColAlpha");

    // Stage 7: gather sorted
    gatherSorted = std::make_shared<GeneralComputation<>>(
        vulkanContext, "shaders/gsplat/gsplat_gather_sorted.comp.spv");
    gatherSorted->setName("GatherSorted");
    gatherSorted->setInput(binScatter,  0, 6);   // binnedPos2D
    gatherSorted->setInput(binScatter,  1, 7);   // binnedZ
    gatherSorted->setInput(binScatter,  2, 8);   // binnedBinMask
    gatherSorted->setInput(binScatter,  3, 9);   // binnedCovInv
    gatherSorted->setInput(binScatter,  4, 10);  // binnedColAlpha
    gatherSorted->setInput(sortOp,      5, 1);   // sortRadixIdxA
    gatherSorted->setInput(binScatter,  6, 5);   // totalCount
    gatherSorted->setInput(sortedPos2D,    7);
    gatherSorted->setInput(sortedZ,        8);
    gatherSorted->setInput(sortedBinMask,  9);
    gatherSorted->setInput(sortedCovInv,   10);
    gatherSorted->setInput(sortedColAlpha, 11);
    gatherSorted->setGroupCountX(numCoverWGs);

    // Stage 8: bin bounds
    auto scratchBounds = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, numBins * 2);
    scratchBounds->setName("Bounds"); scratchBounds->setRecordToZero(true);

    computeBounds = std::make_shared<GeneralComputation<>>(
        vulkanContext, "shaders/gsplat/gsplat_bin_bounds.comp.spv");
    computeBounds->setName("GaussianComputeBounds");
    computeBounds->setInput(gatherSorted, 0, 9);  // sortedBinMask
    computeBounds->setInput(binScatter,   1, 5);  // totalCount
    computeBounds->setInput(scratchBounds, 2);
    computeBounds->setGroupCountX(maxBinned / 256 + 1);

    // Stage 9: splatting
    splat = std::make_shared<GaussianSplatting>(
        vulkanContext, "shaders/gsplat/gsplat_binned_splatting.comp.spv");
    splat->setName("GaussianSplatting");
    splat->setInput(gatherSorted,  0, 7);   // sortedPos2D
    splat->setInput(gatherSorted,  1, 10);  // sortedCovInv
    splat->setInput(gatherSorted,  2, 11);  // sortedColAlpha
    splat->setInput(binScatter,    3, 5);   // totalCount
    splat->setInput(computeBounds, 4, 2);   // scratchBounds
    splat->setInput(_imageViewSrc, 5);

    const uint32_t tbX  = 8, tbY = 8;
    const uint32_t gpbX = uint32_t((W / tbX) / gridSize);
    const uint32_t gpbY = uint32_t((H / tbY) / gridSize);
    splat->setGroupCountX(gpbX);
    splat->setGroupCountY(gpbY);
    splat->setGroupCountZ(1);
    {
        std::vector<SplatPushConstants> pcs;
        for (uint32_t y = 0; y < gridSize; y++)
            for (uint32_t x = 0; x < gridSize; x++)
                pcs.push_back({maxBinned, gridSize, x, y, W, H});
        splat->setPushConstants(pcs);
    }

    outputElements[0] = splat;
}

VulkanGaussianSplatting::~VulkanGaussianSplatting() {}

void VulkanGaussianSplatting::checkInput(ComputeGraphElementPtr input, int index) {
    if (index == 0 && !std::dynamic_pointer_cast<ImageViewSrc>(input))
        throw std::runtime_error("VulkanGaussianSplatting: input 0 must be ImageViewSrc");
    if (index == 1 && !std::dynamic_pointer_cast<CameraUboType>(input))
        throw std::runtime_error("VulkanGaussianSplatting: input 1 must be CameraUboType");
    if (index > 1)
        throw std::runtime_error("VulkanGaussianSplatting: input index out of range");
}

void VulkanGaussianSplatting::_setup(VulkanContext& vulkanContext, uint32_t numberPaths) {
    numberOfPaths = numberPaths;

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

    for (uint32_t i = 0; i < numberPaths; ++i) {
        vkCmdFillBuffer(cmd, sortRadixValA->getVkBuffer(i), 0, VK_WHOLE_SIZE, 0xFFFFFFFF);
        vkCmdFillBuffer(cmd, sortRadixValB->getVkBuffer(i), 0, VK_WHOLE_SIZE, 0xFFFFFFFF);
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

void VulkanGaussianSplatting::_record(VkCommandBuffer commandBuffer, uint32_t pathId) {
    vkCmdFillBuffer(commandBuffer, sortRadixValA->getVkBuffer(pathId), 0, VK_WHOLE_SIZE, 0xFFFFFFFF);
    vkCmdFillBuffer(commandBuffer, sortRadixValB->getVkBuffer(pathId), 0, VK_WHOLE_SIZE, 0xFFFFFFFF);
    {
        VkMemoryBarrier mb{};
        mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    auto* ivs = std::dynamic_pointer_cast<ImageViewSrc>(getInputElement(0)).get();
    VkImage image = ivs->getImage(pathId);

    const VkImageLayout finalLayout = vulkanContext->hasSurface()
        ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR : VK_IMAGE_LAYOUT_GENERAL;

    VkImageMemoryBarrier barrier{};
    barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout                       = VK_IMAGE_LAYOUT_GENERAL;
    barrier.newLayout                       = finalLayout;
    barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    barrier.image                           = image;
    barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount     = 1;
    barrier.subresourceRange.layerCount     = 1;
    barrier.srcAccessMask                   = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask                   = VK_ACCESS_MEMORY_READ_BIT;
    vkCmdPipelineBarrier(commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void VulkanGaussianSplatting::loadSPZModel(std::string path) {
    spz::PackedGaussians packed = spz::loadSpzPacked(path);
    gaussians3DData.clear();
    gaussians3DData.reserve(packed.numPoints);
    spz::CoordinateConverter conv;

    for (int i = 0; i < packed.numPoints; i++) {
        spz::UnpackedGaussian ug = packed.unpack(i, conv);
        Gaussian3D g;
        memcpy(&g, &ug, sizeof(spz::UnpackedGaussian));
        g.alpha    = sigmoid(ug.alpha);
        g.scale[0] = std::exp(ug.scale[0]);
        g.scale[1] = std::exp(ug.scale[1]);
        g.scale[2] = std::exp(ug.scale[2]);
        gaussians3DData.push_back(g);
    }
    number_of_gaussians = static_cast<uint32_t>(gaussians3DData.size());
    std::cout << "Loaded " << number_of_gaussians << " gaussians from " << path << "\n";
}

} // namespace klartraum
