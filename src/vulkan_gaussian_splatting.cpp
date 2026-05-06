#include <array>
#include <glm/glm.hpp>
#include <stdexcept>
#include <filesystem>

#include "load-spz.h"

#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/vulkan_gaussian_splatting.hpp"
#include "klartraum/vulkan_helpers.hpp"

namespace klartraum {

VulkanGaussianSplatting::VulkanGaussianSplatting(
    VulkanContext& vulkanContext,
    std::shared_ptr<ImageViewSrc> _imageViewSrc,
    std::shared_ptr<CameraUboType> _cameraUBO,
    std::string path) {
    loadSPZModel(path);


    const uint32_t gridSize = 4; // 4x4 grid for binning
    const uint32_t numBins = gridSize * gridSize; // number of bins in the grid
    const uint32_t threadsPerGroup = 128; // number of threads per workgroup
    const uint32_t maxGaussiansModifier = 2; // arbitrary number, currently 2x the number of initial 3D gaussians

    assert(number_of_gaussians > 0, "number_of_gaussians must be greater than 0");

    this->vulkanContext = &vulkanContext;
    this->setInput(_imageViewSrc, 0);
    this->setInput(_cameraUBO, 1);

    if (inputs.size() == 0) {
        throw std::runtime_error("no input!");
    }
    std::shared_ptr<ImageViewSrc> imageViewSrc = std::dynamic_pointer_cast<ImageViewSrc>(getInputElement(0));
    if (imageViewSrc == nullptr) {
        throw std::runtime_error("input is not an ImageViewSrc!");
    }

    VkExtent2D imageExtent = imageViewSrc->getImageExtent(0);
    if (imageExtent.width == 0 || imageExtent.height == 0) {
        throw std::runtime_error("ImageViewSrc has invalid image extent!");
    }

    const float screenWidth = static_cast<float>(imageExtent.width);
    const float screenHeight = static_cast<float>(imageExtent.height);

    gaussians3D = std::make_shared<BufferElementSinglePath<Gaussian3DBuffer>>(vulkanContext, number_of_gaussians);
    gaussians3D->setName("Gaussians3D");

    gaussians3D->getBuffer().memcopyFrom(gaussians3DData);

    gaussians2D = std::make_shared<BufferElement<Gaussian2DBuffer>>(vulkanContext, number_of_gaussians * maxGaussiansModifier);
    gaussians2D->setName("Gaussians2D");

    // setup projection stage
    /////////////////////////////////////////////

    ProjectionPushConstants pushConstants = {
        number_of_gaussians, // numElements
        gridSize,             // gridSize (4x4)
        screenWidth,         // screenWidth
        screenHeight         // screenHeight
    };

    project3Dto2D = vulkanContext.create<GaussianProjection>("shaders/gsplat/gsplat_projection.comp.spv");
    project3Dto2D->setName("GaussianProjection");
    project3Dto2D->setInput(gaussians3D, 0);
    project3Dto2D->setInput(_cameraUBO, 1);
    project3Dto2D->setInput(gaussians2D, 2);
    project3Dto2D->setGroupCountX(number_of_gaussians / threadsPerGroup + 1);
    project3Dto2D->setPushConstants({pushConstants});

    // setup binning stage — three-pass deterministic prefix-sum scatter
    /////////////////////////////////////////////
    // Pass 1 (count): per-workgroup histogram of (gaussian,bin) overlaps
    // Pass 2 (prefix sum): exclusive prefix sums → global write offsets
    // Pass 3 (scatter): each gaussian writes to a deterministic position
    // using a Hillis-Steele scan within each workgroup, eliminating the
    // non-deterministic atomicAdd of the old single-pass approach.

    auto totalGaussian2DCounts = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, 1);
    totalGaussian2DCounts->setRecordToZero(true);
    totalGaussian2DCounts->setName("TotalGaussian2DCounts");

    auto binnedGaussians2D = vulkanContext.create<BufferElement<Gaussian2DBuffer>>(number_of_gaussians * maxGaussiansModifier);
    binnedGaussians2D->setRecordToZero(false);
    binnedGaussians2D->setName("BinnedGaussians2D");

    const uint32_t maxBinnedGaussians = number_of_gaussians * maxGaussiansModifier;
    const uint32_t numBinWorkGroups   = maxBinnedGaussians / threadsPerGroup + 1;

    auto binHistogram = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, numBins * numBinWorkGroups);
    binHistogram->setName("BinHistogram"); binHistogram->setRecordToZero(true);
    auto binOffsets = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, numBinWorkGroups + 1);
    binOffsets->setName("BinOffsets"); binOffsets->setRecordToZero(true);

    // Pass 1: count
    binCount = std::make_shared<GaussianBinningCount>(vulkanContext, "shaders/gsplat/gsplat_binning_count.comp.spv");
    binCount->setName("GaussianBinningCount");
    binCount->setInput(project3Dto2D, 0, 2);  // projected Gaussian2D (slot 2)
    binCount->setInput(binHistogram, 1);
    binCount->setInput(binOffsets,   2);
    binCount->setGroupCountX(numBinWorkGroups);
    binCount->setPushConstants({pushConstants});

    // Pass 2: prefix sum (reuses the chained-scan algorithm from the radix sort)
    binPrefixSum = std::make_shared<GeneralComputation<>>(vulkanContext, "shaders/gsplat/gsplat_binning_prefix_sum.comp.spv");
    binPrefixSum->setName("GaussianBinningPrefixSum");
    binPrefixSum->setInput(binCount, 0, 1);  // binHistogram (slot 1 of count stage)
    binPrefixSum->setInput(binCount, 1, 2);  // binOffsets   (slot 2 of count stage)
    binPrefixSum->setGroupCountX(numBinWorkGroups);  // one workgroup per histogram column

    // Pass 3: scatter
    BinningScatterPushConstants scatterPC{
        number_of_gaussians,
        gridSize,
        screenWidth,
        screenHeight,
        maxBinnedGaussians
    };
    binScatter = std::make_shared<GaussianBinningScatter>(vulkanContext, "shaders/gsplat/gsplat_binning_scatter.comp.spv");
    binScatter->setName("GaussianBinningScatter");
    binScatter->setInput(project3Dto2D, 0, 2);       // projected gaussians (slot 2)
    binScatter->setInput(binnedGaussians2D,  1);      // output buffer
    binScatter->setInput(binPrefixSum, 2, 0);         // prefix sums (slot 0 = binHistogram after pass 2)
    binScatter->setInput(totalGaussian2DCounts, 3);   // output totalCount
    binScatter->setGroupCountX(numBinWorkGroups);
    binScatter->setPushConstants({scatterPC});

    // setup sorting stage — extract → radix sort → gather
    /////////////////////////////////////////////

    const uint32_t maxBinned         = number_of_gaussians * maxGaussiansModifier;
    const uint32_t numSortWorkGroups  = maxBinned / threadsPerGroup + 1;

    // Ping-pong value/index buffers for the radix sort.
    // sortRadixValA and sortRadixValB are stored as members so _record can
    // pre-fill them with 0xFFFFFFFF each frame.  Elements beyond totalCount
    // retain that sentinel value and sort to the end of the output, keeping
    // the first totalCount positions clean for computeBounds and splatting.
    sortRadixValA = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, maxBinned);
    sortRadixValA->setName("SortRadixValA");
    sortRadixValB = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, maxBinned);
    sortRadixValB->setName("SortRadixValB");
    auto sortRadixIdxA = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, maxBinned);
    sortRadixIdxA->setName("SortRadixIdxA");
    auto sortRadixIdxB = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, maxBinned);
    sortRadixIdxB->setName("SortRadixIdxB");

    auto scratchHistograms = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, numBins * numSortWorkGroups);
    scratchHistograms->setName("ScratchHistograms"); scratchHistograms->setRecordToZero(true);
    auto scratchCounts  = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, numBins);
    scratchCounts->setName("ScratchCounts");  scratchCounts->setRecordToZero(true);
    auto scratchOffsets = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, numSortWorkGroups + 1);
    scratchOffsets->setName("ScratchOffsets"); scratchOffsets->setRecordToZero(true);

    // Stage: extract (bin+z) sort keys from binnedGaussians2D → sortRadixValA
    extractSortKeys = std::make_shared<GeneralComputation<>>(
        vulkanContext, "shaders/gsplat/gsplat_extract_sort_keys.comp.spv");
    extractSortKeys->setName("ExtractSortKeys");
    extractSortKeys->setInput(binScatter, 0, 1);    // binnedGaussians2D  (slot 1 of scatter)
    extractSortKeys->setInput(binScatter, 1, 3);    // totalGaussian2DCounts (slot 3 of scatter)
    extractSortKeys->setInput(sortRadixValA, 2);
    extractSortKeys->setInput(sortRadixIdxA, 3);
    extractSortKeys->setGroupCountX(numSortWorkGroups);  // fixed; shader returns early beyond totalCount

    // Stage: radix sort (8 passes × 4 bits = 32 bits, last pass=7 odd → output in A buffers)
    std::vector<std::string> sortShaders = {
        "shaders/gsplat/gsplat_radix_sort_histogram.comp.spv",
        "shaders/gsplat/gsplat_radix_sort_hist_prefix_sum.comp.spv",
        "shaders/gsplat/gsplat_radix_sort_hist_scatter.comp.spv"
    };
    sortOp = std::make_shared<RadixSort>(vulkanContext, sortShaders);
    sortOp->setName("RadixSort");
    sortOp->setInput(extractSortKeys, 0, 2);   // sortRadixValA
    sortOp->setInput(extractSortKeys, 1, 3);   // sortRadixIdxA
    sortOp->setInput(sortRadixValB, 2);
    sortOp->setInput(sortRadixIdxB, 3);
    sortOp->addScratchBufferElement(scratchCounts,      true);
    sortOp->addScratchBufferElement(scratchOffsets,     true);
    sortOp->addScratchBufferElement(totalGaussian2DCounts, false);
    sortOp->addScratchBufferElement(scratchHistograms,  true);
    sortOp->setGroupCountX(numSortWorkGroups);
    {
        std::vector<SortPushConstants> pcs;
        for (uint32_t i = 0; i < 32 / 4; ++i)
            pcs.push_back({i, maxBinned, numBins});
        sortOp->setPushConstants(pcs);
    }

    // Stage: gather binnedGaussians2D in sorted order → sortedGaussians2D
    auto sortedGaussians2D = std::make_shared<BufferElement<Gaussian2DBuffer>>(vulkanContext, maxBinned);
    sortedGaussians2D->setName("SortedGaussians2D");

    gatherSorted = std::make_shared<GeneralComputation<>>(
        vulkanContext, "shaders/gsplat/gsplat_gather_sorted.comp.spv");
    gatherSorted->setName("GatherSorted");
    gatherSorted->setInput(binScatter, 0, 1);   // binnedGaussians2D (slot 1 of scatter)
    gatherSorted->setInput(sortOp, 1, 1);       // sortRadixIdxA (sorted indices)
    gatherSorted->setInput(binScatter, 2, 3);   // totalGaussian2DCounts (slot 3 of scatter)
    gatherSorted->setInput(sortedGaussians2D, 3);
    gatherSorted->setGroupCountX(numSortWorkGroups);

    // setup bounds computation stage
    /////////////////////////////////////////////
    auto scratchBinStartAndEnd = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanContext, numBins * 2);
    scratchBinStartAndEnd->setName("ScratchBinStartAndEnd");
    scratchBinStartAndEnd->setRecordToZero(true);

    computeBounds = std::make_shared<GaussianComputeBounds>(vulkanContext, "shaders/gsplat/gsplat_bin_bounds.comp.spv");
    computeBounds->setName("GaussianComputeBounds");

    ProjectionPushConstants computeBoundsPushConstants = {
        maxBinned,    // numElements (upper bound; shader uses totalCount from buffer)
        gridSize,
        screenWidth,
        screenHeight
    };

    computeBounds->setInput(gatherSorted, 0, 3);       // sortedGaussians2D
    computeBounds->setInput(binScatter, 1, 3);         // totalGaussian2DCounts (slot 3 of scatter)
    computeBounds->setInput(scratchBinStartAndEnd, 2);
    computeBounds->setGroupCountX(maxBinnedGaussians / 256 + 1);

    computeBounds->setPushConstants({computeBoundsPushConstants});

    // setup splatting stage
    /////////////////////////////////////////////
    splat = std::make_shared<GaussianSplatting>(vulkanContext, "shaders/gsplat/gsplat_binned_splatting.comp.spv");
    splat->setName("GaussianSplatting");

    std::vector<SplatPushConstants> splatPushConstants;
    for (uint32_t y = 0; y < gridSize; y++) {
        for (uint32_t x = 0; x < gridSize; x++) {
            splatPushConstants.push_back({
                (uint32_t)(number_of_gaussians * maxGaussiansModifier), // max. numElements
                gridSize,                             // gridSize (4x4)
                x,                                    // gridX
                y,                                    // gridY
                screenWidth,                          // screenWidth
                screenHeight                           // screenHeight
            });
        }
    }

    splat->setInput(computeBounds, 0, 0);       // sortedGaussians2D
    splat->setInput(binScatter, 1, 3);           // totalGaussian2DCounts (slot 3 of scatter)
    splat->setInput(computeBounds, 2, 2);        // scratchBinStartAndEnd
    splat->setInput(imageViewSrc, 3);


    // each bin computes several workgroups, each processing 8x8 pixels
    // where each pixel is processed by a single thread
    const uint32_t threadsPerBinX = 8;
    const uint32_t threadsPerBinY = 8;

    const uint32_t groupsPerBinX = uint32_t((screenWidth / threadsPerBinX) / gridSize);
    const uint32_t groupsPerBinY = uint32_t((screenHeight / threadsPerBinY) / gridSize);

    splat->setGroupCountX(groupsPerBinX);
    splat->setGroupCountY(groupsPerBinY);
    splat->setGroupCountZ(1);

    splat->setPushConstants(splatPushConstants);

    // this is the last element in the splatting pipeline
    // it will be used as the output of the computegraphgroup
    // so that the computegraph compilation traversal can
    // traverse from this element back through all the elements of the gaussian splatting pipeline
    outputElements[0] = splat;
}

VulkanGaussianSplatting::~VulkanGaussianSplatting() {
    if (vulkanContext != nullptr) {
    }
}

void VulkanGaussianSplatting::checkInput(ComputeGraphElementPtr input, int index) {
    ImageViewSrc* imageViewSrc = std::dynamic_pointer_cast<ImageViewSrc>(input).get();
    if (index == 0 && imageViewSrc == nullptr) {
        throw std::runtime_error("input is not an ImageViewSrc!");
    }
    CameraUboType* cameraUbo = std::dynamic_pointer_cast<CameraUboType>(input).get();
    if (index == 1 && cameraUbo == nullptr) {
        throw std::runtime_error("input is not a CameraUboType!");
    }
    if (index > 1) {
        throw std::runtime_error("input index out of range!");
    }
}

void VulkanGaussianSplatting::_setup(VulkanContext& vulkanContext, uint32_t numberPaths) {
    numberOfPaths = numberPaths;

    // The _record pre-fill (0xFFFFFFFF) only takes effect from the SECOND use
    // of each path because it runs AFTER the sort.  Pre-fill all paths here
    // during setup so that even the very first frame of each path has correct
    // overflow sentinel values in the sort buffers.
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
    auto& device = vulkanContext->getDevice();
    auto& swapChain = vulkanContext->getSwapChain();
    auto& graphicsQueue = vulkanContext->getGraphicsQueue();

    auto& swapChainExtent = vulkanContext->getSwapChainExtent();

    ImageViewSrc* imageViewSrc = std::dynamic_pointer_cast<ImageViewSrc>(getInputElement(0)).get();
    if (imageViewSrc == nullptr) {
        throw std::runtime_error("input is not an ImageViewSrc!");
    }
    VkImage image = imageViewSrc->getImage(pathId);

    //     // Ensure compute writes are visible to graphics
    //     VkImageMemoryBarrier imageBarrier = {};
    //     imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    //     imageBarrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT;  // Graphics writes
    //     imageBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT; // Compute reads/writes
    //     imageBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; //VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;  // Layout used by graphics rendering
    //     imageBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;  // Layout used by compute shader
    //     imageBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;  // Assume single queue
    //     imageBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    //     // TODO : use the correct image

    //     // imageBarrier.image = image;  // The image used as framebuffer and compute input
    //     imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    //     imageBarrier.subresourceRange.baseMipLevel = 0;
    //     imageBarrier.subresourceRange.levelCount = 1;
    //     imageBarrier.subresourceRange.baseArrayLayer = 0;
    //     imageBarrier.subresourceRange.layerCount = 1;

    //     vkCmdPipelineBarrier(
    //         commandBuffer,
    //         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
    //         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    //         0,  // No dependency flags
    //         0, nullptr,  // No global memory barriers
    //         0, nullptr,  // No buffer memory barriers
    //         1, &imageBarrier // Image memory barrier
    //     );

    // /*
    //     // issue the compute pipeline for gaussian splatting
    //     vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);

    //     auto& cameraUBO = this->getCameraUBO();
    //     auto& descriptorSets = cameraUBO->getDescriptorSets();
    //     std::array<VkDescriptorSet, 2> combinedDescriptorSets = {computeDescriptorSets[pathId], descriptorSets[pathId]};
    //     vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 2, combinedDescriptorSets.data(), 0, 0);

    //     uint32_t num_groups_z = number_of_gaussians / 16;

    //     vkCmdDispatch(commandBuffer, 64, 64, num_groups_z);*/

    // Pre-fill sort value buffers with 0xFFFFFFFF for the NEXT frame's radix
    // sort pass.  The extract stage only writes elements 0..totalCount-1; any
    // remaining elements retain this sentinel and sort to the very end of the
    // output (0xFFFFFFFF > any valid binMask 0..0x8000), so computeBounds and
    // splatting see only the valid sorted gaussians in positions 0..totalCount-1.
    vkCmdFillBuffer(commandBuffer, sortRadixValA->getVkBuffer(pathId), 0, VK_WHOLE_SIZE, 0xFFFFFFFF);
    vkCmdFillBuffer(commandBuffer, sortRadixValB->getVkBuffer(pathId), 0, VK_WHOLE_SIZE, 0xFFFFFFFF);
    {
        VkMemoryBarrier mb{};
        mb.sType          = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask  = VK_ACCESS_TRANSFER_WRITE_BIT;
        mb.dstAccessMask  = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(commandBuffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &mb, 0, nullptr, 0, nullptr);
    }

    // When rendering to a real swapchain the image must be in PRESENT_SRC_KHR
    // for vkQueuePresentKHR; in headless mode GENERAL is sufficient.
    const VkImageLayout finalLayout = vulkanContext->hasSurface()
        ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
        : VK_IMAGE_LAYOUT_GENERAL;

    VkImageMemoryBarrier barrierBack = {};
    barrierBack.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrierBack.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrierBack.newLayout = finalLayout;
    barrierBack.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrierBack.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrierBack.image = image;
    barrierBack.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrierBack.subresourceRange.baseMipLevel = 0;
    barrierBack.subresourceRange.levelCount = 1;
    barrierBack.subresourceRange.baseArrayLayer = 0;
    barrierBack.subresourceRange.layerCount = 1;
    barrierBack.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrierBack.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;

    vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrierBack);
}

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

void VulkanGaussianSplatting::loadSPZModel(std::string path) {

    spz::PackedGaussians packed = spz::loadSpzPacked(path);

    gaussians3DData.clear();
    gaussians3DData.reserve(packed.numPoints);
    number_of_gaussians = 256 * 256;

    // float clipBoundsX = 300.0f;
    // float clipBoundsY = 300.0f;
    // float clipBoundsZ = 300.0f;

    spz::CoordinateConverter defaultCoordinateConverter;

    for (int i = 0; i < packed.numPoints; i++) {
    //for (int i = 60000; i < 60000 + number_of_gaussians; /*packed.numPoints*/ i++) {
        spz::UnpackedGaussian gaussian = packed.unpack(i, defaultCoordinateConverter);
        // if (gaussian.position[0] < -clipBoundsX || gaussian.position[0] > clipBoundsX ||
        //     gaussian.position[1] < -clipBoundsY || gaussian.position[1] > clipBoundsY ||
        //     gaussian.position[2] < -clipBoundsZ || gaussian.position[2] > clipBoundsZ) {
        //     continue; // Skip gaussians outside the clipping bounds
        // }
        Gaussian3D gaussian3D;
        memcpy(&gaussian3D, &gaussian, sizeof(spz::UnpackedGaussian));

        // use activation functions as done in original implementation and described in the paper
        gaussian3D.alpha = sigmoid(gaussian.alpha); // inverse logistic back to alpha

        // color is sh0 encoding, if we want to skip the spherical harmonics, we can use the following:
        // gaussian3D.color[0] = 0.5 + 0.282095 * gaussian.color[0];
        // gaussian3D.color[1] = 0.5 + 0.282095 * gaussian.color[1];
        // gaussian3D.color[2] = 0.5 + 0.282095 * gaussian.color[2];


        gaussian3D.scale[0] = std::exp(gaussian.scale[0]);
        gaussian3D.scale[1] = std::exp(gaussian.scale[1]);
        gaussian3D.scale[2] = std::exp(gaussian.scale[2]);
        gaussians3DData.push_back(gaussian3D);
    }

    number_of_gaussians = (uint32_t)gaussians3DData.size();

    std::cout << "Loaded " << number_of_gaussians << " gaussians from SPZ file: " << path << std::endl;
}

void VulkanGaussianSplatting::loadPLYModel(std::string path) {

    spz::UnpackOptions unpackOptions;
    spz::GaussianCloud cloud = spz::loadSplatFromPly("input/bonsai/point_cloud/iteration_7000/point_cloud.ply", unpackOptions);

    gaussians3DData.clear();
    gaussians3DData.reserve(cloud.numPoints);
    number_of_gaussians = 256 * 256;

    float clipBounds = 1.5f;

    spz::CoordinateConverter defaultCoordinateConverter;

    for (int i = 0; i < cloud.numPoints; i++) {
    //for (int i = 60000; i < 60000 + number_of_gaussians; /*packed.numPoints*/ i++) {
        Gaussian3D gaussian3D;
        // use activation functions as done in original implementation and described in the paper
        // position
        gaussian3D.position[0] = cloud.positions[i * 3 + 0];
        gaussian3D.position[1] = cloud.positions[i * 3 + 1];
        gaussian3D.position[2] = cloud.positions[i * 3 + 2];

        // rotation
        gaussian3D.rotation[0] = cloud.rotations[i * 4 + 0];
        gaussian3D.rotation[1] = cloud.rotations[i * 4 + 1];
        gaussian3D.rotation[2] = cloud.rotations[i * 4 + 2];
        gaussian3D.rotation[3] = cloud.rotations[i * 4 + 3];

        // alpha, color, scale
        gaussian3D.alpha = sigmoid(cloud.alphas[i]); // inverse logistic back to alpha
        gaussian3D.color[0] = 0.5f + 0.282095f * cloud.colors[i*3+0];
        gaussian3D.color[1] = 0.5f + 0.282095f * cloud.colors[i*3+1];
        gaussian3D.color[2] = 0.5f + 0.282095f * cloud.colors[i*3+2];
        
        gaussian3D.scale[0] = std::exp(cloud.scales[i*3+0]);
        gaussian3D.scale[1] = std::exp(cloud.scales[i*3+1]);
        gaussian3D.scale[2] = std::exp(cloud.scales[i*3+2]);


        if (cloud.positions[i*3+0] < -clipBounds || cloud.positions[i*3+0] > clipBounds ||
            cloud.positions[i*3+1] < -clipBounds || cloud.positions[i*3+1] > clipBounds ||
            cloud.positions[i*3+2] < -clipBounds || cloud.positions[i*3+2] > clipBounds) {
            continue; // Skip gaussians outside the clipping bounds
        }
        gaussians3DData.push_back(gaussian3D);
    }

    number_of_gaussians = (uint32_t)gaussians3DData.size();

    std::cout << "Loaded " << number_of_gaussians << " gaussians from PLY file: " << path << std::endl;
}
} // namespace klartraum
