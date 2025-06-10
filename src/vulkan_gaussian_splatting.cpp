#include <glm/glm.hpp>
#include <array>
#include <stdexcept>

#include "load-spz.h"

#include "klartraum/vulkan_gaussian_splatting.hpp"
#include "klartraum/vulkan_helpers.hpp"
#include "klartraum/drawgraph/imageviewsrc.hpp"

namespace klartraum {

VulkanGaussianSplatting::VulkanGaussianSplatting(
    VulkanKernel& vulkanKernel,
    std::shared_ptr<ImageViewSrc> _imageViewSrc,
    std::shared_ptr<CameraUboType> _cameraUBO,
    std::string path
) {
    loadSPZModel(path);

    this->vulkanKernel = &vulkanKernel;
    this->setInput(_imageViewSrc, 0);
    this->setInput(_cameraUBO, 1);

    if(inputs.size() == 0) {
        throw std::runtime_error("no input!");
    }
    std::shared_ptr<ImageViewSrc> imageViewSrc = std::dynamic_pointer_cast<ImageViewSrc>(getInputElement(0));
    if (imageViewSrc == nullptr) {
        throw std::runtime_error("input is not an ImageViewSrc!");
    }

    gaussians3D = std::make_shared<BufferElementSinglePath<Gaussian3DBuffer>>(vulkanKernel, number_of_gaussians);
    gaussians3D->setName("Gaussians3D");

    gaussians3D->getBuffer().memcopyFrom(gaussians3DData);

    // setup projection stage
    /////////////////////////////////////////////

    ProjectionPushConstants pushConstants = {
        number_of_gaussians,   // numElements
        4,                              // gridSize (4x4)
        512.0f,                         // screenWidth
        512.0f                          // screenHeight
    };

    project3Dto2D = std::make_unique<GaussianProjection>(vulkanKernel, "shaders/gaussian_splatting_projection.comp.spv");
    project3Dto2D->setName("GaussianProjection");
    project3Dto2D->setInput(gaussians3D);
    project3Dto2D->setGroupCountX(number_of_gaussians / 128 + 1);
    project3Dto2D->setPushConstants({pushConstants});
    project3Dto2D->setCustomOutputSize(number_of_gaussians * 2);
    project3Dto2D->setUbo(std::dynamic_pointer_cast<CameraUboType>(getInputElement(1)));

    // setup binning stage
    /////////////////////////////////////////////

    auto totalGaussian2DCounts = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanKernel, 1);
    totalGaussian2DCounts->zero();
    totalGaussian2DCounts->setName("TotalGaussian2DCounts");

    auto binnedGaussians2D = std::make_shared<BufferElement<Gaussian2DBuffer>>(vulkanKernel, number_of_gaussians * 2);
    binnedGaussians2D->zero();
    binnedGaussians2D->setName("BinnedGaussians2D");

    bin = std::make_shared<GaussianBinning>(vulkanKernel, "shaders/gaussian_splatting_binning.comp.spv");
    bin->setName("GaussianBinning");
    bin->setInput(project3Dto2D, 0);
    bin->setInput(binnedGaussians2D, 1);
    bin->setInput(totalGaussian2DCounts, 2);
    bin->setGroupCountX((number_of_gaussians*2) / 128 + 1);
    bin->setPushConstants({pushConstants});

    // setup sorting stage
    /////////////////////////////////////////////
    sort2DGaussians = std::make_shared<GaussianSort>(vulkanKernel, "shaders/gaussian_splatting_radix_sort.comp.spv");
    sort2DGaussians->setName("GaussianSort");

    sort2DGaussians->setInput(bin, 0, 1);

    auto scratchBufferCounts = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanKernel, 16);
    scratchBufferCounts->setName("ScratchBufferCounts");
    auto scratchBufferOffsets = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanKernel, 16);
    scratchBufferOffsets->setName("ScratchBufferOffsets");

    sort2DGaussians->addScratchBufferElement(scratchBufferCounts);
    sort2DGaussians->addScratchBufferElement(scratchBufferOffsets);
    sort2DGaussians->addScratchBufferElement(totalGaussian2DCounts);

    sort2DGaussians->setGroupCountX((number_of_gaussians*2) / 128 * 2 + 1);


    uint32_t numElements = (uint32_t)((number_of_gaussians));
    uint32_t numBitsPerPass = 4; // Number of bits per pass (4 bits for 16 bins)
    uint32_t numBins = 16; // Number of bins for sorting = 2 ^ numBitsPerPass
    uint32_t passes = 32 + 16; // 32 bits for depth, 16 bits for binning
    std::vector<SortPushConstants> sortPushConstants;
    for (uint32_t i = 0; i < passes / numBitsPerPass; i++) {
        sortPushConstants.push_back({i, numElements, numBins}); // pass, numElements, numBins
    }

    sort2DGaussians->setPushConstants(sortPushConstants);

    // setup bounds computation stage
    /////////////////////////////////////////////
    auto scratchBinStartAndEnd = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanKernel, 16*2);
    scratchBinStartAndEnd->setName("ScratchBinStartAndEnd");
    scratchBinStartAndEnd->zero();

    computeBounds = std::make_shared<GaussianComputeBounds>(vulkanKernel, "shaders/gaussian_splatting_bin_bounds.comp.spv");
    computeBounds->setName("GaussianComputeBounds");

    ProjectionPushConstants computeBoundsPushConstants = {
        (uint32_t)((number_of_gaussians)),   // numElements
        4,                              // gridSize (4x4)
        512.0f,                         // screenWidth
        512.0f                          // screenHeight
    };

    computeBounds->setInput(sort2DGaussians, 0, 0); //bufferElement, 0);
    computeBounds->setInput(bin, 1, 2); //totalGaussian2DCounts, 1);
    computeBounds->setInput(scratchBinStartAndEnd, 2 ); //scratchBinStartAndEnd, 2);
    computeBounds->setGroupCountX((number_of_gaussians*2) / 256 + 1);

    computeBounds->setPushConstants({computeBoundsPushConstants});


    // setup splatting stage
    /////////////////////////////////////////////
    splat = std::make_shared<GaussianSplatting>(vulkanKernel, "shaders/gaussian_splatting_binned_splatting.comp.spv");
    splat->setName("GaussianSplatting");

    std::vector<SplatPushConstants> splatPushConstants;
    for(uint32_t y = 0; y < 4; y++) {
        for(uint32_t x = 0; x < 4; x++) {
            splatPushConstants.push_back({
                (uint32_t)(number_of_gaussians*2),   // numElements
                4,                              // gridSize (4x4)
                x,                              // gridX
                y,                              // gridY
                512.0f,                         // screenWidth
                512.0f                          // screenHeight
            });
        }
    }
    
    splat->setInput(computeBounds, 0, 0); // bufferElement, 0);
    splat->setInput(computeBounds, 1, 1); // totalGaussian2DCounts, 1);
    splat->setInput(computeBounds, 2, 2); // scratchBinStartAndEnd, 2);
    splat->setInput(imageViewSrc, 3); 

    uint32_t groupsPerBin = (512 / 16) / 4; // = 8 groups per bin with 16 threads each

    splat->setGroupCountX(groupsPerBin);
    splat->setGroupCountY(groupsPerBin);
    splat->setGroupCountZ(1);

    splat->setPushConstants(splatPushConstants);

    // this is the last element in the splatting pipeline
    // it will be used as the output of the drawgraphgroup
    // so that the drawgraph compilation traversal can
    // traverse from this element back through all the elements of the gaussian splatting pipeline
    outputElements[0] = splat;    
}

VulkanGaussianSplatting::~VulkanGaussianSplatting() {
    if(vulkanKernel != nullptr)
    {
    }

}

void VulkanGaussianSplatting::checkInput(DrawGraphElementPtr input, int index) {
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

void VulkanGaussianSplatting::_setup(VulkanKernel& vulkanKernel, uint32_t numberPaths)
{
    numberOfPaths = numberPaths;

}

void VulkanGaussianSplatting::_record(VkCommandBuffer commandBuffer, uint32_t pathId)
{
    auto& device = vulkanKernel->getDevice();
    auto& swapChain = vulkanKernel->getSwapChain();
    auto& graphicsQueue = vulkanKernel->getGraphicsQueue();


    auto& swapChainExtent = vulkanKernel->getSwapChainExtent();

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

    VkImageMemoryBarrier barrierBack = {};
    barrierBack.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrierBack.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrierBack.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
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


void VulkanGaussianSplatting::loadSPZModel(std::string path)
{
    
    spz::PackedGaussians packed = spz::loadSpzPacked(path);
    
    gaussians3DData.clear();
    gaussians3DData.reserve(packed.numPoints);
    for (int i = 70000; i < 70020; /*packed.numPoints*/ i++) {
    //for (int i = 0; i < packed.numPoints; i++) {
        spz::UnpackedGaussian gaussian = packed.unpack(i);
        Gaussian3D gaussian3D;
        memcpy(&gaussian3D, &gaussian, sizeof(spz::UnpackedGaussian));
        gaussians3DData.push_back(gaussian3D);
    }

    // number_of_gaussians = (uint32_t)packed.numPoints;
    number_of_gaussians = 20;


}

} // namespace klartraum
