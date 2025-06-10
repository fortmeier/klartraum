#include <gtest/gtest.h>

#include "klartraum/glfw_frontend.hpp"
#include "klartraum/vulkan_gaussian_splatting.hpp"
#include "klartraum/drawgraph/imageviewsrc.hpp"
#include "klartraum/interface_camera_orbit.hpp"

using namespace klartraum;

TEST(KlartraumVulkanGaussianSplatting, smoke) {
    GlfwFrontend frontend;

    auto& core = frontend.getKlartraumCore();
    auto& vulkanKernel = core.getVulkanKernel();
    auto& device = vulkanKernel.getDevice();
    
    std::vector<VkImageView> imageViews;
    std::vector<VkImage> images;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    for (int i = 0; i < 3; i++) {
        imageViews.push_back(vulkanKernel.getImageView(i));
        images.push_back(vulkanKernel.getSwapChainImage(i));
        imageAvailableSemaphores.push_back(vulkanKernel.imageAvailableSemaphoresPerImage[i]);        
        
    }

    auto imageViewSrc = std::make_shared<ImageViewSrc>(imageViews, images);
    imageViewSrc->setWaitFor(0, imageAvailableSemaphores[0]);
    imageViewSrc->setWaitFor(1, imageAvailableSemaphores[1]);
    imageViewSrc->setWaitFor(2, imageAvailableSemaphores[2]);    

    /*
    STEP 1: create the drawgraph elements
    */
   
    std::shared_ptr<CameraUboType> cameraUBO = std::make_shared<CameraUboType>();
    cameraUBO->ubo.proj = glm::perspective(glm::radians(45.0f), (float)BackendConfig::WIDTH / (float)BackendConfig::HEIGHT, 0.1f, 100.0f);
   
    std::string spzFile = "data/hornedlizard.spz";
    std::shared_ptr<VulkanGaussianSplatting> splatting = std::make_shared<VulkanGaussianSplatting>(vulkanKernel, imageViewSrc, cameraUBO, spzFile);

    /*
    STEP 2: create the drawgraph backend and compile the drawgraph
    */
    auto& drawgraph = DrawGraph(vulkanKernel, 1);
    drawgraph.compileFrom(splatting);

    /*
    STEP 3: submit the drawgraph and compare the output
    */
    VkSemaphore finishSemaphore = VK_NULL_HANDLE;   
    for(int i = 0; i < 1; i++) {
        auto [imageIndex, imageAvailableSemaphore] = vulkanKernel.beginRender();
        finishSemaphore = drawgraph.submitTo(vulkanKernel.getGraphicsQueue(), imageIndex);
        vulkanKernel.endRender(imageIndex, finishSemaphore);
    }

    std::this_thread::sleep_for(std::chrono::seconds(1));
    vkQueueWaitIdle(vulkanKernel.getGraphicsQueue());
    return;
    
}

TEST(KlartraumVulkanGaussianSplatting, project) {
    GlfwFrontend frontend;

    auto& core = frontend.getKlartraumCore();
    auto& vulkanKernel = core.getVulkanKernel();
    auto& device = vulkanKernel.getDevice();

    std::vector<Gaussian3D> gaussians3D;
    gaussians3D.push_back(Gaussian3D{
        {0.0f, 0.0f, 0.0f}, // position
        {0.0f, 0.0f, 0.0f, 1.0f}, // rotation
        {1.0f, 1.0f, 1.0f}, // scale
        {1.0f, 1.0f, 1.0f}, // color
        1.0f, // alpha
        {1.01f}, // shR
        {1.02f}, // shG
        {1.03f}  // shB
    });
    gaussians3D.push_back(Gaussian3D{
        {1.0f, 0.0f, 0.0f}, // position
        {0.0f, 0.0f, 0.0f, 1.0f}, // rotation
        {1.0f, 1.0f, 1.0f}, // scale
        {1.0f, 0.0f, 0.0f}, // color
        1.0f, // alpha
        {1.0f}, // shR
        {1.0f}, // shG
        {1.0f}  // shB
    });
    gaussians3D.push_back(Gaussian3D{
        {0.0f, 1.0f, 0.0f}, // position
        {0.0f, 0.0f, 0.0f, 1.0f}, // rotation
        {1.0f, 1.0f, 1.0f}, // scale
        {0.0f, 1.0f, 0.0f}, // color
        1.0f, // alpha
        {1.0f}, // shR
        {1.0f}, // shG
        {1.0f}  // shB
    });
    gaussians3D.push_back(Gaussian3D{
        {0.0f, 0.0f, 1.0f}, // position
        {0.0f, 0.0f, 0.0f, 1.0f}, // rotation
        {1.0f, 1.0f, 1.0f}, // scale
        {0.0f, 0.0f, 1.0f}, // color
        1.0f, // alpha
        {1.0f}, // shR
        {1.0f}, // shG
        {1.0f}  // shB
    });

    /*
    STEP 1: cerate the drawgraph elements
    */

    auto bufferElement = std::make_shared<BufferElementSinglePath<Gaussian3DBuffer>>(vulkanKernel, gaussians3D.size());
    auto& gaussian3DBuffer = bufferElement->getBuffer();
    gaussian3DBuffer.memcopyFrom(gaussians3D);
    
    std::shared_ptr<GaussianProjection> project3Dto2D = std::make_shared<GaussianProjection>(vulkanKernel, "shaders/gaussian_splatting_projection.comp.spv");

    auto& cameraMVP = project3Dto2D->getUbo()->ubo;
    
    InterfaceCameraOrbit cameraOrbit;
    cameraOrbit.initialize(vulkanKernel);
    cameraOrbit.setDistance(5.0f);
    cameraOrbit.update(cameraMVP);

    project3Dto2D->setInput(bufferElement);

    /*
    STEP 2: create the drawgraph backend and compile the drawgraph
    */
    auto& drawgraph = DrawGraph(vulkanKernel, 1);
    drawgraph.compileFrom(project3Dto2D);

    /*
    STEP 3: submit the drawgraph and compare the output
    */
    drawgraph.submitAndWait(vulkanKernel.getGraphicsQueue(), 0);

    // check the output buffer
    std::vector<Gaussian2D> gaussians2D(gaussians3D.size());
    project3Dto2D->getOutputBuffer().memcopyTo(gaussians2D);

    EXPECT_EQ(gaussians2D[0].position.x, 256.0f);
    EXPECT_EQ(gaussians2D[0].position.y, 256.0f);

    EXPECT_EQ(gaussians2D[1].position.x, 256.0f);
    EXPECT_EQ(gaussians2D[1].position.y, 256.0f);

    
    EXPECT_NEAR(gaussians2D[2].position.x, 379.6f, 0.1f);
    EXPECT_EQ(gaussians2D[2].position.y, 256.0f);

    EXPECT_EQ(gaussians2D[3].position.x, 256.0f);
    EXPECT_NEAR(gaussians2D[3].position.y, 132.39f, 0.1f);

   return;    
}

TEST(KlartraumVulkanGaussianSplatting, sort2DGaussians) {
    GlfwFrontend frontend;

    auto& core = frontend.getKlartraumCore();
    auto& vulkanKernel = core.getVulkanKernel();

    // Create a list of 2D gaussians with different depths (z values)
    std::vector<Gaussian2D> gaussians2D = {
        {{100.0f, 100.0f}, -0.5f, 0},
        {{200.0f, 200.0f}, 0.2f, 0},
        {{150.0f, 150.0f}, 0.8f, 0},
        {{250.0f, 250.0f}, 0.1f, 0},
        {{100.0f, 100.0f}, -0.5f, 4},
        {{200.0f, 200.0f}, 0.2f, 2},
        {{150.0f, 150.0f}, 0.8f, 1},
        {{250.0f, 250.0f}, 0.1f, 16}
    };

    // Create buffer and copy data
    auto bufferElement = std::make_shared<BufferElementSinglePath<Gaussian2DBuffer>>(vulkanKernel, gaussians2D.size());
    auto& gaussian2DBuffer = bufferElement->getBuffer();
    gaussian2DBuffer.memcopyFrom(gaussians2D);

    // Buffer transformation that sorts by z (depth)
    std::shared_ptr<GaussianSort> sort2DGaussians = std::make_shared<GaussianSort>(vulkanKernel, "shaders/gaussian_splatting_radix_sort.comp.spv");

    sort2DGaussians->setInput(bufferElement);

    uint32_t numElements = (uint32_t)gaussians2D.size();
    uint32_t numBitsPerPass = 4; // Number of bits per pass (4 bits for 16 bins)
    uint32_t numBins = 16; // Number of bins for sorting = 2 ^ numBitsPerPass
    uint32_t passes = 32 + 16; // 32 bits for depth, 16 bits for binning
    std::vector<SortPushConstants> pushConstants;
    for (uint32_t i = 0; i < passes / numBitsPerPass; i++) {
        pushConstants.push_back({i, numElements, numBins}); // pass, numElements, numBins
    }

    sort2DGaussians->setPushConstants(pushConstants);

    auto scratchBufferCounts = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanKernel, 16);
    auto scratchBufferOffsets = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanKernel, 16);
    auto totalGaussian2DCounts = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanKernel, 1);
    totalGaussian2DCounts->zero();


    sort2DGaussians->addScratchBufferElement(scratchBufferCounts);
    sort2DGaussians->addScratchBufferElement(scratchBufferOffsets);
    sort2DGaussians->addScratchBufferElement(totalGaussian2DCounts);

    auto& drawgraph = DrawGraph(vulkanKernel, 1);
    drawgraph.compileFrom(sort2DGaussians);

    drawgraph.submitAndWait(vulkanKernel.getGraphicsQueue(), 0);

    // Read back and check sorted order (should be descending by z)
    std::vector<Gaussian2D> sortedGaussians2D(gaussians2D.size());
    sort2DGaussians->getOutputBuffer().memcopyTo(sortedGaussians2D);

    EXPECT_LE(sortedGaussians2D[0].z, sortedGaussians2D[1].z);
    EXPECT_LE(sortedGaussians2D[1].z, sortedGaussians2D[2].z);
    EXPECT_LE(sortedGaussians2D[2].z, sortedGaussians2D[3].z);

    // check the exact sorted depth values
    EXPECT_FLOAT_EQ(sortedGaussians2D[0].z, -0.5f);
    EXPECT_FLOAT_EQ(sortedGaussians2D[1].z, 0.1f);
    EXPECT_FLOAT_EQ(sortedGaussians2D[2].z, 0.2f);
    EXPECT_FLOAT_EQ(sortedGaussians2D[3].z, 0.8f);

    // check th binMask values
    EXPECT_LE(sortedGaussians2D[4].binMask, sortedGaussians2D[5].binMask);
    EXPECT_LE(sortedGaussians2D[5].binMask, sortedGaussians2D[6].binMask);
    EXPECT_LE(sortedGaussians2D[6].binMask, sortedGaussians2D[7].binMask);

    return;
}

TEST(KlartraumVulkanGaussianSplatting, bin2DGaussians) {
    GlfwFrontend frontend;

    auto& core = frontend.getKlartraumCore();
    auto& vulkanKernel = core.getVulkanKernel();

    // Create a list of 2D gaussians with different positions and covariance matrices
    std::vector<Gaussian2D> gaussians2D = {
        // Gaussian in top-left quadrant
        {
            {115.0f, 100.0f},  // position
            0.2f,              // z
            0,
            glm::mat2(         // covariance matrix
                250.0f, 0.0f,
                0.0f, 50.0f
            )
        },
        // Gaussian in top-right quadrant
        {
            {400.0f, 100.0f},
            0.3f,
            0,            
            glm::mat2(
                60.0f, 0.0f,
                0.0f, 60.0f
            )
        },
        // Gaussian in bottom-left quadrant
        {
            {100.0f, 400.0f},
            0.4f,
            0,
            glm::mat2(
                40.0f, 0.0f,
                0.0f, 40.0f
            )
        },
        // Gaussian in bottom-right quadrant
        {
            {400.0f, 400.0f},
            0.5f,
            0,
            glm::mat2(
                70.0f, 0.0f,
                0.0f, 70.0f
            )
        }
    };

    // first, assign bin masks to the gaussians and duplicate entries
    // that are in more than one bin

    // Create buffer and copy data
    auto bufferElement = std::make_shared<BufferElementSinglePath<Gaussian2DBuffer>>(vulkanKernel, gaussians2D.size());
    auto& gaussian2DBuffer = bufferElement->getBuffer();
    gaussian2DBuffer.memcopyFrom(gaussians2D);

    // Create a buffer that holds the number of elements after duplication
    auto totalGaussian2DCounts = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanKernel, 1);
    totalGaussian2DCounts->zero();

    ProjectionPushConstants pushConstants = {
        (uint32_t)gaussians2D.size(),   // numElements
        4,                              // gridSize (4x4)
        512.0f,                         // screenWidth
        512.0f                          // screenHeight
    };

    auto binnedGaussians2D = std::make_shared<BufferElement<Gaussian2DBuffer>>(vulkanKernel, gaussians2D.size() * 2);
    binnedGaussians2D->zero();
    binnedGaussians2D->setName("BinnedGaussians2D");


    auto bin = std::make_shared<GaussianBinning>(vulkanKernel, "shaders/gaussian_splatting_binning.comp.spv");
    bin->setInput(bufferElement, 0);
    bin->setInput(binnedGaussians2D, 1);
    bin->setInput(totalGaussian2DCounts, 2);
    bin->setGroupCountX(gaussians2D.size() / 128 + 1);
    bin->setPushConstants({pushConstants});

    auto& drawgraph = DrawGraph(vulkanKernel, 1);
    drawgraph.compileFrom(bin);

    drawgraph.submitAndWait(vulkanKernel.getGraphicsQueue(), 0);

    std::vector<uint32_t> finalGaussiansCount(1);
    totalGaussian2DCounts->getBuffer(0).memcopyTo(finalGaussiansCount);

    std::vector<Gaussian2D> finalGaussians2D(gaussians2D.size() * 2);
    binnedGaussians2D->getBuffer(0).memcopyTo(finalGaussians2D);

    // TODO PROBLEM: order of gaussians is not deterministic, better to
    // merge the test with the sorting step
    EXPECT_EQ(finalGaussians2D[0].binMask, 0b0000000000000001);
    EXPECT_EQ(finalGaussians2D[1].binMask, 0b0000000000000010);
    EXPECT_EQ(finalGaussians2D[2].binMask, 0b0000000000001000);
    EXPECT_EQ(finalGaussians2D[3].binMask, 0b0001000000000000);
    EXPECT_EQ(finalGaussians2D[4].binMask, 0b1000000000000000);

   
    return;
}



TEST(KlartraumVulkanGaussianSplatting, binAndSortAndBoundsAndRender2DGaussians) {
    GlfwFrontend frontend;

    auto& core = frontend.getKlartraumCore();
    auto& vulkanKernel = core.getVulkanKernel();

    // Create a list of 2D gaussians with different positions and covariance matrices
    std::vector<Gaussian2D> gaussians2D = {
        // Gaussian in top-left quadrant
        {
            {115.0f, 100.0f},  // position
            0.2f,              // z
            0,
            glm::mat2(         // covariance matrix
                250.0f, 0.0f,
                0.0f, 50.0f
            ),
            {1.0f, 0.0f, 0.0f} // color
        },
        // Gaussian in bottom-left quadrant
        {
            {100.0f, 400.0f},
            0.6f,
            0,
            glm::mat2(
                40.0f, 0.0f,
                0.0f, 40.0f
            ),
            {0.0f, 1.0f, 0.0f} // color
        },
        // Gaussian in top-right quadrant
        {
            {400.0f, 100.0f},
            0.1f,
            0,            
            glm::mat2(
                60.0f, 0.0f,
                0.0f, 60.0f
            ),
            {0.0f, 0.0f, 1.0f} // color
        },
        // Gaussian in bottom-right quadrant
        {
            {400.0f, 400.0f},
            0.5f,
            0,
            glm::mat2(
                70.0f, 0.0f,
                0.0f, 70.0f
            ),
            {1.0f, 1.0f, 0.0f} // color
        }
    };

    // first, assign bin masks to the gaussians and duplicate entries
    // that are in more than one bin

    // Create buffer and copy data
    auto bufferElement = std::make_shared<BufferElementSinglePath<Gaussian2DBuffer>>(vulkanKernel, gaussians2D.size()*2);
    auto& gaussian2DBuffer = bufferElement->getBuffer();
    gaussian2DBuffer.memcopyFrom(gaussians2D);

    // Create a buffer that holds the number of elements after duplication
    auto totalGaussian2DCounts = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanKernel, 1);
    totalGaussian2DCounts->zero();

    ProjectionPushConstants pushConstants = {
        (uint32_t)gaussians2D.size(),   // numElements
        4,                              // gridSize (4x4)
        512.0f,                         // screenWidth
        512.0f                          // screenHeight
    };

    auto binnedGaussians2D = std::make_shared<BufferElement<Gaussian2DBuffer>>(vulkanKernel, gaussians2D.size() * 2);
    binnedGaussians2D->zero();
    binnedGaussians2D->setName("BinnedGaussians2D");


    auto bin = std::make_shared<GaussianBinning>(vulkanKernel, "shaders/gaussian_splatting_binning.comp.spv");
    bin->setInput(bufferElement, 0);
    bin->setInput(binnedGaussians2D, 1);
    bin->setInput(totalGaussian2DCounts, 2);
    bin->setGroupCountX(gaussians2D.size() / 128 + 1);
    bin->setPushConstants({pushConstants});


    // Buffer transformation that sorts by z (depth)
    std::shared_ptr<GaussianSort> sort2DGaussians = std::make_shared<GaussianSort>(vulkanKernel, "shaders/gaussian_splatting_radix_sort.comp.spv");

    sort2DGaussians->setInput(bin, 0, 1);

    auto scratchBufferCounts = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanKernel, 16);
    auto scratchBufferOffsets = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanKernel, 16);

    sort2DGaussians->addScratchBufferElement(scratchBufferCounts);
    sort2DGaussians->addScratchBufferElement(scratchBufferOffsets);
    sort2DGaussians->addScratchBufferElement(totalGaussian2DCounts);

    sort2DGaussians->setGroupCountX(gaussians2D.size() / 128 * 2 + 1);


    uint32_t numElements = (uint32_t)gaussians2D.size();
    uint32_t numBitsPerPass = 4; // Number of bits per pass (4 bits for 16 bins)
    uint32_t numBins = 16; // Number of bins for sorting = 2 ^ numBitsPerPass
    uint32_t passes = 32 + 16; // 32 bits for depth, 16 bits for binning
    std::vector<SortPushConstants> sortPushConstants;
    for (uint32_t i = 0; i < passes / numBitsPerPass; i++) {
        sortPushConstants.push_back({i, numElements, numBins}); // pass, numElements, numBins
    }

    sort2DGaussians->setPushConstants(sortPushConstants);

    // std::vector<uint32_t> binStartAndEnd(16*2);
    auto scratchBinStartAndEnd = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanKernel, 16*2);
    // auto& scratchBinStartAndEndBuffer = scratchBinStartAndEnd->getBuffer();
    // scratchBinStartAndEndBuffer.memcopyFrom(binStartAndEnd);
    scratchBinStartAndEnd->zero();

    auto computeBounds = std::make_shared<GaussianComputeBounds>(vulkanKernel, "shaders/gaussian_splatting_bin_bounds.comp.spv");

    ProjectionPushConstants computeBoundsPushConstants = {
        (uint32_t)gaussians2D.size(),   // numElements
        4,                              // gridSize (4x4)
        512.0f,                         // screenWidth
        512.0f                          // screenHeight
    };

    computeBounds->setInput(sort2DGaussians, 0, 0); //bufferElement, 0);
    computeBounds->setInput(bin, 1, 2); //totalGaussian2DCounts, 1);
    computeBounds->setInput(scratchBinStartAndEnd, 2 ); //scratchBinStartAndEnd, 2);
    computeBounds->setGroupCountX(gaussians2D.size() * 2 / 256 + 1);

    computeBounds->setPushConstants({computeBoundsPushConstants});


    std::vector<VkImageView> imageViews;
    std::vector<VkImage> images;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    for (int i = 0; i < 3; i++) {
        imageViews.push_back(vulkanKernel.getImageView(i));
        images.push_back(vulkanKernel.getSwapChainImage(i));
        imageAvailableSemaphores.push_back(vulkanKernel.imageAvailableSemaphoresPerImage[i]);        
        
    }

    auto imageViewSrc = std::make_shared<ImageViewSrc>(imageViews, images);
    imageViewSrc->setWaitFor(0, imageAvailableSemaphores[0]);
    imageViewSrc->setWaitFor(1, imageAvailableSemaphores[1]);
    imageViewSrc->setWaitFor(2, imageAvailableSemaphores[2]);    

    auto splat = std::make_shared<GaussianSplatting>(vulkanKernel, "shaders/gaussian_splatting_binned_splatting.comp.spv");

    std::vector<SplatPushConstants> splatPushConstants;
    for(uint32_t y = 0; y < 4; y++) {
        for(uint32_t x = 0; x < 4; x++) {
            splatPushConstants.push_back({
                (uint32_t)gaussians2D.size(),   // numElements
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

    auto camera = std::make_shared<CameraUboType>();
    
    auto swapChainImageFormat = vulkanKernel.getSwapChainImageFormat();
    auto swapChainExtent = vulkanKernel.getSwapChainExtent();
    auto renderpass = std::make_shared<RenderPass>(swapChainImageFormat, swapChainExtent);
    
    renderpass->setInput(splat, 0, 3);
    renderpass->setInput(camera, 1);    


    auto& drawgraph = DrawGraph(vulkanKernel, 3);
    drawgraph.compileFrom(renderpass);

    VkSemaphore finishSemaphore = VK_NULL_HANDLE;

    for(int i = 0; i < 1; i++) {
        auto [imageIndex, imageAvailableSemaphore] = vulkanKernel.beginRender();
        finishSemaphore = drawgraph.submitTo(vulkanKernel.getGraphicsQueue(), imageIndex);
        vulkanKernel.endRender(imageIndex, finishSemaphore);
    }


    std::vector<uint32_t> finalAdditionalGaussiansCount(1);
    totalGaussian2DCounts->getBuffer(0).memcopyTo(finalAdditionalGaussiansCount);

    std::vector<Gaussian2D> finalGaussians2D(gaussians2D.size() + finalAdditionalGaussiansCount[0]);
    bufferElement->getBuffer(0).memcopyTo(finalGaussians2D);

    EXPECT_EQ(finalGaussians2D[0].binMask, 0b0000000000000001);
    EXPECT_EQ(finalGaussians2D[1].binMask, 0b0000000000000010);
    EXPECT_EQ(finalGaussians2D[2].binMask, 0b0000000000001000);
    EXPECT_EQ(finalGaussians2D[3].binMask, 0b0001000000000000);
    EXPECT_EQ(finalGaussians2D[4].binMask, 0b1000000000000000);

    std::vector<uint32_t> binBounds(16 * 2);
    scratchBinStartAndEnd->getBuffer(0).memcopyTo(binBounds);

    // Print or check the bounds for each bin (start and end indices)
    for (int i = 0; i < 16; i++) {
        std::cout << "Bin " << i << " start: " << binBounds[i * 2]
                  << ", end: " << binBounds[i * 2 + 1] << std::endl;
    }

    vkQueueWaitIdle(vulkanKernel.getGraphicsQueue());
    
    return;
}

