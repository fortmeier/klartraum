#include <gtest/gtest.h>

#include "klartraum/glfw_frontend.hpp"
#include "klartraum/vulkan_buffer.hpp"
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
        
    }

    auto imageViewSrc = std::make_shared<ImageViewSrc>(imageViews, images);

    std::string spzFile = "data/hornedlizard.spz";
    std::shared_ptr<VulkanGaussianSplatting> splatting = std::make_shared<VulkanGaussianSplatting>(spzFile);
    splatting->setInput(imageViewSrc, 0);

    std::shared_ptr<CameraUboType> cameraUBO = std::make_shared<CameraUboType>();
    cameraUBO->ubo.proj = glm::perspective(glm::radians(45.0f), (float)BackendConfig::WIDTH / (float)BackendConfig::HEIGHT, 0.1f, 100.0f);

    splatting->setInput(cameraUBO, 1);
    
    core.add(splatting);
    
}

TEST(KlartraumVulkanGaussianSplatting, project) {
    GlfwFrontend frontend;

    auto& core = frontend.getKlartraumCore();
    auto& vulkanKernel = core.getVulkanKernel();
    auto& device = vulkanKernel.getDevice();
    typedef VulkanBuffer<Gaussian3D> Gaussian3DBuffer;
    typedef VulkanBuffer<Gaussian2D> Gaussian2DBuffer;

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

    auto bufferElement = std::make_shared<BufferElement<Gaussian3DBuffer>>(vulkanKernel, gaussians3D.size());
    auto& gaussian3DBuffer = bufferElement->getBuffer();
    gaussian3DBuffer.memcopyFrom(gaussians3D);
    
    typedef BufferTransformation<Gaussian3DBuffer, Gaussian2DBuffer, CameraUboType> GaussianProjection;
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

struct sortPushConstants {
    uint32_t pass;
    uint32_t numElements;
    uint32_t numBins;
};

TEST(KlartraumVulkanGaussianSplatting, sort2DGaussians) {
    GlfwFrontend frontend;

    auto& core = frontend.getKlartraumCore();
    auto& vulkanKernel = core.getVulkanKernel();
    typedef VulkanBuffer<Gaussian2D> Gaussian2DBuffer;

    // Create a list of 2D gaussians with different depths (z values)
    std::vector<Gaussian2D> gaussians2D = {
        {{100.0f, 100.0f}, -0.5f},
        {{200.0f, 200.0f}, 0.2f},
        {{150.0f, 150.0f}, 0.8f},
        {{250.0f, 250.0f}, 0.1f}
    };

    // Create buffer and copy data
    auto bufferElement = std::make_shared<BufferElement<Gaussian2DBuffer>>(vulkanKernel, gaussians2D.size());
    auto& gaussian2DBuffer = bufferElement->getBuffer();
    gaussian2DBuffer.memcopyFrom(gaussians2D);

    // Buffer transformation that sorts by z (depth)
    typedef BufferTransformation<Gaussian2DBuffer, Gaussian2DBuffer, void, sortPushConstants> GaussianSort;
    std::shared_ptr<GaussianSort> sort2DGaussians = std::make_shared<GaussianSort>(vulkanKernel, "shaders/gaussian_splatting_radix_sort.comp.spv");

    sort2DGaussians->setInput(bufferElement);

    uint32_t numElements = (uint32_t)gaussians2D.size();
    uint32_t numBins = 16; // Number of bins for sorting
    uint32_t numBitsPerPass = 4; // Number of bits per pass (4 bits for 16 bins)
    std::vector<sortPushConstants> pushConstants;
    for (uint32_t i = 0; i < 32 / numBitsPerPass; i++) {
        pushConstants.push_back({i, numElements, numBins}); // pass, numElements, numBins
    }

    sort2DGaussians->setPushConstants(pushConstants);

    auto scratchBufferCounts = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanKernel, 16);
    auto scratchBufferOffsets = std::make_shared<BufferElement<VulkanBuffer<uint32_t>>>(vulkanKernel, 16);

    sort2DGaussians->addScratchBufferElement(scratchBufferCounts);
    sort2DGaussians->addScratchBufferElement(scratchBufferOffsets);

    auto& drawgraph = DrawGraph(vulkanKernel, 1);
    drawgraph.compileFrom(sort2DGaussians);

    drawgraph.submitAndWait(vulkanKernel.getGraphicsQueue(), 0);

    // Read back and check sorted order (should be descending by z)
    std::vector<Gaussian2D> sortedGaussians2D(gaussians2D.size());
    sort2DGaussians->getOutputBuffer().memcopyTo(sortedGaussians2D);

    EXPECT_LE(sortedGaussians2D[0].z, sortedGaussians2D[1].z);
    EXPECT_LE(sortedGaussians2D[1].z, sortedGaussians2D[2].z);
    EXPECT_LE(sortedGaussians2D[2].z, sortedGaussians2D[3].z);

    // Optionally, check the exact sorted values
    EXPECT_FLOAT_EQ(sortedGaussians2D[0].z, -0.5f);
    EXPECT_FLOAT_EQ(sortedGaussians2D[1].z, 0.1f);
    EXPECT_FLOAT_EQ(sortedGaussians2D[2].z, 0.2f);
    EXPECT_FLOAT_EQ(sortedGaussians2D[3].z, 0.8f);

    return;
}

struct binPushConstants {
    uint32_t numElements;
    uint32_t gridSize;  // 4 for 4x4 grid
    float screenWidth;
    float screenHeight;
};

TEST(KlartraumVulkanGaussianSplatting, bin2DGaussians) {
    GlfwFrontend frontend;

    auto& core = frontend.getKlartraumCore();
    auto& vulkanKernel = core.getVulkanKernel();
    typedef VulkanBuffer<Gaussian2D> Gaussian2DBuffer;

    // Create a list of 2D gaussians with different positions and covariance matrices
    std::vector<Gaussian2D> gaussians2D = {
        // Gaussian in top-left quadrant
        {
            {115.0f, 100.0f},  // position
            0.2f,              // z
            glm::mat2(         // covariance matrix
                250.0f, 0.0f,
                0.0f, 50.0f
            )
        },
        // Gaussian in top-right quadrant
        {
            {400.0f, 100.0f},
            0.3f,
            glm::mat2(
                60.0f, 0.0f,
                0.0f, 60.0f
            )
        },
        // Gaussian in bottom-left quadrant
        {
            {100.0f, 400.0f},
            0.4f,
            glm::mat2(
                40.0f, 0.0f,
                0.0f, 40.0f
            )
        },
        // Gaussian in bottom-right quadrant
        {
            {400.0f, 400.0f},
            0.5f,
            glm::mat2(
                70.0f, 0.0f,
                0.0f, 70.0f
            )
        }
    };

    // Create buffer and copy data
    auto bufferElement = std::make_shared<BufferElement<Gaussian2DBuffer>>(vulkanKernel, gaussians2D.size());
    auto& gaussian2DBuffer = bufferElement->getBuffer();
    gaussian2DBuffer.memcopyFrom(gaussians2D);


    typedef VulkanBuffer<uint32_t> Gaussian2DCountsAndIndices;
    // Buffer transformation that bins gaussians into 4x4 grid
    typedef BufferTransformation<Gaussian2DBuffer, Gaussian2DCountsAndIndices, void, binPushConstants> GaussianBinning;
    std::shared_ptr<GaussianBinning> bin2DGaussians = std::make_shared<GaussianBinning>(vulkanKernel, "shaders/gaussian_splatting_binning.comp.spv");

    bin2DGaussians->setInput(bufferElement);

    uint32_t outputSize = 4 * 4 * gaussians2D.size() + 4 * 4; // 4x4 grid + 4x4 counts
    bin2DGaussians->setCustomOutputSize(outputSize);

    // Set up push constants for binning
    binPushConstants pushConstants = {
        (uint32_t)gaussians2D.size(),  // numElements
        4,                             // gridSize (4x4)
        512.0f,                        // screenWidth
        512.0f                         // screenHeight
    };
    bin2DGaussians->setPushConstants({pushConstants});

    auto& drawgraph = DrawGraph(vulkanKernel, 1);
    drawgraph.compileFrom(bin2DGaussians);

    drawgraph.submitAndWait(vulkanKernel.getGraphicsQueue(), 0);

    // Read back and verify binning results
    std::vector<uint32_t> gaussian2DCountsAndIndices(outputSize);
    bin2DGaussians->getOutputBuffer().memcopyTo(gaussian2DCountsAndIndices);

    uint32_t binCounts[16];
    for (uint32_t i = 0; i < 16; i++) {
        binCounts[i] = gaussian2DCountsAndIndices[i];
    }
    std::vector<uint32_t> indicesPerBin[16];
    for (uint32_t i = 0; i < 16; i++) {
        indicesPerBin[i] = std::vector<uint32_t>(binCounts[i]);
        for (uint32_t j = 0; j < binCounts[i]; j++) {
            indicesPerBin[i][j] = gaussian2DCountsAndIndices[16 + i * gaussians2D.size() + j];
        }
    }   

    // Verify that gaussians are binned correctly
    // Each gaussian should be in its respective quadrant
    EXPECT_EQ(binCounts[0], 1);  // Top-left quadrant
    EXPECT_EQ(binCounts[1], 1);  // Right of top-left quadrant 
    EXPECT_EQ(binCounts[3], 1);  // Top-right quadrant
    EXPECT_EQ(binCounts[12], 1); // Bottom-left quadrant
    EXPECT_EQ(binCounts[15], 1); // Bottom-right quadrant

    // Verify that other bins are empty
    for (int i = 0; i < 16; i++) {
        if (i != 0 && i != 1 && i != 3 && i != 12 && i != 15) {
            EXPECT_EQ(binCounts[i], 0);
        }
    }

    return;
}

