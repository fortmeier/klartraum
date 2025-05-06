#include <gtest/gtest.h>

#include "klartraum/glfw_frontend.hpp"
#include "klartraum/vulkan_buffer.hpp"
#include "klartraum/vulkan_gaussian_splatting.hpp"
#include "klartraum/drawgraph/imageviewsrc.hpp"

using namespace klartraum;

// TEST(KlartraumVulkanGaussianSplatting, smoke) {
//     GlfwFrontend frontend;

//     auto& core = frontend.getKlartraumCore();
//     auto& vulkanKernel = core.getVulkanKernel();
//     auto& device = vulkanKernel.getDevice();
    
//     std::vector<VkImageView> imageViews;
//     std::vector<VkSemaphore> imageAvailableSemaphores;
//     for (int i = 0; i < 3; i++) {
//         imageViews.push_back(vulkanKernel.getImageView(i));
//     }

//     auto imageViewSrc = std::make_shared< ImageViewSrc>(imageViews);

//     std::string spzFile = "data/hornedlizard.spz";
//     std::shared_ptr<VulkanGaussianSplatting> splatting = std::make_shared<VulkanGaussianSplatting>(spzFile);
//     splatting->setInput(imageViewSrc, 0);

//     std::shared_ptr<CameraUboType> cameraUBO = std::make_shared<CameraUboType>();
//     cameraUBO->ubo.proj = glm::perspective(glm::radians(45.0f), (float)BackendConfig::WIDTH / (float)BackendConfig::HEIGHT, 0.1f, 100.0f);

//     splatting->setInput(cameraUBO, 1);
    
//     core.add(splatting);
    
// }

TEST(KlartraumVulkanGaussianSplatting, smoke) {
    GlfwFrontend frontend;

    auto& core = frontend.getKlartraumCore();
    auto& vulkanKernel = core.getVulkanKernel();
    auto& device = vulkanKernel.getDevice();
    typedef VulkanBuffer<Gaussian3D> Gaussian3DBuffer;
    typedef VulkanBuffer<Gaussian2D> Gaussian2DBuffer;

    std::vector<Gaussian3D> gaussians3D;
    gaussians3D.push_back(Gaussian3D{
        {0.6f, 0.7f, 0.8f}, // position
        {0.0f, 0.0f, 0.0f, 1.0f}, // rotation
        {1.0f, 1.0f, 1.0f}, // scale
        {1.0f, 1.0f, 1.0f}, // color
        1.0f, // alpha
        {1.01f}, // shR
        {1.02f}, // shG
        {1.03f}  // shB
    });
    gaussians3D.push_back(Gaussian3D{
        {1.1f, 0.0f, 0.0f}, // position
        {0.0f, 0.0f, 0.0f, 1.05f}, // rotation
        {1.06f, 1.07f, 1.0f}, // scale
        {1.0f, 0.0f, 0.0f}, // color
        1.0f, // alpha
        {1.0f}, // shR
        {1.0f}, // shG
        {1.0f}  // shB
    });
    gaussians3D.push_back(Gaussian3D{
        {0.0f, 1.2f, 0.0f}, // position
        {0.0f, 0.0f, 0.0f, 1.0f}, // rotation
        {1.0f, 1.0f, 1.0f}, // scale
        {0.0f, 1.0f, 0.0f}, // color
        1.0f, // alpha
        {1.0f}, // shR
        {1.0f}, // shG
        {1.0f}  // shB
    });
    gaussians3D.push_back(Gaussian3D{
        {0.0f, 0.0f, 1.3f}, // position
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
    
    cameraMVP.proj = glm::perspective(glm::radians(45.0f), (float)BackendConfig::WIDTH / (float)BackendConfig::HEIGHT, 0.1f, 100.0f);

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

    for(auto& g : gaussians2D) {
        std::cout << g.position.x << " " << g.position.y << std::endl;
    }

//    for (int i = 0; i < 7; i++) {
//        EXPECT_EQ(data[i] * 3.0f, data_out[i]);
//    }
   return;    

}