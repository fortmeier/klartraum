#include <iostream>
#include <filesystem>

#include "klartraum/glfw_frontend.hpp"

#include "klartraum/draw_basics.hpp"
#include "klartraum/vulkan_gaussian_splatting.hpp"
#include "klartraum/interface_camera_orbit.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"

int main() {
    std::cout << "Wake up, dreamer!" << std::endl;

    klartraum::GlfwFrontend frontend;

    auto& engine = frontend.getKlartraumEngine();

    klartraum::RenderPassPtr renderpass = engine.createRenderPass();

    std::shared_ptr<klartraum::DrawBasics> axes = std::make_shared<klartraum::DrawBasics>(klartraum::DrawBasicsType::Axes);
    renderpass->addDrawComponent(axes);

    auto& vulkanContext = engine.getVulkanContext();

    // Build ImageViewSrc from the Vulkan context's offscreen images.
    // VulkanGaussianSplatting writes its output to these images and transitions
    // them to VK_IMAGE_LAYOUT_GENERAL (no swapchain required).
    uint32_t numImages = vulkanContext.getNumberOfSwapChainImages();
    auto extent = vulkanContext.getSwapChainExtent();
    std::vector<VkImageView> imageViews;
    std::vector<VkImage>     images;
    std::vector<VkExtent2D>  extents(numImages, extent);
    for (uint32_t i = 0; i < numImages; ++i) {
        imageViews.push_back(vulkanContext.getImageView(i));
        images.push_back(vulkanContext.getSwapChainImage(i));
    }
    auto imageViewSrc = std::make_shared<klartraum::ImageViewSrc>(imageViews, images, extents);

    auto cameraUBO = renderpass->getCameraUBO();
    cameraUBO->setName("CameraUBO");

    std::string spzFile = "./3rdparty/spz/samples/racoonfamily.spz";
    std::shared_ptr<klartraum::VulkanGaussianSplatting> splatting =
        vulkanContext.create<klartraum::VulkanGaussianSplatting>(imageViewSrc, cameraUBO, spzFile);

    engine.add(splatting);

    std::shared_ptr<klartraum::InterfaceCameraOrbit> cameraOrbit = std::make_shared<klartraum::InterfaceCameraOrbit>(klartraum::InterfaceCameraOrbit::UpDirection::Y);
    cameraOrbit->setAzimuth(0.9);
    cameraOrbit->setElevation(-0.5);
    cameraOrbit->setPosition({-0.5, 0.0, 0.5});
    cameraOrbit->setDistance(1.0);
    engine.setInterfaceCamera(cameraOrbit);
    engine.setCameraUBO(cameraUBO);

    frontend.loop();

    return 0;
}
