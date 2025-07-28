#include <iostream>
#include <filesystem>

#include "klartraum/glfw_frontend.hpp"

#include "klartraum/draw_basics.hpp"
#include "klartraum/vulkan_gaussian_splatting.hpp"
#include "klartraum/interface_camera_orbit.hpp"

int main() {
    std::cout << "Wake up, dreamer!" << std::endl;

    klartraum::GlfwFrontend frontend;

    auto& engine = frontend.getKlartraumEngine();

    klartraum::RenderPassPtr renderpass = engine.createRenderPass();

    std::shared_ptr<klartraum::DrawBasics> axes = std::make_shared<klartraum::DrawBasics>(klartraum::DrawBasicsType::Axes);
    renderpass->addDrawComponent(axes);

    auto& vulkanContext = engine.getVulkanContext();
    
    auto cameraUBO = renderpass->getCameraUBO();
    cameraUBO->setName("CameraUBO");
    
    std::string spzFile = "./3rdparty/spz/samples/racoonfamily.spz";
    std::shared_ptr<klartraum::VulkanGaussianSplatting> splatting = vulkanContext.create<klartraum::VulkanGaussianSplatting>(renderpass, cameraUBO, spzFile);
    
    // this adds a second renderpass
    // to test if this fixes the swapchain issue
    auto swapChainImageFormat = vulkanContext.getSwapChainImageFormat();
    auto swapChainExtent = vulkanContext.getSwapChainExtent();
    auto renderpass2 = std::make_shared<klartraum::RenderPass>(swapChainImageFormat, swapChainExtent);

    renderpass2->setInput(splatting, 0, 0);

    auto camera = std::make_shared<klartraum::CameraUboType>();

    renderpass2->setInput(camera, 1);
    renderpass2->loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; // Clear the render pass

    engine.add(renderpass2);

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
