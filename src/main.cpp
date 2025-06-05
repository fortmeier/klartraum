#include <iostream>

#include "klartraum/glfw_frontend.hpp"

#include "klartraum/draw_basics.hpp"
#include "klartraum/vulkan_gaussian_splatting.hpp"
#include "klartraum/interface_camera_orbit.hpp"

int main() {
    std::cout << "Wake up, dreamer!" << std::endl;

    klartraum::GlfwFrontend frontend;

    auto& core = frontend.getKlartraumCore();

    klartraum::RenderPassPtr renderpass = core.createRenderPass();

    std::shared_ptr<klartraum::DrawBasics> axes = std::make_shared<klartraum::DrawBasics>(klartraum::DrawBasicsType::Axes);
    renderpass->addDrawComponent(axes);

    auto& vulkanKernel = core.getVulkanKernel();
    
    auto& cameraUBO = renderpass->getCameraUBO();
    std::string spzFile = "data/hornedlizard.spz";
    std::shared_ptr<klartraum::VulkanGaussianSplatting> splatting = std::make_shared<klartraum::VulkanGaussianSplatting>(vulkanKernel, renderpass, cameraUBO, spzFile);
    
    core.add(splatting);

    std::shared_ptr<klartraum::InterfaceCamera> cameraOrbit = std::make_shared<klartraum::InterfaceCameraOrbit>(klartraum::InterfaceCameraOrbit::UpDirection::Y);
    core.setInterfaceCamera(cameraOrbit);
    core.setCameraUBO(cameraUBO);

    frontend.loop();

    return 0;
}