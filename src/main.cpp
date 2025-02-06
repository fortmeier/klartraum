#include <iostream>

#include "klartraum/glfw_frontend.hpp"

#include "klartraum/draw_basics.hpp"
#include "klartraum/vulkan_gaussian_splatting.hpp"
#include "klartraum/interface_camera_orbit.hpp"

int main() {
    std::cout << "Wake up, dreamer!" << std::endl;

    klartraum::GlfwFrontend frontend;

    auto& core = frontend.getKlartraumCore();

    std::shared_ptr<klartraum::DrawBasics> axes = std::make_shared<klartraum::DrawBasics>(klartraum::DrawBasicsType::Axes);
    core.addDrawComponent(axes);

    std::string spzFile = "data/hornedlizard.spz";
    std::shared_ptr<klartraum::VulkanGaussianSplatting> splatting = std::make_shared<klartraum::VulkanGaussianSplatting>(spzFile);
    core.addDrawComponent(splatting);

    std::shared_ptr<klartraum::InterfaceCamera> cameraOrbit = std::make_shared<klartraum::InterfaceCameraOrbit>();
    core.setInterfaceCamera(cameraOrbit);

    frontend.loop();

    return 0;
}