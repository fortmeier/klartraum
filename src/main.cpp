#include <iostream>
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "klartraum/glfw_frontend.hpp"

#include "klartraum/draw_basics.hpp"
#include "klartraum/vulkan_gaussian_splatting.hpp"
#include "klartraum/interface_camera_orbit.hpp"

int main() {
    std::cout << "Hello, World!" << std::endl;

    PyStatus status;
    PyConfig config;
    PyConfig_InitPythonConfig(&config);

    klartraum::GlfwFrontend frontend;

    frontend.initialize();

    auto& core = frontend.getKlartraumCore();

    auto& kernel = core.getVulkanKernel();

    
    std::shared_ptr<klartraum::DrawBasics> axes = std::make_shared<klartraum::DrawBasics>(kernel, klartraum::DrawBasicsType::Axes);
    core.addDrawComponent(axes);


    std::string spzFile = "data/hornedlizard.spz";
    std::shared_ptr<klartraum::VulkanGaussianSplatting> splatting = std::make_shared<klartraum::VulkanGaussianSplatting>(kernel, spzFile);
    core.addDrawComponent(splatting);


    std::shared_ptr<klartraum::InterfaceCamera> cameraOrbit = std::make_shared<klartraum::InterfaceCameraOrbit>(&kernel);
    core.setInterfaceCamera(cameraOrbit);

    frontend.loop();

    frontend.shutdown();

    return 0;
}