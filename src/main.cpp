#include <iostream>
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "klartraum/backend_vulkan.hpp"
#include "klartraum/draw_basics.hpp"
#include "klartraum/vulkan_gaussian_splatting.hpp"

#include "klartraum/interface_camera_orbit.hpp"

int main() {
    std::cout << "Hello, World!" << std::endl;

    PyStatus status;
    PyConfig config;
    PyConfig_InitPythonConfig(&config);

    klartraum::GlfwFrontend backendVulkan;

    backendVulkan.initialize();

    
    std::unique_ptr<klartraum::DrawBasics> axes = std::make_unique<klartraum::DrawBasics>(backendVulkan, klartraum::DrawBasicsType::Axes);
    backendVulkan.addDrawComponent(std::move(axes));


    std::string spzFile = "data/hornedlizard.spz";
    std::unique_ptr<klartraum::VulkanGaussianSplatting> splatting = std::make_unique<klartraum::VulkanGaussianSplatting>(backendVulkan, spzFile);
    backendVulkan.addDrawComponent(std::move(splatting));


    std::shared_ptr<klartraum::InterfaceCamera> cameraOrbit = std::make_shared<klartraum::InterfaceCameraOrbit>(&backendVulkan);
    backendVulkan.setInterfaceCamera(cameraOrbit);

    backendVulkan.loop();

    backendVulkan.shutdown();

    return 0;
}