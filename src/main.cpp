#include <iostream>
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "klartraum/backend_vulkan.hpp"
#include "klartraum/draw_basics.hpp"

#include "klartraum/interface_camera_orbit.hpp"

int main() {
    std::cout << "Hello, World!" << std::endl;

    PyStatus status;
    PyConfig config;
    PyConfig_InitPythonConfig(&config);

    klartraum::BackendVulkan backendVulkan;

    backendVulkan.initialize();

    std::string spzFile = "data/hornedlizard.spz";
    
    std::unique_ptr<klartraum::DrawBasics> gaussianSplatting = std::make_unique<klartraum::DrawBasics>(backendVulkan, klartraum::DrawBasicsType::Axes);

    backendVulkan.addDrawComponent(std::move(gaussianSplatting));

    std::shared_ptr<klartraum::InterfaceCamera> cameraOrbit = std::make_shared<klartraum::InterfaceCameraOrbit>(&backendVulkan);
    backendVulkan.setInterfaceCamera(cameraOrbit);

    backendVulkan.loop();

    backendVulkan.shutdown();

    return 0;
}