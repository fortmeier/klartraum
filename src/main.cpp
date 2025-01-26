#include <iostream>
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "klartraum/backend_vulkan.hpp"
#include "klartraum/vulkan_gaussian_splatting.hpp"

int main() {
    std::cout << "Hello, World!" << std::endl;

    PyStatus status;
    PyConfig config;
    PyConfig_InitPythonConfig(&config);

    klartraum::BackendVulkan backendVulkan;

    backendVulkan.initialize();

    std::string spzFile = "data/hornedlizard.spz";
    
    std::unique_ptr<klartraum::VulkanGaussianSplatting> gaussianSplatting = std::make_unique<klartraum::VulkanGaussianSplatting>(backendVulkan, spzFile);

    backendVulkan.addDrawComponent(std::move(gaussianSplatting));

    backendVulkan.loop();

    backendVulkan.shutdown();

    return 0;
}