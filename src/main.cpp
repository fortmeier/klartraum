#include <iostream>
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "klartraum/backend_vulkan.hpp"

int main() {
    std::cout << "Hello, World!" << std::endl;

    PyStatus status;
    PyConfig config;
    PyConfig_InitPythonConfig(&config);

    klartraum::BackendVulkan backendVulkan;

    backendVulkan.initialize();

    backendVulkan.loop();

    backendVulkan.shutdown();

    return 0;
}