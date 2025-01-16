#include <iostream>
#include <Python.h> // Add this line

#include "klartraum/backend_vulkan.hpp"

int main() {
    std::cout << "Hello, World!" << std::endl;

    // Initialize the Python Interpreter
    Py_Initialize();

    klartraum::BackendVulkan backendVulkan;

    backendVulkan.initialize();

    backendVulkan.loop();

    backendVulkan.shutdown();

    // Finalize the Python Interpreter
    Py_Finalize();
    
    return 0;
}