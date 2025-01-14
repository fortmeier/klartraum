#include <iostream>

#include "klartraum/backend_vulkan.hpp"

int main() {
    std::cout << "Hello, World!" << std::endl;

    klartraum::BackendVulkan backendVulkan;

    backendVulkan.initialize();

    backendVulkan.loop();

    backendVulkan.shutdown();
    
    return 0;
}