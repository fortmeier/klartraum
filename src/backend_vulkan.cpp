#include <stdexcept>

#include <GLFW/glfw3.h>

#include "klartraum/backend_vulkan.hpp"

namespace klartraum {

BackendVulkan::BackendVulkan() {
    // Constructor
}

BackendVulkan::~BackendVulkan()
{
}

void BackendVulkan::initialize() {
    // Initialize Vulkan
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }
}

void BackendVulkan::loop() {
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(800, 600, "Vulkan window", nullptr, nullptr);

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
    }

    glfwDestroyWindow(window);
}

void BackendVulkan::shutdown() {
    // Shutdown Vulkan
    glfwTerminate();
}

} // namespace klartraum