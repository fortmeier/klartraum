#include <stdexcept>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan/vulkan.h>

#include "klartraum/backend_vulkan.hpp"

namespace klartraum {

class BackendVulkanImplentation {
    private:
    VkInstance instance;

    public:
    void createInstance() {
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "klartraum demo";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "klartraum";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;

        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        createInfo.enabledExtensionCount = glfwExtensionCount;
        createInfo.ppEnabledExtensionNames = glfwExtensions;

        createInfo.enabledLayerCount = 0;

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
           throw std::runtime_error("failed to create instance!");
        }
    }

};


BackendVulkan::BackendVulkan() {
    // Constructor
    impl = new BackendVulkanImplentation();
}

BackendVulkan::~BackendVulkan()
{
    // Destructor
    delete impl;
}

void BackendVulkan::initialize() {
    // Initialize Vulkan
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }

    impl->createInstance();
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