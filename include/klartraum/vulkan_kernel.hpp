#ifndef VULKAN_KERNEL_HPP
#define VULKAN_KERNEL_HPP

#include <vector>
#include <queue>
#include <optional>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <set>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan/vulkan.h>


#include "klartraum/backend_config.hpp"
#include "klartraum/draw_component.hpp"
#include "klartraum/camera.hpp"
#include "klartraum/interface_camera.hpp"
#include "klartraum/events.hpp"

namespace klartraum {

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};


VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocater, VkDebugUtilsMessengerEXT* pDebugMessenger);


void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator);

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

class BackendVulkanImplementation {
    private:
    //GLFWwindow* window = nullptr;
    const uint32_t WIDTH = 800;
    const uint32_t HEIGHT = 600;

    VkDebugUtilsMessengerEXT debugMessenger;



    // VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
    //     if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
    //         return capabilities.currentExtent;
    //     }
    //     else {
    //         int width, height;
    //         glfwGetFramebufferSize(window, &width, &height);

    //         VkExtent2D actualExtent = {
    //             static_cast<uint32_t>(width),
    //             static_cast<uint32_t>(height)
    //         };

    //         actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
    //         actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

    //         return actualExtent;
    //     }
    // }
    std::vector<VkImage> swapChainImages;

    VkFormat swapChainImageFormat;
    

    std::vector<VkImageView> swapChainImageViews;

    VkCommandPool commandPool;

    std::vector<VkCommandBuffer> commandBuffers;

    uint32_t currentFrame = 0;

    const std::vector<const char*> validationLayers = {
		"VK_LAYER_KHRONOS_validation"
	};

    const std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };


#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif

    // void initWindow() {
    //     glfwInit();
    //     glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    //     glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    //     window = glfwCreateWindow(this->WIDTH, this->HEIGHT, "Vulkan", nullptr, nullptr);
    // }

    bool checkValidationLayerSupport();

    std::vector<const char*> getRequiredExtensions();

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData);

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);

    void createInstance();

    void setupDebugMessenger();

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);

    bool isDeviceSuitable(VkPhysicalDevice device);

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
    
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);

    void createSwapChain();
    
    void createImageViews();

    bool checkDeviceExtensionSupport(VkPhysicalDevice device);

    void pickPhysicalDevice();

    void createLogicalDevice();

    void createRenderPass();

    void createFramebuffers();

    public:
    BackendVulkanImplementation(/*GLFWwindow* window*/);

    void initialize(VkSurfaceKHR& surface);


    ~BackendVulkanImplementation();

    QueueFamilyIndices findQueueFamiliesPhysicalDevice();

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;

    VkDevice device;
    VkSwapchainKHR swapChain;
    VkRenderPass renderPass;

    VkQueue graphicsQueue;
    VkQueue presentQueue;


    VkInstance instance;

    VkSurfaceKHR surface;




    std::vector<VkFramebuffer> swapChainFramebuffers;
    VkExtent2D swapChainExtent;

};

} // namespace klartraum

#endif // VULKAN_KERNEL_HPP