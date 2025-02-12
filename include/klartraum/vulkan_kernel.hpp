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

#include <vulkan/vulkan.h>

#include "klartraum/backend_config.hpp"
#include "klartraum/camera.hpp"

namespace klartraum {

struct QueueFamilyIndices {
    std::optional<uint32_t> presentFamily;
    std::optional<uint32_t> graphicsAndComputeFamily;

    bool isComplete() {
        return graphicsAndComputeFamily.has_value() && presentFamily.has_value();
    }
};


VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocater, VkDebugUtilsMessengerEXT* pDebugMessenger);


void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator);

struct SwapChainSupportDetails {
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

class VulkanKernel {
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

    const std::vector<const char*> validationLayers = {
		"VK_LAYER_KHRONOS_validation"
	};

    const std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME 
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
    VulkanKernel(/*GLFWwindow* window*/);

    void initialize(VkSurfaceKHR& surface);


    ~VulkanKernel();

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

    VkInstance& getInstance();

    VkDevice& getDevice();
    VkSwapchainKHR& getSwapChain();
    VkRenderPass& getRenderPass();
    
    VkQueue& getGraphicsQueue();


    VkFramebuffer& getFramebuffer(uint32_t imageIndex);
    VkImageView& getImageView(uint32_t imageIndex);
    
    VkExtent2D& getSwapChainExtent();

    

    QueueFamilyIndices getQueueFamilyIndices();

    BackendConfig config;

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    BackendConfig& getConfig();

    std::vector<VkFence> inFlightFences;
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;  
    //std::vector<VkSemaphore> signalSemaphores;

    void createSyncObjects();

    uint32_t currentFrame = 0;

    uint32_t beginRender();
    void endRender(uint32_t imageIndex);

    void beginRenderPass(uint32_t currentFrame, VkFramebuffer& framebuffer);
    void endRenderPass(uint32_t currentFrame);

    void createCommandPool();
    void createCommandBuffers();

    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;


    std::shared_ptr<Camera> camera;
 
    Camera& getCamera();


};

} // namespace klartraum

#endif // VULKAN_KERNEL_HPP