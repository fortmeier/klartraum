#ifndef VULKAN_CONTEXT_HPP
#define VULKAN_CONTEXT_HPP

#include <algorithm>
#include <iostream>
#include <memory>
#include <optional>
#include <queue>
#include <set>
#include <stdexcept>
#include <vector>

#include <vulkan/vulkan.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

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

class VulkanContext {
    /*
     * The VulkanContext kernels responsibities are
     * - create the Vulkan instance
     * - create the Vulkan device
     * - create the Vulkan swapchain
     * - setup the debug messenger
     * - setup the layers and extensions
     *
     * Is should not deal with
     * - creating the Vulkan renderpasses, pipelines, framebuffers, etc.
     *
     * This is to be done in the computegraphs, which can be used standalone or via the
     * Klartraum core.
     *
     */
private:
    const uint32_t WIDTH = 512;
    const uint32_t HEIGHT = 512;

    VkDebugUtilsMessengerEXT debugMessenger;

    std::vector<VkImage> swapChainImages;

    VkFormat swapChainImageFormat;

    std::vector<VkImageView> swapChainImageViews;

    const std::vector<const char*> validationLayers = {
        "VK_LAYER_KHRONOS_validation"};

    const std::vector<const char*> deviceExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME,
        VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME,
        VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME,
#ifdef __APPLE__
        "VK_KHR_portability_subset",  // Required for MoltenVK on macOS
#endif
    };

#ifdef NDEBUG
    const bool enableValidationLayers = false;
    const bool enableGPUPrintf = false;
#else
    const bool enableValidationLayers = true;
    const bool enableGPUPrintf = true;
#endif

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

    enum class State {
        UNINITIALIZED,
        INITIALIZED,
        SHUTDOWN
    } state = State::UNINITIALIZED;

public:
    VulkanContext();

    ~VulkanContext();

    void initialize(VkSurfaceKHR& surface);
    void shutdown();

    template<typename T, typename... Args>
    std::shared_ptr<T> create(Args&&... args) {
        if (state != State::INITIALIZED) {
            throw std::runtime_error("VulkanContext is not initialized!");
        }
        return std::make_shared<T>(*this, std::forward<Args>(args)...);
    }

    QueueFamilyIndices findQueueFamiliesPhysicalDevice();

    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;

    VkDevice device;
    VkSwapchainKHR swapChain;

    VkQueue graphicsQueue;
    VkQueue presentQueue;

    VkInstance instance;

    VkSurfaceKHR surface;

    std::vector<VkFramebuffer> swapChainFramebuffers;
    VkExtent2D swapChainExtent;

    VkInstance& getInstance();

    VkDevice& getDevice();
    VkSwapchainKHR& getSwapChain();

    VkQueue& getGraphicsQueue();

    VkImageView& getImageView(uint32_t imageIndex);

    VkImage& getSwapChainImage(uint32_t imageIndex);

    VkExtent2D& getSwapChainExtent();

    const VkFormat& getSwapChainImageFormat() const;

    QueueFamilyIndices getQueueFamilyIndices();

    BackendConfig config;

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    BackendConfig& getConfig();

    std::vector<VkFence> inFlightFences;
    std::vector<VkSemaphore> imageAvailableSemaphoresPerFrame;
    std::vector<VkSemaphore> imageAvailableSemaphoresPerImage;
    std::vector<VkSemaphore> renderEndSemaphores;

    void createSyncObjects();

    uint32_t currentFrame = 0;

    std::tuple<uint32_t, VkSemaphore&> beginRender();
    void endRender(uint32_t imageIndex, VkSemaphore& renderFinishedSemaphore);
    void stopRender();

    void createCommandPool();

    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;
};

} // namespace klartraum

#endif // VULKAN_CONTEXT_HPP