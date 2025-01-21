#ifndef BACKEND_VULKAN_HPP
#define BACKEND_VULKAN_HPP

#include <vector>
#include <optional>
#include <memory>

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#include "klartraum/draw_component.hpp"

namespace klartraum {

class BackendVulkanImplementation;

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete() {
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

class BackendVulkan {
public:
    BackendVulkan();
    ~BackendVulkan();

    void initialize();

    void loop();

    void shutdown();

    VkDevice& getDevice();
    VkSwapchainKHR& getSwapChain();
    VkRenderPass& getRenderPass();
    
    VkPipeline& getGraphicsPipeline();
    VkQueue& getGraphicsQueue();


    VkFramebuffer& getFramebuffer(uint32_t imageIndex);
    VkExtent2D& getSwapChainExtent();

    

    QueueFamilyIndices getQueueFamilyIndices();

    static constexpr size_t MAX_FRAMES_IN_FLIGHT = 2;
    static constexpr uint32_t WIDTH = 800;
    static constexpr uint32_t HEIGHT = 600;

    void addDrawComponent(std::unique_ptr<DrawComponent> drawComponent);

private:
    BackendVulkanImplementation* impl;
    GLFWwindow* window;
    std::vector<std::unique_ptr<DrawComponent> > drawComponents;

    std::vector<VkFence> inFlightFences;
    std::vector<VkSemaphore> imageAvailableSemaphores;

    void createSyncObjects();
};

} // namespace klartraum

#endif // BACKEND_VULKAN_HPP