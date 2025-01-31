#ifndef BACKEND_VULKAN_HPP
#define BACKEND_VULKAN_HPP

#include <vector>
#include <queue>
#include <optional>
#include <memory>

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#include "klartraum/backend_config.hpp"
#include "klartraum/draw_component.hpp"
#include "klartraum/camera.hpp"
#include "klartraum/interface_camera.hpp"
#include "klartraum/events.hpp"

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

    void processGLFWEvents();

    VkDevice& getDevice();
    VkSwapchainKHR& getSwapChain();
    VkRenderPass& getRenderPass();
    
    VkQueue& getGraphicsQueue();


    VkFramebuffer& getFramebuffer(uint32_t imageIndex);
    VkExtent2D& getSwapChainExtent();

    

    QueueFamilyIndices getQueueFamilyIndices();


    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

    Camera& getCamera();
    void setInterfaceCamera(std::shared_ptr<InterfaceCamera> camera);

    BackendConfig& getConfig();

    void addDrawComponent(std::unique_ptr<DrawComponent> drawComponent);

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);


private:
    BackendConfig config;
    BackendVulkanImplementation* impl;
    GLFWwindow* window;

    std::shared_ptr<Camera> camera;
    std::shared_ptr<InterfaceCamera> interfaceCamera;

    std::queue<std::unique_ptr<Event> > eventQueue;

    std::vector<std::unique_ptr<DrawComponent> > drawComponents;

    int old_mouse_x;
    int old_mouse_y;

    std::vector<VkFence> inFlightFences;
    std::vector<VkSemaphore> imageAvailableSemaphores;

    void createSyncObjects();
};

} // namespace klartraum

#endif // BACKEND_VULKAN_HPP