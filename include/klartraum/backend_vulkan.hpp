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

#include "klartraum/vulkan_kernel.hpp"

namespace klartraum {

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

    VkSurfaceKHR surface;

    std::shared_ptr<Camera> camera;
    std::shared_ptr<InterfaceCamera> interfaceCamera;

    std::queue<std::unique_ptr<Event> > eventQueue;

    std::vector<std::unique_ptr<DrawComponent> > drawComponents;

    int old_mouse_x = 0;
    int old_mouse_y = 0;

    bool leftButtonDown = false;
    bool rightButtonDown = false;

    std::vector<VkFence> inFlightFences;
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;  

    void createSyncObjects();

    void beginRenderPass(uint32_t currentFrame, VkFramebuffer& framebuffer);
    void endRenderPass(uint32_t currentFrame);

    void createCommandPool();
    void createCommandBuffers();

    VkCommandPool commandPool;
    std::vector<VkCommandBuffer> commandBuffers;    
};

} // namespace klartraum

#endif // BACKEND_VULKAN_HPP