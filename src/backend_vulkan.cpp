#include <stdexcept>
#include <vector>
#include <optional>
#include <iostream>
#include <algorithm>
#include <set>
#include <fstream>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <vulkan/vulkan.h>

#include "klartraum/backend_vulkan.hpp"
#include "klartraum/interface_camera.hpp"

#include "klartraum/vulkan_kernel.hpp"

namespace klartraum {


BackendVulkan::BackendVulkan() : impl(nullptr), old_mouse_x(0), old_mouse_y(0)
{
}

BackendVulkan::~BackendVulkan()
{
    auto device = impl->device;
    
    vkDestroyCommandPool(device, commandPool, nullptr);
    
    for (size_t i = 0; i < config.MAX_FRAMES_IN_FLIGHT; i++)
    {
        vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
        vkDestroyFence(device, inFlightFences[i], nullptr);
    }
    
    for (size_t i = 0; i < getConfig().MAX_FRAMES_IN_FLIGHT; i++)
    {
        vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
    }

    delete impl;


}

static double scrollYAccum = 0.0;
static double scrollXAccum = 0.0;

static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    scrollXAccum += xoffset;
    scrollYAccum += yoffset;
}


void BackendVulkan::initialize() {
    // Initialize Vulkan
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    window = glfwCreateWindow(config.WIDTH, config.HEIGHT, "Vulkan", nullptr, nullptr);

    glfwSetScrollCallback(window, scroll_callback);

    impl = new VulkanKernel();

    if (glfwCreateWindowSurface(impl->instance, window, nullptr, &surface) != VK_SUCCESS) {
        throw std::runtime_error("failed to create window surface!");
    }

    impl->initialize(surface);

    
    camera = std::make_unique<Camera>(this);

    createCommandPool();
    createCommandBuffers();

    createSyncObjects();

}


void BackendVulkan::loop() {
    auto device = impl->device;

    uint32_t currentFrame = 0;
    // Main loop
    std::vector<VkSemaphore> signalSemaphores;
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        uint32_t imageIndex;
        vkAcquireNextImageKHR(device, impl->swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

        processGLFWEvents();

        auto& framebuffer = getFramebuffer(imageIndex);
        auto& commandBuffer = commandBuffers[currentFrame];
        
        beginRenderPass(currentFrame, framebuffer);
        
        for(auto &drawComponent : drawComponents) {
            auto imageAvailableSemaphore = imageAvailableSemaphores[currentFrame];
            drawComponent->draw(currentFrame, commandBuffer, framebuffer, imageAvailableSemaphore);
        }

        signalSemaphores.push_back(renderFinishedSemaphores[currentFrame]);
        endRenderPass(currentFrame);

        if(interfaceCamera != nullptr && camera != nullptr)
        {
            while (!eventQueue.empty()) {
                auto event = std::move(eventQueue.front());
                eventQueue.pop();
                interfaceCamera->onEvent(*event);
            }
            interfaceCamera->update(*camera);
        }

        camera->update(currentFrame);

        if (vkQueueSubmit(impl->graphicsQueue, 0, nullptr, inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit inFlightFence");
        }

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        presentInfo.waitSemaphoreCount = signalSemaphores.size();
        presentInfo.pWaitSemaphores = signalSemaphores.data();

        VkSwapchainKHR swapChains[] = { impl->swapChain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;

        presentInfo.pResults = nullptr; // Optional

        
        vkQueuePresentKHR(impl->presentQueue, &presentInfo);
            
        currentFrame = (currentFrame + 1) % config.MAX_FRAMES_IN_FLIGHT;
    }

}

void BackendVulkan::shutdown() {
    camera = nullptr;

    vkDestroySurfaceKHR(impl->instance, surface, nullptr);


    glfwDestroyWindow(window);
    glfwTerminate();
}


void BackendVulkan::processGLFWEvents() {
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    int new_mouse_x = (int)xpos;
    int new_mouse_y = (int)ypos;
    if(new_mouse_x != old_mouse_x || new_mouse_y != old_mouse_y) {
        int dx = new_mouse_x - old_mouse_x;
        int dy = new_mouse_y - old_mouse_y;
        auto event = std::make_unique<EventMouseMove>(new_mouse_x, new_mouse_y, dx, dy);
        eventQueue.push(std::move(event));
        old_mouse_x = new_mouse_x;
        old_mouse_y = new_mouse_y;
    }

    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS && !leftButtonDown) {
        auto event = std::make_unique<EventMouseButton>(EventMouseButton::Button::Left, EventMouseButton::Action::Press);
        eventQueue.push(std::move(event));
        leftButtonDown = true;
    }
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_RELEASE && leftButtonDown) {
        auto event = std::make_unique<EventMouseButton>(EventMouseButton::Button::Left, EventMouseButton::Action::Release);
        eventQueue.push(std::move(event));
        leftButtonDown = false;
    }

    if (scrollXAccum != 0.0 || scrollYAccum != 0.0) {
        auto event = std::make_unique<EventMouseScroll>(scrollXAccum, scrollYAccum);
        eventQueue.push(std::move(event));
        scrollXAccum = 0.0;
        scrollYAccum = 0.0;
    }

}

VkDevice& BackendVulkan::getDevice() {
    if(impl == nullptr)
    {
        throw std::runtime_error("BackendVulkan::getDevice() called before initialize()");
    }
    return impl->device;
}

VkSwapchainKHR& BackendVulkan::getSwapChain()
{
    if(impl == nullptr)
    {
        throw std::runtime_error("BackendVulkan::getDevice() called before initialize()");
    }
    return impl->swapChain;
}

VkRenderPass& BackendVulkan::getRenderPass()
{
    if(impl == nullptr)
    {
        throw std::runtime_error("BackendVulkan::getDevice() called before initialize()");
    }
    return impl->renderPass;
}

VkQueue& BackendVulkan::getGraphicsQueue()
{
    if(impl == nullptr)
    {
        throw std::runtime_error("BackendVulkan::getDevice() called before initialize()");
    }
    return impl->graphicsQueue;
}

VkFramebuffer& BackendVulkan::getFramebuffer(uint32_t imageIndex){
    if(impl == nullptr)
    {
        throw std::runtime_error("BackendVulkan::getDevice() called before initialize()");
    }
    if (imageIndex >= impl->swapChainFramebuffers.size()) {
        throw std::runtime_error("Invalid image index!");
    }
    return impl->swapChainFramebuffers[imageIndex];
}

VkExtent2D& BackendVulkan::getSwapChainExtent()
{
    if(impl == nullptr)
    {
        throw std::runtime_error("BackendVulkan::getDevice() called before initialize()");
    }
    return impl->swapChainExtent;
}

QueueFamilyIndices BackendVulkan::getQueueFamilyIndices() {
    if(impl == nullptr)
    {
        throw std::runtime_error("BackendVulkan::getQueueFamilyIndices() called before initialize()");
    }    
    QueueFamilyIndices queueFamilyIndices = impl->findQueueFamiliesPhysicalDevice();
    return queueFamilyIndices;
}

Camera& BackendVulkan::getCamera()
{
    if(impl == nullptr)
    {
        throw std::runtime_error("BackendVulkan::getCamera() called before initialize()");
    }    
    return *camera;
}

void BackendVulkan::setInterfaceCamera(std::shared_ptr<InterfaceCamera> camera)
{
    if(impl == nullptr)
    {
        throw std::runtime_error("BackendVulkan::setInterfaceCamera() called before initialize()");
    }    
    this->interfaceCamera = camera;
}

BackendConfig& BackendVulkan::getConfig()
{
    return config;
}

void BackendVulkan::addDrawComponent(std::unique_ptr<DrawComponent> drawComponent)
{
    drawComponents.push_back(std::move(drawComponent));
}

void BackendVulkan::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    auto device = impl->device;

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

void BackendVulkan::createSyncObjects()
{
    imageAvailableSemaphores.resize(config.MAX_FRAMES_IN_FLIGHT);
    inFlightFences.resize(config.MAX_FRAMES_IN_FLIGHT);

    auto device = impl->device;

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < config.MAX_FRAMES_IN_FLIGHT; i++)
    {
        if (
            vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create semaphores!");
        }
    }

    renderFinishedSemaphores.resize(getConfig().MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < getConfig().MAX_FRAMES_IN_FLIGHT; i++)
    {
        if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create semaphores!");
        }
    }
}

uint32_t BackendVulkan::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    if(impl == nullptr)
    {
        throw std::runtime_error("BackendVulkan::findMemoryType() called before initialize()");
    }   

    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(impl->physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type!");
}

void BackendVulkan::beginRenderPass(uint32_t currentFrame, VkFramebuffer& framebuffer) {
    VkCommandBuffer& commandBuffer = commandBuffers[currentFrame];

    vkResetCommandBuffer(commandBuffer, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = 0; // Optional
    beginInfo.pInheritanceInfo = nullptr; // Optional

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording command buffer!");
    }

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = impl->renderPass;
    renderPassInfo.framebuffer = framebuffer;

    renderPassInfo.renderArea.offset = { 0, 0 };
    renderPassInfo.renderArea.extent = getSwapChainExtent();

    VkClearValue clearColor = { {{0.0f, 0.0f, 0.0f, 1.0f}} };
    renderPassInfo.clearValueCount = 1;
    renderPassInfo.pClearValues = &clearColor;

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);    
}

void BackendVulkan::endRenderPass(uint32_t currentFrame) {
    VkCommandBuffer& commandBuffer = commandBuffers[currentFrame];

    vkCmdEndRenderPass(commandBuffer);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
    }

    VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &renderFinishedSemaphores[currentFrame];

    if (vkQueueSubmit(impl->graphicsQueue, 1, &submitInfo, nullptr) != VK_SUCCESS) {
        throw std::runtime_error("failed to submit draw command buffer!");
    }    
}

void BackendVulkan::createCommandPool() {

    auto device = impl->device;

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = getQueueFamilyIndices().graphicsFamily.value();

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }    
}

void BackendVulkan::createCommandBuffers() {
    auto device = impl->device;

    commandBuffers.resize(getConfig().MAX_FRAMES_IN_FLIGHT);
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers.size());

    if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) {
        throw std::runtime_error("failed to allocate command buffers!");
    }
}

} // namespace klartraum