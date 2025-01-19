#ifndef BACKEND_VULKAN_HPP
#define BACKEND_VULKAN_HPP

#include <vector>
#include <optional>

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

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

    QueueFamilyIndices getQueueFamilyIndices();

    static constexpr size_t MAX_FRAMES_IN_FLIGHT = 2;
    static constexpr uint32_t WIDTH = 800;
    static constexpr uint32_t HEIGHT = 600;

private:
    BackendVulkanImplementation* impl;
    GLFWwindow* window;
};

} // namespace klartraum

#endif // BACKEND_VULKAN_HPP