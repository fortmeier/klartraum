#ifndef BACKEND_VULKAN_HPP
#define BACKEND_VULKAN_HPP

#include <vulkan/vulkan.h>

namespace klartraum {

class BackendVulkanImplentation;    

class BackendVulkan {
public:
    BackendVulkan();
    ~BackendVulkan();

    void initialize();

    void loop();

    void shutdown();

    VkDevice& getDevice();

    static const size_t MAX_FRAMES_IN_FLIGHT = 2;

private:
    BackendVulkanImplentation* impl;
    // Private member variables and methods
};

} // namespace klartraum

#endif // BACKEND_VULKAN_HPP