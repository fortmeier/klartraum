#ifndef BACKEND_VULKAN_HPP
#define BACKEND_VULKAN_HPP

namespace klartraum {

class BackendVulkanImplentation;    

class BackendVulkan {
public:
    BackendVulkan();
    ~BackendVulkan();

    void initialize();

    void loop();

    void shutdown();

private:
    BackendVulkanImplentation* impl;
    // Private member variables and methods
};

} // namespace klartraum

#endif // BACKEND_VULKAN_HPP