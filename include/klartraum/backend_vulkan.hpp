#ifndef BACKEND_VULKAN_HPP
#define BACKEND_VULKAN_HPP

namespace klartraum {

class BackendVulkan {
public:
    BackendVulkan();
    ~BackendVulkan();

    void initialize();

    void loop();
    
    void shutdown();

private:
    // Private member variables and methods
};

} // namespace klartraum

#endif // BACKEND_VULKAN_HPP