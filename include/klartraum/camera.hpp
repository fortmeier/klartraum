#ifndef KLARTRAUM_CAMERA_HPP
#define KLARTRAUM_CAMERA_HPP

#include <vector>

#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

#include "klartraum/backend_config.hpp"

namespace klartraum {

struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

class BackendVulkan;

class Camera {
public:
    Camera(BackendVulkan* backend);
    ~Camera();

    VkDescriptorSetLayout& getDescriptorSetLayout();

    void update(uint32_t currentImage);

private:
    BackendVulkan* backend;

    void createDescriptorSetLayout();
    void createUniformBuffers();
    
    VkDescriptorSetLayout descriptorSetLayout;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;

};

} // namespace klartraum

#endif // KLARTRAUM_CAMERA_HPP