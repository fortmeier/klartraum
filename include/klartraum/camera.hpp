#ifndef KLARTRAUM_CAMERA_HPP
#define KLARTRAUM_CAMERA_HPP

#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

#include "klartraum/backend_config.hpp"


namespace klartraum {

struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

class VulkanKernel;

class Camera {
public:
    Camera(VulkanKernel* vulkanKernel);
    ~Camera();

    VkDescriptorSetLayout& getDescriptorSetLayout();
    std::vector<VkDescriptorSet>& getDescriptorSets();

    void update(uint32_t currentImage);

    UniformBufferObject ubo;

private:
    VulkanKernel* vulkanKernel;

    void createDescriptorSetLayout();
    void createUniformBuffers();
    void createDescriptorPool();
    void createDescriptorSets();

    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<void*> uniformBuffersMapped;

};

} // namespace klartraum

#endif // KLARTRAUM_CAMERA_HPP