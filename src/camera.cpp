#include <stdexcept>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <chrono>

#include "klartraum/camera.hpp"
#include "klartraum/backend_vulkan.hpp"


namespace klartraum {

Camera::Camera(BackendVulkan* backend) : backend(backend) {
    createDescriptorSetLayout();
    createUniformBuffers();
}

Camera::~Camera() {
    auto config = backend->getConfig();
    auto device = backend->getDevice();

    for (size_t i = 0; i < config.MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroyBuffer(device, uniformBuffers[i], nullptr);
        vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
    }

    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
}

VkDescriptorSetLayout& Camera::getDescriptorSetLayout() {
     return descriptorSetLayout;
}

void Camera::update(uint32_t currentImage) {
    auto swapChainExtent = backend->getSwapChainExtent();

    static auto startTime = std::chrono::high_resolution_clock::now();
    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

    UniformBufferObject ubo{};
    ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));

    ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));

    ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float) swapChainExtent.height, 0.1f, 10.0f);

    // Vulkan has inverted Y coordinates compared to OpenGL
    ubo.proj[1][1] *= -1;

    memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));

}

void Camera::createDescriptorSetLayout() {
    auto device = backend->getDevice();

    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;

    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &uboLayoutBinding;

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor set layout!");
    }
}

void Camera::createUniformBuffers() {
    auto config = backend->getConfig();
    auto device = backend->getDevice();

    VkDeviceSize bufferSize = sizeof(UniformBufferObject);

    uniformBuffers.resize(config.MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMemory.resize(config.MAX_FRAMES_IN_FLIGHT);
    uniformBuffersMapped.resize(config.MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < config.MAX_FRAMES_IN_FLIGHT; i++) {
        backend->createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);

        vkMapMemory(device, uniformBuffersMemory[i], 0, bufferSize, 0, &uniformBuffersMapped[i]);
    }
}

} // namespace klartraum