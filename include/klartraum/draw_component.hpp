#ifndef DRAW_COMPONENT_HPP
#define DRAW_COMPONENT_HPP

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <memory>

#include "klartraum/vulkan_context.hpp"
#include "klartraum/computegraph/uniformbufferobject.hpp"
#include "klartraum/camera.hpp"

namespace klartraum {

typedef UniformBufferObject<CameraMVP> CameraUboType;

class DrawComponent {
public:
    virtual void initialize(VulkanContext& vulkanContext, VkRenderPass& renderPass, std::shared_ptr<CameraUboType> cameraUBO)
    {
        this->vulkanContext = &vulkanContext;
        this->renderPass = &renderPass;
        this->cameraUBO = cameraUBO;
    }

    virtual void recordCommandBuffer(VkCommandBuffer commandBuffer, VkFramebuffer framebuffer, uint32_t pathId) = 0;

protected:
    VulkanContext* vulkanContext = nullptr;
    VkRenderPass* renderPass = nullptr;
    std::shared_ptr<CameraUboType> cameraUBO;
};

} // namespace klartraum

#endif // DRAW_COMPONENT_HPP