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

    // The number of paths of the graph the component is recorded into; per-path
    // resources are sized by it. RenderPass sets it before initialize(). A
    // graph may have fewer paths than there are swapchain images, e.g. a
    // single-path graph rendering into an OffscreenTarget.
    void setNumberPaths(uint32_t numberPaths) { this->numberPaths = numberPaths; }

protected:
    uint32_t numberPaths = 0;
    VulkanContext* vulkanContext = nullptr;
    VkRenderPass* renderPass = nullptr;
    std::shared_ptr<CameraUboType> cameraUBO;
};

} // namespace klartraum

#endif // DRAW_COMPONENT_HPP