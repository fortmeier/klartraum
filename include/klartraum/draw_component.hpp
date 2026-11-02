// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_DRAW_COMPONENT_HPP
#define KLARTRAUM_DRAW_COMPONENT_HPP

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <memory>

#include "klartraum/vulkan_context.hpp"
#include "klartraum/computegraph/uniformbufferobject.hpp"
#include "klartraum/camera.hpp"
#include "klartraum/depth_attachment.hpp"

namespace klartraum {

typedef UniformBufferObject<CameraMVP> CameraUboType;

/**
 * @brief Something a RenderPass draws with its own graphics pipeline.
 *
 * Add it with RenderPass::addDrawComponent(). When the graph is compiled, the
 * render pass calls setNumberPaths() and initialize() with its Vulkan render pass
 * and camera uniform buffer, and then recordCommandBuffer() once per path.
 * Graphics pipelines should use pipelineDepthState() to honor the pass's optional
 * depth attachment and its independent testing/writing controls.
 */
class DrawComponent {
public:
    virtual void initialize(VulkanContext& vulkanContext, VkRenderPass& renderPass,
                            std::shared_ptr<CameraUboType> cameraUBO) {
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
    void setDepthState(bool attached, DepthState state) {
        depthAttached = attached;
        depthState = state;
    }

protected:
    VkPipelineDepthStencilStateCreateInfo pipelineDepthState() const {
        VkPipelineDepthStencilStateCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        info.depthTestEnable = depthAttached && (depthState.test || depthState.write);
        info.depthWriteEnable = depthAttached && depthState.write;
        info.depthCompareOp = depthState.test ? VK_COMPARE_OP_LESS_OR_EQUAL : VK_COMPARE_OP_ALWAYS;
        return info;
    }
    bool depthAttached = false;
    DepthState depthState{};
    uint32_t numberPaths = 0;
    VulkanContext* vulkanContext = nullptr;
    VkRenderPass* renderPass = nullptr;
    std::shared_ptr<CameraUboType> cameraUBO;
};

} // namespace klartraum

#endif // KLARTRAUM_DRAW_COMPONENT_HPP
