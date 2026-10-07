// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_DRAW_BASICS_HPP
#define KLARTRAUM_DRAW_BASICS_HPP

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vector>
#include <string>

#include "klartraum/draw_component.hpp"

namespace klartraum {

/**
 * @brief Shapes that DrawBasics can draw.
 */
enum class DrawBasicsType { Triangle, Cube, Axes };

/**
 * @brief Draws a built-in shape (a triangle, a cube or the coordinate axes) with the camera of its render pass.
 */
class DrawBasics : public DrawComponent {
public:
    DrawBasics(DrawBasicsType type);
    ~DrawBasics();

    virtual void initialize(VulkanContext& vulkanContext, VkRenderPass& renderpass,
                            std::shared_ptr<CameraUboType> cameraUBO) override;

    void recordCommandBuffer(VkCommandBuffer commandBuffer, VkFramebuffer framebuffer, uint32_t pathId) override;

private:
    void createGraphicsPipeline();
    void createSyncObjects();
    void createVertexBuffer();

    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;

    DrawBasicsType type;
};

} // namespace klartraum

#endif // KLARTRAUM_DRAW_BASICS_HPP
