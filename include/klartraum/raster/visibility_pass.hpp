// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_RASTER_VISIBILITY_PASS_HPP
#define KLARTRAUM_RASTER_VISIBILITY_PASS_HPP

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

#include "klartraum/computegraph/computegraphelement.hpp"
#include "klartraum/vulkan_context.hpp"

namespace klartraum::raster {

/**
 * @brief Rasterizes a triangle mesh into a visibility buffer: per pixel the index of the nearest triangle plus one,
 * zero where no triangle covers the pixel centre.
 *
 * The hardware rasterizer decides visibility only; the differentiable passes recompute everything else from the
 * triangle index. The mesh and the cameras are read from storage buffers and the drawn view from a one-element
 * buffer, so an optimizer and a view-selection kernel can change them on the GPU between submissions without
 * re-recording. Both faces of every triangle are drawn; the depth test keeps the nearest one.
 *
 * Slots: 0 positions (four floats per vertex, xyz used), 1 triangles (three uint32 vertex indices each), 2 cameras
 * (a column-major 4x4 clip-from-world matrix per view, Vulkan clip space: y down, depth 0 to 1), 3 view index (one
 * uint32), 4 visibility (one uint32 per pixel, row-major, written).
 */
class VisibilityPass : public ComputeGraphElement {
public:
    VisibilityPass(VulkanContext& vulkanContext, uint32_t width, uint32_t height, uint32_t triangleCount);
    ~VisibilityPass();
    VisibilityPass(const VisibilityPass&) = delete;
    VisibilityPass& operator=(const VisibilityPass&) = delete;

    const char* getType() const override { return "VisibilityPass"; }
    void checkInput(ComputeGraphElementPtr input, int index = 0) override;
    void _setup(VulkanContext& vulkanContext, uint32_t numberPaths, VkQueue setupQueue = VK_NULL_HANDLE) override;
    void _record(VkCommandBuffer commandBuffer, uint32_t pathId) override;

    uint32_t width() const { return imageWidth; }
    uint32_t height() const { return imageHeight; }

private:
    // One path's render target, depth buffer and framebuffer.
    struct Target {
        VkImage color = VK_NULL_HANDLE, depth = VK_NULL_HANDLE;
        VkDeviceMemory colorMemory = VK_NULL_HANDLE, depthMemory = VK_NULL_HANDLE;
        VkImageView colorView = VK_NULL_HANDLE, depthView = VK_NULL_HANDLE;
        VkFramebuffer framebuffer = VK_NULL_HANDLE;
        VkDescriptorSet descriptors = VK_NULL_HANDLE;
    };

    void createImage(VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect, VkImage& image,
                     VkDeviceMemory& memory, VkImageView& view);
    void createRenderPass();
    void createPipeline();

    // The context the pass was created with; it outlives the pass.
    VulkanContext& context;
    uint32_t imageWidth, imageHeight, triangleCount;
    std::vector<Target> targets;
    VkRenderPass renderPass = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

} // namespace klartraum::raster

#endif // KLARTRAUM_RASTER_VISIBILITY_PASS_HPP
