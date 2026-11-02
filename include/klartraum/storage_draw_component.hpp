// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_STORAGE_DRAW_COMPONENT_HPP
#define KLARTRAUM_STORAGE_DRAW_COMPONENT_HPP

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/draw_component.hpp"
#include "klartraum/rendering/vulkan_gaussian_splatting_types.hpp"

namespace klartraum {

/// How a StorageDrawComponent draws.
struct StorageDrawConfig {
    std::string vertexShader;
    std::string fragmentShader;
    VkExtent2D extent{0, 0};
    bool depthTest = true; ///< needs a RenderPass with setDepthAttachment(true)
    bool depthWrite = true;
    bool blend = false; ///< premultiplied "over" blending
    VkCullModeFlags cullMode = VK_CULL_MODE_NONE;
    uint32_t pushConstantSize = 0; ///< bytes, for the vertex and fragment stages
};

/**
 * @brief Draws geometry that its shaders build from storage buffers, without vertex buffers.
 *
 * The shaders see the camera UBO at set 0, binding 0, and the storage buffers
 * at set 1, bindings 0, 1, ... in the order given. Each draw is a non-indexed
 * triangle list, either with fixed counts or with arguments that compute
 * shaders wrote (a VkDrawIndirectCommand at `offset` in `indirect`), and its
 * own push constants. Everything is recorded once.
 */
class StorageDrawComponent : public DrawComponent {
public:
    struct Draw {
        BufferRef indirect; ///< no element: draw vertexCount x instanceCount
        VkDeviceSize offset = 0;
        uint32_t vertexCount = 0;
        uint32_t instanceCount = 1;
        std::vector<uint8_t> pushConstants;
    };

    StorageDrawComponent(StorageDrawConfig config, std::vector<BufferRef> buffers);
    ~StorageDrawComponent();

    void addDraw(Draw draw) { draws.push_back(std::move(draw)); }

    template <typename P>
    static std::vector<uint8_t> bytes(const P& value) {
        std::vector<uint8_t> data(sizeof(P));
        std::memcpy(data.data(), &value, sizeof(P));
        return data;
    }

    void initialize(VulkanContext& vulkanContext, VkRenderPass& renderPass,
                    std::shared_ptr<CameraUboType> cameraUBO) override;
    void recordCommandBuffer(VkCommandBuffer commandBuffer, VkFramebuffer framebuffer, uint32_t pathId) override;

private:
    void createDescriptors();
    void createPipeline();

    StorageDrawConfig config;
    std::vector<BufferRef> buffers;
    std::vector<Draw> draws;

    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
};

/// `buffer`, or a buffer of one float for a binding that is not used.
inline BufferRef bufferOrPlaceholder(VulkanContext& vulkanContext, const BufferRef& buffer) {
    if (buffer.element) {
        return buffer;
    }
    return {std::make_shared<BufferElementSinglePath<VulkanBuffer<float>>>(vulkanContext, 1u), -1};
}

} // namespace klartraum

#endif // KLARTRAUM_STORAGE_DRAW_COMPONENT_HPP
