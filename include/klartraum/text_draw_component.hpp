#pragma once

#include <string>
#include <vector>

#include "klartraum/draw_component.hpp"
#include "klartraum/simple_text_renderer.hpp"

namespace klartraum {

// Renders a single line of debug text (e.g. an FPS counter) as textured
// quads sampled from a procedurally generated glyph atlas.
//
// The engine pre-records command buffers once at setup time and resubmits
// them unchanged every frame, so the draw call itself (vertex/index counts)
// must stay constant. To allow the text to change at runtime, this component
// reserves vertex-buffer space for a fixed `maxCharacters` and pre-builds an
// index buffer for that many glyph quads; `setText()` rewrites the mapped
// vertex buffers in place, padding unused glyph slots with zero-alpha,
// zero-area quads so they contribute nothing to the image.
class TextDrawComponent : public DrawComponent {
public:
    explicit TextDrawComponent(uint32_t maxCharacters = 256);
    ~TextDrawComponent();

    void initialize(VulkanContext& vulkanContext, VkRenderPass& renderPass,
                    std::shared_ptr<CameraUboType> cameraUBO) override;
    void recordCommandBuffer(VkCommandBuffer commandBuffer, VkFramebuffer framebuffer, uint32_t pathId) override;

    // Replaces the rendered text. `text` is truncated to `maxCharacters`.
    // (x, y) is the top-left corner in screen pixels, `scale` multiplies the
    // glyphs' native pixel size, and (r, g, b, a) tints every glyph.
    void setText(const std::string& text, float x, float y, float scale, float r, float g, float b, float a);

private:
    void createGlyphAtlasTexture();
    void createDescriptorSetLayoutAndSet();
    void createGraphicsPipeline();
    void createIndexBuffer();
    void createVertexBuffers();
    void uploadVertices();

    const uint32_t maxCharacters;

    SimpleTextRenderer textRenderer;
    uint32_t currentCharacterCount = 0;

    // Glyph atlas texture (created once, shared across all paths).
    VkImage atlasImage = VK_NULL_HANDLE;
    VkDeviceMemory atlasImageMemory = VK_NULL_HANDLE;
    VkImageView atlasImageView = VK_NULL_HANDLE;
    VkSampler atlasSampler = VK_NULL_HANDLE;

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;

    // Index buffer: a fixed {0,1,2, 2,3,0} pattern repeated (offset by 4 per
    // glyph) for `maxCharacters` quads. Content never changes at runtime.
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory indexBufferMemory = VK_NULL_HANDLE;

    // One persistently-mapped, host-visible vertex buffer per swapchain path
    // (mirrors UniformBufferObject's per-path update pattern), each sized for
    // `maxCharacters` glyph quads (4 vertices each).
    std::vector<VkBuffer> vertexBuffers;
    std::vector<VkDeviceMemory> vertexBufferMemories;
    std::vector<void*> vertexBuffersMapped;
};

}  // namespace klartraum
