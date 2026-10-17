// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_CHARACTER_MASK_HPP
#define KLARTRAUM_CHARACTER_MASK_HPP

#include <cstdint>
#include <memory>
#include <vector>

#include <glm/glm.hpp>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/draw_component.hpp"

namespace klartraum {

/**
 * @brief A shape of the character mask: a capsule between two COCO-18 joints, or with `disc` a disc around their
 * middle; its radius is in the joints' length unit (metres).
 */
struct CharacterMaskSegment {
    uint32_t first;
    uint32_t second;
    float radius;
    bool disc = false;
};

/// The shapes the character mask is made of: every OpenPose limb, the sides of the torso, and the head as a disc.
const std::vector<CharacterMaskSegment>& characterMaskSegments();

/// Push constants of shaders/skeleton/character_mask.{vert,frag}.
struct CharacterMaskPushConstants {
    glm::vec2 resolution;
    float margin;
    float occlusionBias;
    uint32_t useOcclusion;
};

/**
 * @brief Draws the region a character's body occupies, in white on black, e.g. as the inpainting mask for ControlNet.
 *
 * Reads the same COCO-18 joint buffer as OpenPoseSkeletonRenderer (one skeleton,
 * `vec4` in world space, `w` the visibility) and draws a capsule along every
 * segment of characterMaskSegments(): its radius is the segment's plus a margin,
 * projected with the render pass camera at each end, so the mask shrinks with
 * distance. The radii are in the joints' length unit before the camera's model
 * matrix, which may scale uniformly (e.g. a character in metres placed in a
 * smaller scene). A segment with an invisible joint is left out.
 *
 * With setOcclusionDepth() the mask is cut per pixel where a scene surface lies
 * in front of the body, exactly like OpenPoseSkeletonRenderer::setOcclusionDepth().
 */
class CharacterMaskRenderer : public DrawComponent {
public:
    /// Default margin around the body segments, in the joints' length unit.
    static constexpr float kDefaultMargin = 0.05f;

    CharacterMaskRenderer(std::shared_ptr<BufferElementInterface> joints, VkExtent2D targetExtent);
    ~CharacterMaskRenderer();

    void initialize(VulkanContext& vulkanContext, VkRenderPass& renderPass,
                    std::shared_ptr<CameraUboType> cameraUBO) override;

    void recordCommandBuffer(VkCommandBuffer commandBuffer, VkFramebuffer framebuffer, uint32_t pathId) override;

    /// Margin added to every segment's radius; set before the graph is compiled.
    void setMargin(float margin) { this->margin = margin; }
    float getMargin() const { return margin; }

    /// See OpenPoseSkeletonRenderer::setOcclusionDepth().
    void setOcclusionDepth(std::shared_ptr<BufferElementInterface> depth, float relativeBias = 0.02f);

private:
    void createDescriptorSets();
    void createGraphicsPipeline();

    std::shared_ptr<BufferElementInterface> joints;
    std::shared_ptr<BufferElementInterface> occlusionDepth;
    VkExtent2D targetExtent;
    float margin = kDefaultMargin;
    float occlusionBias = 0.02f;

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;
};

/**
 * @brief A compute element that writes the scene mask: white where `depth` holds a surface, black where it holds +inf.
 *
 * `depth` is a per-pixel depth buffer of the target's size, such as the output
 * of getGaussianSplattingDepth(); pass the element that writes it as `producer`
 * so the mask follows it in the graph. The target is left in
 * VK_IMAGE_LAYOUT_GENERAL.
 */
std::shared_ptr<ComputeGraphElement> createSceneMask(VulkanContext& vulkanContext,
                                                     std::shared_ptr<BufferElementInterface> depth,
                                                     std::shared_ptr<ImageViewSrc> target,
                                                     ComputeGraphElementPtr producer = nullptr);

} // namespace klartraum

#endif // KLARTRAUM_CHARACTER_MASK_HPP
