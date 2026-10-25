// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_OPENPOSE_SKELETON_RENDERER_HPP
#define KLARTRAUM_OPENPOSE_SKELETON_RENDERER_HPP

#include <array>
#include <cstdint>
#include <memory>

#include <glm/glm.hpp>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/draw_component.hpp"

namespace klartraum {

/**
 * @brief The body-pose convention of ControlNet's OpenPose annotator (COCO-18 keypoints).
 *
 * ControlNet-OpenPose models are trained on pose maps drawn by `draw_bodypose`
 * (lllyasviel/ControlNet, controlnet_aux): over a black canvas, the 17 limbs as
 * filled ellipses in 60 % of their colour, then the 18 keypoints as filled
 * circles in full colour. Limb `i` takes colour `i` of the same table as joint `i`.
 */
namespace openpose {

enum Joint : uint32_t {
    Nose = 0,
    Neck,
    RightShoulder,
    RightElbow,
    RightWrist,
    LeftShoulder,
    LeftElbow,
    LeftWrist,
    RightHip,
    RightKnee,
    RightAnkle,
    LeftHip,
    LeftKnee,
    LeftAnkle,
    RightEye,
    LeftEye,
    RightEar,
    LeftEar,
};

constexpr uint32_t kJointCount = 18;
constexpr uint32_t kLimbCount = 17;

/// Joint pairs of the limbs, in drawing order.
constexpr std::array<std::array<uint32_t, 2>, kLimbCount> kLimbs = {{
    {Neck, RightShoulder},
    {Neck, LeftShoulder},
    {RightShoulder, RightElbow},
    {RightElbow, RightWrist},
    {LeftShoulder, LeftElbow},
    {LeftElbow, LeftWrist},
    {Neck, RightHip},
    {RightHip, RightKnee},
    {RightKnee, RightAnkle},
    {Neck, LeftHip},
    {LeftHip, LeftKnee},
    {LeftKnee, LeftAnkle},
    {Neck, Nose},
    {Nose, RightEye},
    {RightEye, RightEar},
    {Nose, LeftEye},
    {LeftEye, LeftEar},
}};

/// RGB colour of each joint.
constexpr std::array<std::array<uint8_t, 3>, kJointCount> kJointColors = {{
    {255, 0, 0},
    {255, 85, 0},
    {255, 170, 0},
    {255, 255, 0},
    {170, 255, 0},
    {85, 255, 0},
    {0, 255, 0},
    {0, 255, 85},
    {0, 255, 170},
    {0, 255, 255},
    {0, 170, 255},
    {0, 85, 255},
    {0, 0, 255},
    {85, 0, 255},
    {170, 0, 255},
    {255, 0, 255},
    {255, 0, 170},
    {255, 0, 85},
}};

/// RGB colour of each limb: int(0.6 * colour) of the joint colour with the same index.
constexpr std::array<std::array<uint8_t, 3>, kLimbCount> kLimbColors = [] {
    std::array<std::array<uint8_t, 3>, kLimbCount> colors{};
    for (uint32_t limb = 0; limb < kLimbCount; ++limb) {
        for (uint32_t channel = 0; channel < 3; ++channel) {
            colors[limb][channel] = static_cast<uint8_t>(kJointColors[limb][channel] * 6 / 10);
        }
    }
    return colors;
}();

} // namespace openpose

/// Push constants of shaders/skeleton/openpose.{vert,frag}.
struct OpenPosePushConstants {
    glm::vec2 resolution;
    float stickWidth;
    float jointRadius;
    uint32_t firstJoint;       ///< index of the skeleton's first joint in the joint buffer
    uint32_t drawJoints;       ///< 0: limbs, 1: joints
    float occlusionBias;       ///< relative distance a joint may lie behind the occlusion depth
    uint32_t useOcclusion;     ///< 1: test joints against the occlusion depth buffer
    uint32_t hideOccludedFace; ///< 1: hide face keypoints the head hides from the camera
};

/**
 * @brief Draws skeletons as ControlNet OpenPose pose maps (see klartraum::openpose).
 *
 * The joints come from a storage buffer of `vec4` in world space, kJointCount
 * per skeleton in COCO-18 order; `w` is the visibility (0 hides the joint and
 * its limbs). They are projected with the render pass camera (proj * view *
 * model). Joints behind the camera are hidden as well.
 *
 * The shapes follow the OpenCV drawing of the annotator pixel for pixel: limb
 * and circle centres are truncated to whole pixels, the limb angle to whole
 * degrees, and nothing is blended, later shapes cover earlier ones. Each
 * skeleton is drawn completely, limbs then joints, before the next one. The
 * annotator draws at 512 pixels with stick width and joint radius 4; by
 * default both are scaled by the shorter side of the target over 512.
 *
 * The joint buffer may be written by compute work in the same graph; place a
 * BufferToGraphicsBarrier between it and the render pass.
 */
class OpenPoseSkeletonRenderer : public DrawComponent {
public:
    OpenPoseSkeletonRenderer(std::shared_ptr<BufferElementInterface> joints, uint32_t skeletonCount,
                             VkExtent2D targetExtent);
    ~OpenPoseSkeletonRenderer();

    void initialize(VulkanContext& vulkanContext, VkRenderPass& renderPass,
                    std::shared_ptr<CameraUboType> cameraUBO) override;

    void recordCommandBuffer(VkCommandBuffer commandBuffer, VkFramebuffer framebuffer, uint32_t pathId) override;

    /**
     * @brief Hides the parts of the skeleton that lie behind a scene surface, e.g. of a Gaussian splatting scene.
     *
     * `depth` holds one NDC depth per pixel of the target (row-major, +inf where
     * there is no surface), seen through the same view and projection as the
     * render pass camera, such as the output of GaussianSplattingInterface::getDepth(). The
     * test is per pixel: a pixel of a limb or joint is dropped when the skeleton
     * there is farther from the camera than the surface by more than
     * `relativeBias` of the surface's distance (with a perspective projection;
     * otherwise NDC depths are compared). A limb's depth is interpolated along its
     * bone, perspective-correctly, so a limb passing behind the surface is cut
     * where it does. Call before the graph is compiled; the writer of the buffer
     * must come before the render pass in the graph.
     */
    void setOcclusionDepth(std::shared_ptr<BufferElementInterface> depth, float relativeBias = 0.02f);

    /**
     * @brief Hides the face keypoints the head itself hides from the camera, as an OpenPose detector would not find
     * them (on by default).
     *
     * The head's centre is the middle of the ears, it faces towards the nose.
     * Turned away from the camera, nose and eyes are hidden; an ear or eye on
     * the far side of the head (seen from the side) is hidden as well. Without
     * this, ControlNet sees a face in every pose map and draws the person facing
     * the camera even when walking away. Set before the graph is compiled.
     */
    void setHideOccludedFace(bool hide) { hideOccludedFace = hide; }

    /// Half the limb thickness in pixels (OpenCV's ellipse minor axis); set before the graph is compiled.
    void setStickWidth(float pixels) { stickWidth = pixels; }
    /// Radius of the joint discs in pixels; set before the graph is compiled.
    void setJointRadius(float pixels) { jointRadius = pixels; }

    float getStickWidth() const { return stickWidth; }
    float getJointRadius() const { return jointRadius; }

private:
    void createDescriptorSets();
    void createGraphicsPipeline();

    std::shared_ptr<BufferElementInterface> joints;
    std::shared_ptr<BufferElementInterface> occlusionDepth;
    float occlusionBias = 0.02f;
    bool hideOccludedFace = true;
    uint32_t skeletonCount;
    VkExtent2D targetExtent;
    float stickWidth;
    float jointRadius;

    VkDescriptorSetLayout jointDescriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> jointDescriptorSets;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;
};

} // namespace klartraum

#endif // KLARTRAUM_OPENPOSE_SKELETON_RENDERER_HPP
