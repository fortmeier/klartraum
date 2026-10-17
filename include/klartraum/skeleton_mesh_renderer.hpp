// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_SKELETON_MESH_RENDERER_HPP
#define KLARTRAUM_SKELETON_MESH_RENDERER_HPP

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/skeletongaussians.hpp"
#include "klartraum/draw_component.hpp"
#include "klartraum/vulkan_buffer.hpp"

namespace klartraum {

/// Push constants of shaders/skeleton/skeleton_mesh.{vert,frag}.
struct SkeletonMeshPushConstants {
    glm::vec2 resolution;
    uint32_t boneCount;
    uint32_t drawSpheres;  ///< 0: the bones' cylinders, 1: the joints' spheres
    float occlusionBias;   ///< see setOcclusionDepth()
    uint32_t useOcclusion; ///< 1: test against the occlusion depth buffer
};

/**
 * @brief Draws a skeleton as triangle meshes, a plain visualization helper.
 *
 * Every bone is a cylinder between its two joints, every joint a bone uses a
 * sphere, lit by how much the surface faces the camera. The joints are `vec4`
 * (e.g. CharacterController::getJoints(); `w` 0 hides a joint and its bones).
 * Colour, opacity, size and placement come from a SkeletonGaussianParams buffer
 * (e.g. HostValues<SkeletonGaussianParams>), read every run, so a skeleton drawn
 * as Gaussians and one drawn with this renderer take the same parameters:
 * `toScene` places the joints in the scene, followed by the render pass camera
 * (proj * view * model); `shape.x` is the bone radius in scene units, `shape.y`
 * the joint radius relative to it. Opacity 0 draws nothing.
 *
 * The render pass needs a depth buffer (RenderPass::setDepthAttachment()). The
 * joint buffer may be written by compute work in the same graph; place a
 * BufferToGraphicsBarrier between it and the render pass.
 */
class SkeletonMeshRenderer : public DrawComponent {
public:
    SkeletonMeshRenderer(std::shared_ptr<BufferElementInterface> joints,
                         std::vector<std::pair<uint32_t, uint32_t>> bones,
                         std::shared_ptr<BufferElementInterface> params, VkExtent2D targetExtent);
    ~SkeletonMeshRenderer();

    void initialize(VulkanContext& vulkanContext, VkRenderPass& renderPass,
                    std::shared_ptr<CameraUboType> cameraUBO) override;

    void recordCommandBuffer(VkCommandBuffer commandBuffer, VkFramebuffer framebuffer, uint32_t pathId) override;

    /**
     * @brief Hides the skeleton where a scene surface lies in front of it, e.g. of a Gaussian splatting scene.
     *
     * `depth` holds one NDC depth per pixel of the target (+inf: no surface),
     * seen with the render pass camera, such as getGaussianSplattingDepth(). A
     * fragment farther from the camera than the surface by more than
     * `relativeBias` of the surface's distance is dropped. Call before the graph
     * is compiled.
     */
    void setOcclusionDepth(std::shared_ptr<BufferElementInterface> depth, float relativeBias = 0.02f);

    /// Number of joints that get a sphere: those the bones use.
    uint32_t getSphereCount() const { return sphereCount; }

private:
    void createDescriptorSets();
    void createGraphicsPipeline();

    std::shared_ptr<BufferElementInterface> joints;
    std::shared_ptr<BufferElementInterface> params;
    std::shared_ptr<BufferElementInterface> occlusionDepth;
    float occlusionBias = 0.02f;
    VkExtent2D targetExtent;
    std::vector<uint32_t> indices; // bone pairs, then the joints with a sphere
    uint32_t boneCount;
    uint32_t sphereCount;

    std::unique_ptr<VulkanBuffer<uint32_t>> indexBuffer;
    std::unique_ptr<VulkanBuffer<glm::vec4>> meshBuffer; // cylinder vertices, then sphere vertices
    uint32_t cylinderVertexCount = 0;
    uint32_t sphereVertexCount = 0;

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;
};

} // namespace klartraum

#endif // KLARTRAUM_SKELETON_MESH_RENDERER_HPP
