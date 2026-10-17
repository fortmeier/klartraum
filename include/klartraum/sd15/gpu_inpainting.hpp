// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_SD15_GPU_INPAINTING_HPP
#define KLARTRAUM_SD15_GPU_INPAINTING_HPP

#include <cstdint>
#include <memory>
#include <vector>

#include <glm/glm.hpp>

#include "klartraum/camera.hpp"
#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/rendergraphelement.hpp"
#include "klartraum/sd15/lcm_scheduler.hpp"
#include "klartraum/vulkan_buffer.hpp"

namespace klartraum {

/**
 * @brief Graph elements for SD1.5 latent inpainting with LCM entirely on the GPU.
 *
 * The CPU loop of sampleInpainted() becomes a chain of graphs (see GraphChain):
 *
 *     encoder → LCM init → [denoiser → LCM step 0] → … → [denoiser → LCM step n-1] → decoder
 *
 * The LCM elements read and write the denoiser's sample and timestep tensors in
 * place, so the denoiser graph is compiled once and appears in the chain once per
 * step. Their float operations follow LcmScheduler::step() and sampleInpainted().
 */

/**
 * @brief Buffers shared by the LCM elements.
 *
 * The known latent and the mask are data of one frame and have a buffer per path, so frames of different paths can
 * be in flight at once; they exist once the first graph using them is compiled. The noise does not change and is
 * shared by all paths. Review: frames in flight have not been exercised with the full SD1.5 chain yet (it runs one
 * path and waits for each frame).
 */
struct LcmInpaintingBuffers {
    using Floats = BufferElement<VulkanBuffer<float>>;
    using SharedFloats = BufferElementSinglePath<VulkanBuffer<float>>;

    std::shared_ptr<Floats> known;       ///< the clean latent kept outside the mask [channels * plane]
    std::shared_ptr<Floats> mask;        ///< per latent pixel: 1 generate, 0 keep [plane]
    std::shared_ptr<SharedFloats> noise; ///< the noise of every step [steps * channels * plane]
    uint32_t channels = 0;
    uint32_t plane = 0;

    /// Allocates the buffers and uploads `noise` (one latent per step of `scheduler`).
    static LcmInpaintingBuffers create(VulkanContext& vulkanContext, uint32_t channels, uint32_t plane,
                                       const std::vector<std::vector<float>>& noise);

    uint32_t count() const { return channels * plane; }
};

struct LcmInitPushConstants {
    uint32_t count;
    uint32_t plane;
    float scale;
    float sqrtAlpha;
    float sqrtOneMinusAlpha;
    uint32_t timestepLow;
    uint32_t timestepHigh;
};

struct LcmStepPushConstants {
    uint32_t count;
    uint32_t plane;
    uint32_t next;
    uint32_t last;
    float sqrtAlpha;
    float sqrtOneMinusAlpha;
    float sqrtAlphaNext;
    float sqrtOneMinusAlphaNext;
    float cSkip;
    float cOut;
    float sqrtAlphaKnown;
    float sqrtOneMinusAlphaKnown;
    float scale;
    uint32_t timestepLow;
    uint32_t timestepHigh;
};

/**
 * @brief Starts LCM inpainting: known = encoded * scale, the start latent in `latents`, the first timestep.
 *
 * `encoded` is the encoder's latent, `latents` the denoiser's sample, `timestep` its int64 timestep.
 */
std::shared_ptr<GeneralComputation<LcmInitPushConstants>>
createLcmInit(VulkanContext& vulkanContext, const LcmScheduler& scheduler, float scale, ComputeGraphElementPtr encoded,
              const LcmInpaintingBuffers& buffers, ComputeGraphElementPtr latents, ComputeGraphElementPtr timestep);

/**
 * @brief Step `index` of LCM inpainting, in place on `latents`, after the denoiser wrote `prediction`.
 *
 * Before the last step it sets `timestep` to the next step's; after the last step it writes
 * latents / scale to `decoderInput`.
 */
std::shared_ptr<GeneralComputation<LcmStepPushConstants>>
createLcmStep(VulkanContext& vulkanContext, const LcmScheduler& scheduler, size_t index, float scale,
              ComputeGraphElementPtr latents, ComputeGraphElementPtr prediction, const LcmInpaintingBuffers& buffers,
              ComputeGraphElementPtr timestep, ComputeGraphElementPtr decoderInput);

// ---- The close-up around the character (closeup_inpainting.hpp on the GPU) ----

/**
 * Images the elements below read are rendered by other elements (splatting, render passes) and are in `layout`
 * (an OffscreenTarget: TRANSFER_SRC_OPTIMAL); the elements move them to GENERAL. An image read a second time in
 * the same frame is already in GENERAL.
 */

struct ImageTensorRangePushConstants {
    uint32_t width;
    uint32_t height;
    float low;
    float high;
};

/// bgraToTensor(): the RGB of `image` into `tensor` [3, height, width], from `low` to `high`.
std::shared_ptr<GeneralComputation<ImageTensorRangePushConstants>>
createImageToTensor(VulkanContext& vulkanContext, ComputeGraphElementPtr image, VkImageLayout layout,
                    ComputeGraphElementPtr tensor, VkExtent2D extent, float low, float high);

struct LatentMaskPushConstants {
    uint32_t width;
    uint32_t height;
    uint32_t factor;
    uint32_t dilation;
};

/// closeUpLatentMask() of the masks `marginImage` and `bodyImage` (red channel set) into `latentMask`.
std::shared_ptr<GeneralComputation<LatentMaskPushConstants>>
createCloseUpLatentMask(VulkanContext& vulkanContext, ComputeGraphElementPtr marginImage, VkImageLayout marginLayout,
                        ComputeGraphElementPtr bodyImage, VkImageLayout bodyLayout, ComputeGraphElementPtr latentMask,
                        VkExtent2D extent, uint32_t factor = 8, uint32_t dilation = 1);

struct CloseUpWindowPushConstants {
    glm::mat4 characterToScene;
    float aspect;
    uint32_t jointCount;
};

/**
 * @brief CloseUpWindow::update() and the cameras of the close-up on the GPU.
 *
 * Computes the window from the joints and the full view's camera, writes it to the state buffer (centre.xy,
 * halfExtent.xy, halfY, initialized) and copies the scene and skeleton cameras into their uniform buffers, which
 * the close-up's graphs read. The state carries the window's smoothing from frame to frame, so frames must run in
 * order, as for CharacterController. Review: the state is shared by all paths.
 */
class CloseUpWindowElement : public GeneralComputation<CloseUpWindowPushConstants> {
public:
    using GeneralComputation<CloseUpWindowPushConstants>::GeneralComputation;

    void setCameras(std::shared_ptr<CameraUboType> scene, std::shared_ptr<CameraUboType> skeleton) {
        sceneCamera = std::move(scene);
        skeletonCamera = std::move(skeleton);
    }

    void _record(VkCommandBuffer commandBuffer, uint32_t pathId) override;

    const char* getType() const override { return "CloseUpWindow"; }

private:
    std::shared_ptr<CameraUboType> sceneCamera;
    std::shared_ptr<CameraUboType> skeletonCamera;
};

struct CloseUpWindowElements {
    std::shared_ptr<CloseUpWindowElement> element;
    /// centre.xy, halfExtent.xy, halfY, initialized; zero: not initialized
    std::shared_ptr<BufferElementSinglePath<VulkanBuffer<float>>> state;
};

/**
 * @brief The window around `jointCount` joints (vec4, character space) seen by `fullCamera`.
 *
 * `sceneCamera` and `skeletonCamera` depend on the element, so elements reading them run after it.
 */
CloseUpWindowElements createCloseUpWindow(VulkanContext& vulkanContext, ComputeGraphElementPtr joints,
                                          uint32_t jointCount, std::shared_ptr<CameraUboType> fullCamera,
                                          const glm::mat4& characterToScene, float aspect,
                                          std::shared_ptr<CameraUboType> sceneCamera,
                                          std::shared_ptr<CameraUboType> skeletonCamera);

struct CloseUpPastePushConstants {
    uint32_t size;
};

/// pasteGenerated(): `generated` into `sceneImage` through `bodyImage`; `closeUp` holds B, G, R, alpha per pixel.
std::shared_ptr<GeneralComputation<CloseUpPastePushConstants>>
createCloseUpPaste(VulkanContext& vulkanContext, ComputeGraphElementPtr generated, ComputeGraphElementPtr sceneImage,
                   VkImageLayout sceneLayout, ComputeGraphElementPtr bodyImage, VkImageLayout bodyLayout,
                   ComputeGraphElementPtr closeUp, uint32_t size);

struct CloseUpCompositePushConstants {
    uint32_t width;
    uint32_t height;
    uint32_t size;
};

/// compositeCloseUp(): `closeUp` into `sceneImage` inside `maskImage`, at the window in `state`; `frame` BGRA.
std::shared_ptr<GeneralComputation<CloseUpCompositePushConstants>>
createCloseUpComposite(VulkanContext& vulkanContext, ComputeGraphElementPtr sceneImage, VkImageLayout sceneLayout,
                       ComputeGraphElementPtr maskImage, VkImageLayout maskLayout, ComputeGraphElementPtr closeUp,
                       ComputeGraphElementPtr state, ComputeGraphElementPtr frame, VkExtent2D extent, uint32_t size);

} // namespace klartraum

#endif // KLARTRAUM_SD15_GPU_INPAINTING_HPP
