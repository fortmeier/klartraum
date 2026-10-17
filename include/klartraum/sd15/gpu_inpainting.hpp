// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_SD15_GPU_INPAINTING_HPP
#define KLARTRAUM_SD15_GPU_INPAINTING_HPP

#include <cstdint>
#include <memory>
#include <vector>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/generalcomputation.hpp"
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

} // namespace klartraum

#endif // KLARTRAUM_SD15_GPU_INPAINTING_HPP
