// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_SPLATTING_SPLAT_TRAINING_HPP
#define KLARTRAUM_SPLATTING_SPLAT_TRAINING_HPP

#include <cstdint>
#include <memory>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/computegraphelement.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/vulkan_context.hpp"

/**
 * @brief Differentiable Gaussian splatting for training (Kerbl et al. 2023): the forward chain from Gaussian
 * parameters to a premultiplied image with coverage, as compute graph elements (shaders/splatting/).
 */
namespace klartraum::splatting {

/** @brief The trained parameters of a set of Gaussians, one row per Gaussian. */
struct SplatParameters {
    uint32_t count = 0;                              ///< number of Gaussians
    std::shared_ptr<TensorElement<float>> means;     ///< count x 4: position xyz, unused
    std::shared_ptr<TensorElement<float>> logScales; ///< count x 4: log of the scale along each axis, unused
    std::shared_ptr<TensorElement<float>> rotations; ///< count x 4: quaternion (x, y, z, w), normalized when used
    std::shared_ptr<TensorElement<float>> colors;    ///< count x 4: rgb, unused
    std::shared_ptr<TensorElement<float>> opacities; ///< count: logit of the opacity
};

/** @brief Creates the parameter tensors of @p count Gaussians. */
SplatParameters createSplatParameters(VulkanContext& vulkanContext, uint32_t count);

/** @brief Image size and the limits of the forward chain. */
struct SplatRasterOptions {
    uint32_t width = 256, height = 192; ///< image size in pixels
    uint32_t capacity = 1u << 20;       ///< most tile instances per image; more are dropped (and counted)
    float depthScale = 65536.0f;        ///< depth quantization steps per unit of clip w (2^20 steps: w < 16)
    float nearDepth = 0.05f;            ///< Gaussians with a smaller clip w are culled
};

/**
 * @brief The forward chain of one view: clear the counters, project and emit tile instances, clamp the count, radix
 * sort by (tile, depth), find each tile's range, blend.
 *
 * The camera of the drawn view comes from @p cameras (column-major 4x4 per view, clip from world) at the index in
 * slot @p viewSlot of @p viewSource: the element writing the one-element uint32 view index, or the buffer itself
 * with slot -1.
 */
class SplatForward {
public:
    SplatForward(VulkanContext& vulkanContext, const SplatParameters& parameters,
                 const std::shared_ptr<TensorElement<float>>& cameras, const ComputeGraphElementPtr& viewSource,
                 int viewSlot, const SplatRasterOptions& options);

    using UintBuffer = BufferElement<VulkanBuffer<uint32_t>>;

    SplatRasterOptions options;
    uint32_t tilesX = 0, tilesY = 0;

    /// Three float4 per Gaussian: (mean x, mean y, depth, radius), (conic xx, xy, yy, opacity), (rgb, 0).
    std::shared_ptr<TensorElement<float>> projected;
    /// Per pixel: premultiplied rgb and coverage (1 - final transmittance). Slot 3 of render.
    std::shared_ptr<TensorElement<float>> colors;
    /// Sort keys and values (the Gaussian) of the tile instances; the sorted ones end in keys and values.
    std::shared_ptr<UintBuffer> keys, values, keysScratch, valuesScratch;
    /// [0]: tile instances emitted (may exceed the capacity).
    std::shared_ptr<UintBuffer> counters;
    /// The number of instances sorted: the emitted count, at most the capacity.
    std::shared_ptr<UintBuffer> sortCount;
    /// Per tile: [start, end) of its sorted instances.
    std::shared_ptr<UintBuffer> tileRanges;
    /// Per pixel: how many of its tile's instances the blending went through.
    std::shared_ptr<UintBuffer> lastContributor;

    /// The elements in graph order; render is the last.
    ComputeGraphElementPtr clear, project, clampCount, sort, ranges, render;
};

/**
 * @brief The backward chain of SplatForward: the gradients of a loss with respect to the Gaussian parameters, from
 * the loss's gradients with respect to the rendered image (dL/d(rgb, coverage) per pixel, e.g. raster::imageLoss()
 * slot 3, taken from slot @p imageGradientSlot of @p imageGradientSource).
 *
 * The render backward walks every pixel's splats back to front and adds the gradients of each splat's screen mean,
 * conic, opacity and colour; the projection backward turns them into parameter gradients (bwd_diff of
 * shaders/splatting/splat_common.slang) and accumulates the densification statistics.
 */
class SplatBackward {
public:
    SplatBackward(VulkanContext& vulkanContext, const SplatParameters& parameters,
                  const std::shared_ptr<TensorElement<float>>& cameras, const ComputeGraphElementPtr& viewSource,
                  int viewSlot, const SplatForward& forward, const ComputeGraphElementPtr& imageGradientSource,
                  int imageGradientSlot);

    /// Twelve floats per Gaussian: gradients of mean xy, conic xx xy yy, opacity, rgb, 3 unused. Cleared every run.
    std::shared_ptr<TensorElement<float>> splatGradients;
    /// The parameter gradients, in the layout of the parameters (written every run, zero for culled Gaussians).
    SplatParameters gradients;
    /// Per Gaussian: the sum over runs of the length of its screen-mean gradient and the number of runs it was
    /// visible in (for densification). Never cleared by the graph.
    std::shared_ptr<TensorElement<float>> statistics;

    /// The elements in graph order; projectBackward is the last.
    ComputeGraphElementPtr clear, renderBackward, projectBackward;
};

} // namespace klartraum::splatting

#endif // KLARTRAUM_SPLATTING_SPLAT_TRAINING_HPP
