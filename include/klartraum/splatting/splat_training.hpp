// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_SPLATTING_SPLAT_TRAINING_HPP
#define KLARTRAUM_SPLATTING_SPLAT_TRAINING_HPP

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include <glm/glm.hpp>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/computegraph.hpp"
#include "klartraum/computegraph/computegraphelement.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/layers/training.hpp"
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

/** @brief Gaussians on the host, in the layout of SplatParameters. */
struct SplatSet {
    std::vector<float> means, logScales, rotations, colors, opacities;
    uint32_t size() const { return uint32_t(opacities.size()); }
};

/**
 * @brief Isotropic Gaussians at @p points: scale @p scale, identity rotation, colour @p color, opacity @p opacity.
 */
SplatSet splatsAt(const std::vector<glm::vec3>& points, float scale, glm::vec3 color = glm::vec3(0.5f),
                  float opacity = 0.1f);

/**
 * @brief Renders @p splats from every camera: four floats per pixel and view (premultiplied rgb, coverage), views
 * one after another.
 */
std::vector<float> renderSplats(VulkanContext& vulkanContext, const SplatSet& splats,
                                const std::vector<glm::mat4>& cameras, const SplatRasterOptions& options);

/**
 * @brief The Gaussians mapped by a similarity transform (rotation, uniform scale, translation): positions,
 * orientations and scales follow it.
 */
SplatSet transformSplats(const SplatSet& splats, const glm::mat4& similarity);

/**
 * @brief Writes Gaussians as a binary PLY file in the layout of the reference implementation of Kerbl et al.
 * (positions, zero normals, degree-0 spherical harmonics, logit opacity, log scales, rotation w x y z), which common
 * splat viewers read.
 */
void writeSplatPly(const SplatSet& splats, const std::filesystem::path& path);

/** @brief Loss weight, optimizers and the densification schedule of a SplatTraining. */
struct SplatTrainingOptions {
    SplatRasterOptions raster;                 ///< image size and limits
    float maskWeight = 1.0f;                   ///< weight of the coverage term of raster::imageLoss()
    uint32_t steps = 6000;                     ///< planned steps: the learning rates decay over them
    training::AdamParameters means{1e-3f};     ///< optimizer of the positions (decays to a hundredth)
    training::AdamParameters logScales{5e-3f}; ///< optimizer of the log scales
    training::AdamParameters rotations{1e-3f}; ///< optimizer of the rotations
    training::AdamParameters colors{1e-2f};    ///< optimizer of the colours
    training::AdamParameters opacities{5e-2f}; ///< optimizer of the logit opacities
    uint32_t densifyFrom = 300;                ///< first step that densifies
    uint32_t densifyUntil = 3000;              ///< no densification from this step on
    uint32_t densifyInterval = 100;            ///< steps between densifications
    /// Gaussians whose mean screen-mean gradient (times the pixel count, so independent of the image size) exceeds
    /// this are cloned or split.
    float densifyGradient = 0.05f;
    float cloneScale = 0.02f;    ///< Gaussians up to this largest scale are cloned, larger ones split in two
    float pruneOpacity = 0.005f; ///< Gaussians fainter than this are removed when densifying
    float pruneScale = 0.5f;     ///< Gaussians with a larger scale are removed when densifying
    uint32_t maxCount = 400000;  ///< densification stops adding Gaussians at this count
    uint32_t seed = 1;           ///< seed of the split samples
};

/**
 * @brief Trains Gaussians on masked images seen by known cameras.
 *
 * One training step is one pre-recorded command buffer and one submit: advance the view index (on the GPU, cycling
 * through all cameras), SplatForward, raster::imageLoss() against the view's target, SplatBackward and an Adam step
 * per parameter group. Densification (adaptive density control of Kerbl et al.) runs on the host between steps:
 * it reads the Gaussians, their Adam moments and gradient statistics back, clones or splits Gaussians with large
 * screen-space gradients, removes faint and oversized ones and rebuilds the graph for the new count.
 */
class SplatTraining {
public:
    /**
     * @brief Builds the training of @p start on @p targets (four floats per pixel and view, views in camera order:
     * rgb premultiplied by the mask, and the mask).
     */
    SplatTraining(VulkanContext& vulkanContext, const SplatSet& start, const std::vector<glm::mat4>& cameras,
                  const std::vector<float>& targets, const SplatTrainingOptions& options);

    /** @brief Runs one training step on the next view, densifying first when the schedule says so. */
    void step();

    /** @brief The loss of the last step. */
    float loss() const;

    /** @brief The steps run so far. */
    uint32_t stepCount() const { return steps; }

    /** @brief The current Gaussians. */
    SplatSet splats() const;

    /** @brief The tile instances the last step emitted (more than the capacity means some were dropped). */
    uint32_t instanceCount() const;

    /** @brief Clones, splits and prunes Gaussians and rebuilds the graph (see the class description). */
    void densify();

private:
    using Tensor = TensorElement<float>;
    using UintBuffer = BufferElement<VulkanBuffer<uint32_t>>;

    // Builds the graph for `splats` with Adam moments `moments` (two per parameter group, in group order) and
    // the densification statistics `statistics` (two floats per Gaussian).
    void build(const SplatSet& splats, const std::vector<std::vector<float>>& moments,
               const std::vector<float>& statistics);

    // The context the graph runs on; it outlives this object.
    VulkanContext& vulkanContext;
    SplatTrainingOptions options;
    std::vector<glm::mat4> cameras;
    std::vector<float> targetValues;
    uint32_t steps = 0;
    uint32_t seed = 1;
    // The graph of the current Gaussian count and the tensors it reads and writes.
    SplatParameters parameters;
    std::vector<std::shared_ptr<Tensor>>
        moments; // two per parameter group: means, scales, rotations, colours, opacities
    std::shared_ptr<Tensor> cameraBuffer, targets, imageGradients, lossValue;
    std::shared_ptr<UintBuffer> viewIndex, stepBuffer;
    std::unique_ptr<SplatForward> forward;
    std::unique_ptr<SplatBackward> backward;
    std::unique_ptr<ComputeGraph> graph;
};

} // namespace klartraum::splatting

#endif // KLARTRAUM_SPLATTING_SPLAT_TRAINING_HPP
