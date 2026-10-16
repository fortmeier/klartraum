// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_GAUSSIAN_SPLATTING_FACTORY_HPP
#define KLARTRAUM_GAUSSIAN_SPLATTING_FACTORY_HPP

#include <memory>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/computegraphelement.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/draw_component.hpp" // for CameraUboType
#include "klartraum/vulkan_buffer.hpp"
#include "klartraum/vulkan_gaussian_splatting_types.hpp"

namespace klartraum {

class GaussianDataStandard;

/**
 * @brief Selects the Gaussian splatting backend that createGaussianSplatting() builds.
 */
enum class GsplatBackend {
    Compute, // VulkanGaussianSplatting — sort-per-tile, compute-rasterized (binned splatting)
    Raster,  // VulkanGaussianSplattingRaster — sort-once, hardware-rasterized
};

/**
 * @brief Creates a Gaussian splatting backend that renders @p model into @p imageViewSrc.
 *
 * Both backends take the same inputs and produce a single image-producing
 * ComputeGraphElement that KlartraumEngine::add() accepts, so switching @p backend
 * does not change the surrounding graph.
 *
 * The caller owns the GaussianDataStandard (e.g.
 * `std::make_shared<GaussianDataStandard>(vulkanContext, path)`); the same instance
 * can be passed to several backends or viewports, so the model is loaded only once.
 */
std::shared_ptr<ComputeGraphElement> createGaussianSplatting(VulkanContext& vulkanContext, GsplatBackend backend,
                                                             std::shared_ptr<ImageViewSrc> imageViewSrc,
                                                             std::shared_ptr<CameraUboType> cameraUBO,
                                                             std::shared_ptr<GaussianDataStandard> model,
                                                             GsplatConfig config = GsplatConfig{});

/**
 * @brief Creates a Gaussian splatting backend for Gaussians from any source, e.g. the outputs of a GaussianTransform or
 * GaussianMerge.
 *
 * The backend then depends on the elements that produce the buffers.
 */
std::shared_ptr<ComputeGraphElement> createGaussianSplatting(VulkanContext& vulkanContext, GsplatBackend backend,
                                                             std::shared_ptr<ImageViewSrc> imageViewSrc,
                                                             std::shared_ptr<CameraUboType> cameraUBO,
                                                             const GaussianSoABuffers& buffers,
                                                             GsplatConfig config = GsplatConfig{});

/**
 * @brief The depth buffer of a backend made by createGaussianSplatting() with GsplatConfig::outputDepth.
 *
 * Returns nullptr if the backend was made without depth output. See
 * GsplatConfig::outputDepth for its contents. Elements that read it in the same
 * graph must depend on the backend (and, for graphics stages, on a
 * BufferToGraphicsBarrier with the buffer).
 */
std::shared_ptr<BufferElement<VulkanBuffer<float>>>
getGaussianSplattingDepth(const std::shared_ptr<ComputeGraphElement>& splatting);

} // namespace klartraum

#endif // KLARTRAUM_GAUSSIAN_SPLATTING_FACTORY_HPP
