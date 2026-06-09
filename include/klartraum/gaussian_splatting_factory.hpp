#ifndef KLARTRAUM_GAUSSIAN_SPLATTING_FACTORY_HPP
#define KLARTRAUM_GAUSSIAN_SPLATTING_FACTORY_HPP

#include <memory>

#include "klartraum/computegraph/computegraphelement.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/draw_component.hpp" // for CameraUboType
#include "klartraum/vulkan_gaussian_splatting_types.hpp"

namespace klartraum {

class GaussianDataStandard;

// Selects which Gaussian-splatting backend createGaussianSplatting builds
// (klartraum_rasterized_gs_backend_guide.md §7 step 6 "Selector").
enum class GsplatBackend {
    Compute, // VulkanGaussianSplatting — sort-per-tile, compute-rasterized (binned splatting)
    Raster,  // VulkanGaussianSplattingRaster — sort-once, hardware-rasterized
};

// Both backends share the same constructor shape (VulkanContext&, ImageViewSrc,
// CameraUboType, model, GsplatConfig) and both produce a single image-producing
// ComputeGraphElement that KlartraumEngine::add accepts — so callers can switch
// backends via `backend` without touching any surrounding graph-wiring code.
//
// The caller owns the GaussianDataStandard (build it via
// std::make_shared<GaussianDataStandard>(vulkanContext, path | gaussians)); the
// same instance can be passed to several backends/viewports to load the model
// only once.
std::shared_ptr<ComputeGraphElement> createGaussianSplatting(
    VulkanContext& vulkanContext,
    GsplatBackend backend,
    std::shared_ptr<ImageViewSrc> imageViewSrc,
    std::shared_ptr<CameraUboType> cameraUBO,
    std::shared_ptr<GaussianDataStandard> model,
    GsplatConfig config = GsplatConfig{});

} // namespace klartraum

#endif // KLARTRAUM_GAUSSIAN_SPLATTING_FACTORY_HPP
