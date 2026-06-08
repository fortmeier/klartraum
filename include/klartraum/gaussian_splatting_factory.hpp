#ifndef KLARTRAUM_GAUSSIAN_SPLATTING_FACTORY_HPP
#define KLARTRAUM_GAUSSIAN_SPLATTING_FACTORY_HPP

#include <memory>
#include <string>
#include <vector>

#include "klartraum/computegraph/computegraphelement.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/draw_component.hpp" // for CameraUboType
#include "klartraum/vulkan_gaussian_splatting_types.hpp"

namespace klartraum {

// Selects which Gaussian-splatting backend createGaussianSplatting builds
// (klartraum_rasterized_gs_backend_guide.md §7 step 6 "Selector").
enum class GsplatBackend {
    Compute, // VulkanGaussianSplatting — sort-per-tile, compute-rasterized (binned splatting)
    Raster,  // VulkanGaussianSplattingRaster — sort-once, hardware-rasterized
};

// Both backends share the same constructor shape (VulkanContext&, ImageViewSrc,
// CameraUboType, model source, GsplatConfig) and both produce a single
// image-producing ComputeGraphElement that KlartraumEngine::add accepts — so
// callers can switch backends via `backend` without touching any surrounding
// graph-wiring code.
std::shared_ptr<ComputeGraphElement> createGaussianSplatting(
    VulkanContext& vulkanContext,
    GsplatBackend backend,
    std::shared_ptr<ImageViewSrc> imageViewSrc,
    std::shared_ptr<CameraUboType> cameraUBO,
    std::string path,
    GsplatConfig config = GsplatConfig{});

std::shared_ptr<ComputeGraphElement> createGaussianSplatting(
    VulkanContext& vulkanContext,
    GsplatBackend backend,
    std::shared_ptr<ImageViewSrc> imageViewSrc,
    std::shared_ptr<CameraUboType> cameraUBO,
    std::vector<Gaussian3D> gaussians,
    GsplatConfig config = GsplatConfig{});

} // namespace klartraum

#endif // KLARTRAUM_GAUSSIAN_SPLATTING_FACTORY_HPP
