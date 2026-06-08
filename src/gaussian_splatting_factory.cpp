#include <stdexcept>

#include "klartraum/gaussian_splatting_factory.hpp"

#include "klartraum/vulkan_gaussian_splatting.hpp"
#include "klartraum/vulkan_gaussian_splatting_raster.hpp"

namespace klartraum {

std::shared_ptr<ComputeGraphElement> createGaussianSplatting(
    VulkanContext& vulkanContext,
    GsplatBackend backend,
    std::shared_ptr<ImageViewSrc> imageViewSrc,
    std::shared_ptr<CameraUboType> cameraUBO,
    std::string path,
    GsplatConfig config)
{
    switch (backend) {
        case GsplatBackend::Compute:
            return vulkanContext.create<VulkanGaussianSplatting>(imageViewSrc, cameraUBO, path, config);
        case GsplatBackend::Raster:
            return vulkanContext.create<VulkanGaussianSplattingRaster>(imageViewSrc, cameraUBO, path, config);
    }
    throw std::runtime_error("createGaussianSplatting: unknown GsplatBackend");
}

std::shared_ptr<ComputeGraphElement> createGaussianSplatting(
    VulkanContext& vulkanContext,
    GsplatBackend backend,
    std::shared_ptr<ImageViewSrc> imageViewSrc,
    std::shared_ptr<CameraUboType> cameraUBO,
    std::vector<Gaussian3D> gaussians,
    GsplatConfig config)
{
    switch (backend) {
        case GsplatBackend::Compute:
            return vulkanContext.create<VulkanGaussianSplatting>(imageViewSrc, cameraUBO, std::move(gaussians), config);
        case GsplatBackend::Raster:
            return vulkanContext.create<VulkanGaussianSplattingRaster>(imageViewSrc, cameraUBO, std::move(gaussians), config);
    }
    throw std::runtime_error("createGaussianSplatting: unknown GsplatBackend");
}

} // namespace klartraum
