// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include <stdexcept>

#include "klartraum/rendering/gaussian_splatting_factory.hpp"

#include "klartraum/rendering/gaussian_data_standard.hpp"
#include "klartraum/rendering/vulkan_gaussian_splatting.hpp"
#include "klartraum/rendering/vulkan_gaussian_splatting_raster.hpp"

namespace klartraum {

std::shared_ptr<GaussianSplattingInterface> createGaussianSplatting(VulkanContext& vulkanContext, GsplatBackend backend,
                                                                    std::shared_ptr<ImageViewSrc> imageViewSrc,
                                                                    std::shared_ptr<CameraUboType> cameraUBO,
                                                                    std::shared_ptr<GaussianDataStandard> model,
                                                                    GsplatConfig config) {
    return createGaussianSplatting(vulkanContext, backend, imageViewSrc, cameraUBO, model->buffers(), config);
}

std::shared_ptr<GaussianSplattingInterface> createGaussianSplatting(VulkanContext& vulkanContext, GsplatBackend backend,
                                                                    std::shared_ptr<ImageViewSrc> imageViewSrc,
                                                                    std::shared_ptr<CameraUboType> cameraUBO,
                                                                    const GaussianSoABuffers& buffers,
                                                                    GsplatConfig config) {
    switch (backend) {
    case GsplatBackend::Compute:
        return vulkanContext.create<VulkanGaussianSplatting>(imageViewSrc, cameraUBO, buffers, config);
    case GsplatBackend::Raster:
        return vulkanContext.create<VulkanGaussianSplattingRaster>(imageViewSrc, cameraUBO, buffers, config);
    }
    throw std::runtime_error("createGaussianSplatting: unknown GsplatBackend");
}

} // namespace klartraum
