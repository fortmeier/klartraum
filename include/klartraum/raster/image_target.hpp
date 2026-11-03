// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_RASTER_IMAGE_TARGET_HPP
#define KLARTRAUM_RASTER_IMAGE_TARGET_HPP

#include <cstdint>
#include <memory>

#include "klartraum/computegraph/computegraphelement.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/computegraph/tensorelement.hpp"
#include "klartraum/vulkan_context.hpp"

namespace klartraum::raster {

/**
 * @brief Turns a picture rendered into @p image (e.g. Gaussians splatted into an OffscreenTarget cleared to
 * transparent black) into view @p view of training targets (MeshTraining::getTargets()): per pixel the rgb as drawn
 * and the alpha (coverage) as the mask.
 *
 * The picture is copied after @p renderer (which leaves the image in @p layout, and in which it is left) into a
 * buffer and converted; the returned element writes the targets. The image is of the swapchain format, 8 bits per
 * channel, and of the targets' size.
 */
ComputeGraphElementPtr imageTarget(VulkanContext& vulkanContext, const std::shared_ptr<ImageViewSrc>& image,
                                   VkImageLayout layout, const ComputeGraphElementPtr& renderer,
                                   const std::shared_ptr<TensorElement<float>>& targets, uint32_t view);

} // namespace klartraum::raster

#endif // KLARTRAUM_RASTER_IMAGE_TARGET_HPP
