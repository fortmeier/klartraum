// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_IMAGE_FILL_HPP
#define KLARTRAUM_IMAGE_FILL_HPP

#include <memory>

#include <glm/glm.hpp>

#include "klartraum/computegraph/computegraphelement.hpp"
#include "klartraum/computegraph/hostvalues.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/vulkan_context.hpp"

namespace klartraum {

/**
 * @brief Fills @p target with the colour in @p colour (rgba, one value) each time the graph runs, e.g. a background
 * that changes per frame of a recorded graph.
 *
 * Renderers that draw over the target's contents (the Gaussian splatting backends, a RenderPass that loads them)
 * then draw over this colour: make them depend on the returned element. The colour channels are written as rgba;
 * for a BGRA target red and blue swap, which greys do not mind.
 */
ComputeGraphElementPtr imageFill(VulkanContext& vulkanContext, const std::shared_ptr<ImageViewSrc>& target,
                                 const std::shared_ptr<HostValues<glm::vec4>>& colour);

} // namespace klartraum

#endif // KLARTRAUM_IMAGE_FILL_HPP
