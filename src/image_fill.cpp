// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#include "klartraum/image_fill.hpp"

#include "klartraum/computegraph/generalcomputation.hpp"

namespace klartraum {

ComputeGraphElementPtr imageFill(VulkanContext& vulkanContext, const std::shared_ptr<ImageViewSrc>& target,
                                 const std::shared_ptr<HostValues<glm::vec4>>& colour) {
    auto fill = vulkanContext.create<GeneralComputation<>>("shaders/image/fill_image.comp.spv");
    fill->setName("ImageFill");
    fill->setInput(target, 0);
    fill->setInput(colour, 1);
    const VkExtent2D extent = target->getImageExtent(0);
    fill->setGroupCount((extent.width + 15) / 16, (extent.height + 15) / 16, 1);
    // Every pixel is overwritten: the previous contents need not be kept.
    fill->setImageLayoutTransition(0, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                   VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                   VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT);
    return fill;
}

} // namespace klartraum
