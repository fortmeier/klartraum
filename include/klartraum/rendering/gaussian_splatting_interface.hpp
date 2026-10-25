// SPDX-FileCopyrightText: Dirk Fortmeier
//
// SPDX-License-Identifier: MIT

#ifndef KLARTRAUM_RENDERING_GAUSSIAN_SPLATTING_INTERFACE_HPP
#define KLARTRAUM_RENDERING_GAUSSIAN_SPLATTING_INTERFACE_HPP

#include <memory>

#include "klartraum/computegraph/bufferelement.hpp"
#include "klartraum/computegraph/computegraphgroup.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/computegraph/rendergraphelement.hpp"
#include "klartraum/vulkan_buffer.hpp"

namespace klartraum {

/**
 * @brief Common graph and depth-output interface of the Gaussian splatting backends.
 *
 * Implementations can be passed to KlartraumEngine::add() or used as a dependency
 * of another graph element, independently of its rendering implementation.
 */
class GaussianSplattingInterface : public ImageViewForward,
                                   public virtual RenderGraphElement,
                                   public virtual ComputeGraphGroup {
public:
    using DepthBuffer = BufferElement<VulkanBuffer<float>>;

    virtual ~GaussianSplattingInterface() = default;
    const char* getType() const override { return "GaussianSplattingInterface"; }
    VkImageLayout getOutputLayout() const override { return outputLayout; }

    /**
     * @brief Returns the per-path float depth buffer, or nullptr when depth output was not requested.
     *
     * See GsplatConfig::outputDepth for the depth convention. This is a storage
     * buffer, not a depth-attachment image. Consumers must depend on this backend
     * as well as bind the buffer, so the graph orders them after its producer.
     */
    virtual std::shared_ptr<DepthBuffer> getDepth() const = 0;

protected:
    void configureOutputLayout(VulkanContext& context) {
        outputLayout = source().getFinalLayoutOverride().value_or(context.hasSurface() ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
                                                                                       : VK_IMAGE_LAYOUT_GENERAL);
    }

private:
    VkImageLayout outputLayout = VK_IMAGE_LAYOUT_GENERAL;
};

} // namespace klartraum

#endif // KLARTRAUM_RENDERING_GAUSSIAN_SPLATTING_INTERFACE_HPP
