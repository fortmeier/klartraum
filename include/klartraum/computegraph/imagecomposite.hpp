#ifndef KLARTRAUM_COMPUTEGRAPH_IMAGECOMPOSITE_HPP
#define KLARTRAUM_COMPUTEGRAPH_IMAGECOMPOSITE_HPP

#include <cstdint>
#include <memory>
#include <string>

#include <vulkan/vulkan.h>

#include "klartraum/computegraph/generalcomputation.hpp"
#include "klartraum/computegraph/imageviewsrc.hpp"

namespace klartraum {

/** @brief How ImageComposite combines the source with the target. */
enum class CompositeMode : uint32_t {
    Replace = 0,  ///< The source's pixels replace the target's.
    Over = 1,     ///< The source is blended over the target by its alpha.
};

/** @brief Where ImageComposite draws the source on the target. */
enum class CompositeFit : uint32_t {
    Stretch = 0,  ///< Over the whole target, aspect ratio not kept.
    Fit = 1,      ///< As large as fits whole, centred; the rest of the target stays.
    Fill = 2,     ///< As small as covers the target, centred; the source is cropped.
};

// Matches shaders/image/composite.comp.
struct ImageCompositePushConstants {
    uint32_t srcWidth;
    uint32_t srcHeight;
    uint32_t dstWidth;
    uint32_t dstHeight;
    uint32_t mode;
    uint32_t fit;
};

/**
 * @brief Draws a source image onto a target and stands for the target's
 *        images: connect a renderer to it to render over the result.
 *
 * The target is written in place, per path, and keeps its contents where the
 * source is not drawn (or shows through, with CompositeMode::Over); it must be
 * in VK_IMAGE_LAYOUT_GENERAL, as ImageViewSrc leaves it. The source is only
 * read, so it may be an image every path shares, such as a SinglePathImage.
 * Both are ImageViewSrc elements (or elements passing one through a slot)
 * whose format stores as rgba8. Set the source, then the target.
 */
class ImageComposite : public ImageViewForward {
public:
    /**
     * @param sourceLayout The layout the source is in; it is read in, and
     *        left in, VK_IMAGE_LAYOUT_GENERAL.
     */
    ImageComposite(VulkanContext& vulkanContext, VkExtent2D sourceExtent, VkExtent2D targetExtent,
                   CompositeMode mode = CompositeMode::Replace, CompositeFit fit = CompositeFit::Stretch,
                   VkImageLayout sourceLayout = VK_IMAGE_LAYOUT_GENERAL)
        : draw_(std::make_shared<GeneralComputation<ImageCompositePushConstants>>(vulkanContext,
                                                                                  "shaders/image/composite.comp.spv")) {
        draw_->setName("Composite draw");
        draw_->setPushConstants({{sourceExtent.width, sourceExtent.height, targetExtent.width, targetExtent.height,
                                  static_cast<uint32_t>(mode), static_cast<uint32_t>(fit)}});
        draw_->setGroupCount((targetExtent.width + 7) / 8, (targetExtent.height + 7) / 8, 1);
        draw_->setImageLayoutTransition(0, sourceLayout, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
                                        VK_ACCESS_SHADER_READ_BIT);
        // The target's contents are kept.
        draw_->setImageLayoutTransition(1, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                        VK_ACCESS_MEMORY_WRITE_BIT,
                                        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    }

    const char* getType() const override { return "ImageComposite"; }

    /** @brief The image drawn: `element` itself, or what it passes through `slot`. */
    void setSource(ComputeGraphElementPtr element, int slot = -1) { draw_->setInput(element, 0, slot); }

    /** @brief The image drawn onto, which this element then stands for. */
    void setTarget(ComputeGraphElementPtr element, int slot = -1) {
        draw_->setInput(element, 1, slot);
        setInput(draw_, 0, 1);
    }

    /** @brief The compute element that draws, e.g. to name it. */
    const std::shared_ptr<GeneralComputation<ImageCompositePushConstants>>& draw() const { return draw_; }

private:
    std::shared_ptr<GeneralComputation<ImageCompositePushConstants>> draw_;
};

} // namespace klartraum

#endif // KLARTRAUM_COMPUTEGRAPH_IMAGECOMPOSITE_HPP
