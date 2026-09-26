#ifndef KLARTRAUM_COMPUTEGRAPH_IMAGERESAMPLE_HPP
#define KLARTRAUM_COMPUTEGRAPH_IMAGERESAMPLE_HPP

#include <cstdint>

#include <vulkan/vulkan.h>

#include "klartraum/computegraph/generalcomputation.hpp"

namespace klartraum {

enum class ResampleFilter : uint32_t { Nearest = 0, Bilinear = 1 };

// Matches shaders/image/resample.comp.
struct ImageResamplePushConstants {
    uint32_t srcWidth;
    uint32_t srcHeight;
    uint32_t dstWidth;
    uint32_t dstHeight;
    uint32_t filter;
};

/**
 * @brief Resamples an image into another image of any size.
 *
 * Input 0 is the source image, input 1 the destination; both are ImageViewSrc
 * elements (or elements passing one through a slot) whose format stores as
 * rgba8, such as OffscreenTarget or the swapchain. The whole destination is
 * written, so its previous contents are discarded.
 *
 * The source is read in VK_IMAGE_LAYOUT_GENERAL, transitioned from
 * `srcLayout` (e.g. TRANSFER_SRC_OPTIMAL for an OffscreenTarget a splatting
 * backend rendered into). Both images are left in GENERAL.
 */
class ImageResample : public GeneralComputation<ImageResamplePushConstants> {
public:
    ImageResample(VulkanContext& vulkanContext, VkExtent2D srcExtent, VkExtent2D dstExtent,
                  ResampleFilter filter = ResampleFilter::Bilinear,
                  VkImageLayout srcLayout = VK_IMAGE_LAYOUT_GENERAL)
        : GeneralComputation<ImageResamplePushConstants>(vulkanContext, "shaders/image/resample.comp.spv") {
        setPushConstants({{srcExtent.width, srcExtent.height, dstExtent.width, dstExtent.height,
                           static_cast<uint32_t>(filter)}});
        setGroupCount((dstExtent.width + 7) / 8, (dstExtent.height + 7) / 8, 1);
        setImageLayoutTransition(0, srcLayout, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
                                 VK_ACCESS_SHADER_READ_BIT);
        setImageLayoutTransition(1, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                 VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                 VK_ACCESS_SHADER_WRITE_BIT);
    }

    const char* getType() const override {
        return "ImageResample";
    }
};

} // namespace klartraum

#endif // KLARTRAUM_COMPUTEGRAPH_IMAGERESAMPLE_HPP
