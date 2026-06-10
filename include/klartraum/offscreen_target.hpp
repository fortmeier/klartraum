#ifndef KLARTRAUM_OFFSCREEN_TARGET_HPP
#define KLARTRAUM_OFFSCREEN_TARGET_HPP

#include <vector>

#include <vulkan/vulkan.h>

#include "klartraum/computegraph/imageviewsrc.hpp"
#include "klartraum/vulkan_context.hpp"

namespace klartraum {

// An ImageViewSrc that owns its images instead of wrapping the swapchain's.
// Used as a viewport render target: a scene renders into it at full (0,0)..(w,h)
// extent exactly as it would the swapchain, and the Window composite later blits
// it into a sub-region of the real swapchain image.
//
// It allocates `numImages` images (one per swapchain image / compute-graph path),
// matching the swapchain format with usage STORAGE | COLOR_ATTACHMENT |
// TRANSFER_SRC so both the compute and raster backends can write it and the
// composite can blit from it. getFinalLayoutOverride() reports TRANSFER_SRC_OPTIMAL
// so the backends leave it ready for that blit (PRESENT_SRC is illegal here).
class OffscreenTarget : public ImageViewSrc {
public:
    OffscreenTarget(VulkanContext& vulkanContext, VkExtent2D extent, uint32_t numImages)
        : vulkanContext_(vulkanContext), extent_(extent) {
        auto& device = vulkanContext.getDevice();
        VkFormat format = vulkanContext.getSwapChainImageFormat();

        images_.resize(numImages);
        memories_.resize(numImages);
        views_.resize(numImages);

        for (uint32_t i = 0; i < numImages; ++i) {
            VkImageCreateInfo imageInfo{};
            imageInfo.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
            imageInfo.imageType     = VK_IMAGE_TYPE_2D;
            imageInfo.format        = format;
            imageInfo.extent        = { extent.width, extent.height, 1 };
            imageInfo.mipLevels     = 1;
            imageInfo.arrayLayers   = 1;
            imageInfo.samples       = VK_SAMPLE_COUNT_1_BIT;
            imageInfo.tiling        = VK_IMAGE_TILING_OPTIMAL;
            // COLOR_ATTACHMENT/STORAGE: the raster/compute backends write it;
            // TRANSFER_SRC: the composite blits from it; TRANSFER_DST: a viewport
            // may be cleared or uploaded to.
            imageInfo.usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                                    | VK_IMAGE_USAGE_STORAGE_BIT
                                    | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                                    | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            if (vkCreateImage(device, &imageInfo, nullptr, &images_[i]) != VK_SUCCESS) {
                throw std::runtime_error("OffscreenTarget: failed to create image!");
            }

            VkMemoryRequirements memReq;
            vkGetImageMemoryRequirements(device, images_[i], &memReq);
            VkMemoryAllocateInfo allocInfo{};
            allocInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.allocationSize  = memReq.size;
            allocInfo.memoryTypeIndex = vulkanContext.findMemoryType(
                memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            if (vkAllocateMemory(device, &allocInfo, nullptr, &memories_[i]) != VK_SUCCESS) {
                throw std::runtime_error("OffscreenTarget: failed to allocate image memory!");
            }
            vkBindImageMemory(device, images_[i], memories_[i], 0);

            VkImageViewCreateInfo viewInfo{};
            viewInfo.sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            viewInfo.image    = images_[i];
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format   = format;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.layerCount = 1;
            if (vkCreateImageView(device, &viewInfo, nullptr, &views_[i]) != VK_SUCCESS) {
                throw std::runtime_error("OffscreenTarget: failed to create image view!");
            }
        }

        setResources(views_, images_, std::vector<VkExtent2D>(numImages, extent));
    }

    ~OffscreenTarget() {
        auto& device = vulkanContext_.getDevice();
        for (auto view : views_)     vkDestroyImageView(device, view, nullptr);
        for (auto image : images_)   vkDestroyImage(device, image, nullptr);
        for (auto mem : memories_)   vkFreeMemory(device, mem, nullptr);
    }

    OffscreenTarget(const OffscreenTarget&) = delete;
    OffscreenTarget& operator=(const OffscreenTarget&) = delete;

    const char* getType() const override { return "OffscreenTarget"; }

    std::optional<VkImageLayout> getFinalLayoutOverride() const override {
        return VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    }

    VkExtent2D extent() const { return extent_; }

private:
    VulkanContext& vulkanContext_;
    VkExtent2D extent_;
    std::vector<VkImage>        images_;
    std::vector<VkDeviceMemory> memories_;
    std::vector<VkImageView>    views_;
};

} // namespace klartraum

#endif // KLARTRAUM_OFFSCREEN_TARGET_HPP
