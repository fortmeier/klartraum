#ifndef KLARTRAUM_COMPUTEGRAPH_FRAMEBUFFERSRC_HPP
#define KLARTRAUM_COMPUTEGRAPH_FRAMEBUFFERSRC_HPP

#include <map>
#include <optional>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "klartraum/computegraph/computegraphelement.hpp"

namespace klartraum {

/**
 * @brief Records a clear of `image` to `color` and leaves it in
 *        VK_IMAGE_LAYOUT_GENERAL for whatever reads or writes it next.
 *
 * The previous contents are discarded; the clear waits for all earlier work
 * on the queue (e.g. the previous frame's reads of a swapchain image).
 */
inline void recordClearImage(VkCommandBuffer commandBuffer, VkImage image, const VkClearColorValue& color) {
    VkImageSubresourceRange range{};
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.levelCount = 1;
    range.layerCount = 1;

    VkImageMemoryBarrier toTransfer{};
    toTransfer.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toTransfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toTransfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toTransfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransfer.image = image;
    toTransfer.subresourceRange = range;
    toTransfer.srcAccessMask = 0;
    toTransfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &toTransfer);

    vkCmdClearColorImage(commandBuffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &color, 1, &range);

    VkImageMemoryBarrier toGeneral = toTransfer;
    toGeneral.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    toGeneral.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    toGeneral.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                         nullptr, 0, nullptr, 1, &toGeneral);
}

class ImageViewSrcInterface : public virtual ComputeGraphElement {
public:
    virtual const char* getType() const {
        return "ImageViewSrcInterface";
    }

    virtual VkImageView& getImageView(uint32_t pathId) = 0;
    virtual VkImage& getImage(uint32_t pathId) = 0;
    virtual VkExtent2D& getImageExtent(uint32_t pathId) = 0;

    // The layout an image-writing consumer (gsplat backends, RenderPass) should
    // leave this target in after writing it. nullopt means "no opinion" — the
    // consumer falls back to its default (PRESENT_SRC for a presentable
    // swapchain image, GENERAL otherwise). Offscreen viewport targets return
    // TRANSFER_SRC_OPTIMAL so the Window composite can blit straight from them
    // (PRESENT_SRC is illegal on a non-swapchain image).
    virtual std::optional<VkImageLayout> getFinalLayoutOverride() const {
        return std::nullopt;
    }
};

/**
 * @brief The images a graph renders into or reads, one per path.
 *
 * By default the source clears its image to opaque black every time its path
 * runs, before any element that uses it, and leaves it in
 * VK_IMAGE_LAYOUT_GENERAL: renderers such as the Gaussian-splatting backends
 * draw over what the image holds, so a fresh frame starts from black.
 * setClear(false) keeps the previous contents instead; the image must then
 * already be in VK_IMAGE_LAYOUT_GENERAL. Clearing needs images created with
 * VK_IMAGE_USAGE_TRANSFER_DST_BIT.
 */
class ImageViewSrc : public virtual ImageViewSrcInterface {
public:
    ImageViewSrc() {};

    ImageViewSrc(VkImageView imageView) {
        imageViews.push_back(imageView);
    };

    ImageViewSrc(std::vector<VkImageView> imageViews) {
        this->imageViews = imageViews;
    };

    ImageViewSrc(std::vector<VkImageView> imageViews, std::vector<VkImage> images) {
        this->images = images;
        this->imageViews = imageViews;
    };

    ImageViewSrc(std::vector<VkImageView> imageViews, std::vector<VkImage> images, std::vector<VkExtent2D> imageExtents) {
        this->images = images;
        this->imageViews = imageViews;
        this->imageExtents = imageExtents;
    };

    virtual const char* getType() const {
        return "ImageViewSrc";
    }

    /** @brief Whether the image is cleared each time its path runs (default: yes). */
    void setClear(bool clear) { clear_ = clear; }
    bool clears() const { return clear_; }
    /** @brief The color the image is cleared to (default: opaque black). */
    void setClearColor(const VkClearColorValue& color) { clearColor_ = color; }
    const VkClearColorValue& clearColor() const { return clearColor_; }

    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) override {
        ComputeGraphElement::_record(commandBuffer, pathId);
        if (clear_ && pathId < images.size()) {
            recordClearImage(commandBuffer, images[pathId], clearColor_);
        }
    }

    virtual VkImageView& getImageView(uint32_t pathId) {
        if (pathId >= imageViews.size()) {
            throw std::runtime_error("pathId out of range!");
        }
        return imageViews[pathId];
    }

    virtual VkImage& getImage(uint32_t pathId) {
        if (pathId >= images.size()) {
            throw std::runtime_error("pathId out of range!");
        }
        return images[pathId];
    }

    virtual VkExtent2D& getImageExtent(uint32_t pathId) {
        if (pathId >= imageExtents.size()) {
            throw std::runtime_error("pathId out of range!");
        }
        return imageExtents[pathId];
    }

protected:
    // For subclasses (e.g. OffscreenTarget) that allocate their own images and
    // populate the handle vectors after construction.
    void setResources(std::vector<VkImageView> views,
                      std::vector<VkImage> imgs,
                      std::vector<VkExtent2D> exts) {
        imageViews   = std::move(views);
        images       = std::move(imgs);
        imageExtents = std::move(exts);
    }

private:
    std::vector<VkImageView> imageViews;
    std::vector<VkImage> images;
    std::vector<VkExtent2D> imageExtents;
    bool clear_ = true;
    VkClearColorValue clearColor_ = {{0.0f, 0.0f, 0.0f, 1.0f}};
};

/**
 * @brief Stands for the images of the ImageViewSrc at input 0, e.g. a target
 *        another element (at an output slot) writes: elements that use it
 *        then run after that element. It does not clear the images.
 */
class ImageViewForward : public ImageViewSrc {
public:
    ImageViewForward() { setClear(false); }

    const char* getType() const override { return "ImageViewForward"; }

    void checkInput(ComputeGraphElementPtr input, int index = 0) override {
        if (index != 0 || !std::dynamic_pointer_cast<ImageViewSrc>(input)) {
            throw std::runtime_error(std::string(getType()) + ": input 0 must be an ImageViewSrc");
        }
    }

    VkImageView& getImageView(uint32_t pathId) override { return source().getImageView(pathId); }
    VkImage& getImage(uint32_t pathId) override { return source().getImage(pathId); }
    VkExtent2D& getImageExtent(uint32_t pathId) override { return source().getImageExtent(pathId); }
    std::optional<VkImageLayout> getFinalLayoutOverride() const override {
        if (inputs.empty()) {
            return std::nullopt;
        }
        auto input =
            std::dynamic_pointer_cast<ImageViewSrc>(const_cast<ImageViewForward*>(this)->getInputElement(0));
        return input ? input->getFinalLayoutOverride() : std::nullopt;
    }

protected:
    ImageViewSrc& source() {
        auto input = std::dynamic_pointer_cast<ImageViewSrc>(getInputElement(0));
        if (!input) {
            throw std::runtime_error(std::string(getType()) + ": input 0 is not an ImageViewSrc");
        }
        return *input;
    }
};

/**
 * @brief Clears the images of the ImageViewSrc at input 0 to a color and
 *        stands for them: connect renderers to it instead of the source.
 *
 * For sources that do not clear themselves (ImageViewSrc::setClear(false)) or
 * should start from another color than black. The images are left in
 * VK_IMAGE_LAYOUT_GENERAL.
 */
class ClearImage : public ImageViewForward {
public:
    explicit ClearImage(const VkClearColorValue& color = {{0.0f, 0.0f, 0.0f, 1.0f}}) : color_(color) {}

    const char* getType() const override { return "ClearImage"; }

    void _record(VkCommandBuffer commandBuffer, uint32_t pathId) override {
        ComputeGraphElement::_record(commandBuffer, pathId);
        recordClearImage(commandBuffer, getImage(pathId), color_);
    }

private:
    VkClearColorValue color_;
};

class ImageSrc : public ComputeGraphElement {
public:
    ImageSrc(VkImage image) {
        images.push_back(image);
    };

    ImageSrc(std::vector<VkImage> images) {
        this->images = images;
    };

    virtual const char* getType() const {
        return "ImageSrc";
    }

    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) override {

    };

    std::vector<VkImage> images;
    
};

/**
 * @brief A compute graph element that performs image layout transitions for Vulkan images.
 * 
 * This class handles the transition of image layouts using VkImageMemoryBarrier.
 * It's designed to work with ImageViewSrc elements in a compute graph and
 * records the necessary commands to transition an image from one layout to another.
 * 
 * Default transition is from VK_IMAGE_LAYOUT_GENERAL to VK_IMAGE_LAYOUT_PRESENT_SRC_KHR.
 */
class ImageViewSrcTransition : public ComputeGraphElement {
    
public:

    ImageViewSrcTransition(VkImageLayout oldLayout, VkImageLayout newLayout) 
        : oldLayout(oldLayout), newLayout(newLayout) {}

    // With explicit synchronization scopes, e.g. for an image a raster pass
    // wrote as a color attachment.
    ImageViewSrcTransition(VkImageLayout oldLayout, VkImageLayout newLayout,
                           VkPipelineStageFlags srcStageMask, VkPipelineStageFlags dstStageMask,
                           VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask)
        : oldLayout(oldLayout), newLayout(newLayout), srcStageMask(srcStageMask), dstStageMask(dstStageMask),
          srcAccessMask(srcAccessMask), dstAccessMask(dstAccessMask) {}

    // create also default constructor
    ImageViewSrcTransition() = default;

    virtual const char* getType() const {
        return "ImageViewSrcTransition";
    }

    virtual void checkInput(ComputeGraphElementPtr input, int index = 0) override {
        ImageViewSrcInterface* imageViewSrc = std::dynamic_pointer_cast<ImageViewSrcInterface>(input).get();
        if (index == 0 && imageViewSrc == nullptr) {
            throw std::runtime_error("input is not an ImageViewSrcInterface!");
        }
    }

    virtual void _record(VkCommandBuffer commandBuffer, uint32_t pathId) override {
        ImageViewSrc* imageViewSrc = std::dynamic_pointer_cast<ImageViewSrc>(getInputElement(0)).get();
        if (imageViewSrc == nullptr) {
            throw std::runtime_error("input is not an ImageViewSrc!");
        }
        VkImage image = imageViewSrc->getImage(pathId);

        VkImageMemoryBarrier barrierBack = {};
        barrierBack.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrierBack.oldLayout = oldLayout;
        barrierBack.newLayout = newLayout;
        barrierBack.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrierBack.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrierBack.image = image;
        barrierBack.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrierBack.subresourceRange.baseMipLevel = 0;
        barrierBack.subresourceRange.levelCount = 1;
        barrierBack.subresourceRange.baseArrayLayer = 0;
        barrierBack.subresourceRange.layerCount = 1;

        barrierBack.srcAccessMask = srcAccessMask;
        barrierBack.dstAccessMask = dstAccessMask;

        vkCmdPipelineBarrier(
            commandBuffer,
            srcStageMask,
            dstStageMask,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrierBack);
    };

private:
    VkImageLayout oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkImageLayout newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    VkAccessFlags srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    VkAccessFlags dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;

};

}

#endif // KLARTRAUM_COMPUTEGRAPH_FRAMEBUFFERSRC_HPP