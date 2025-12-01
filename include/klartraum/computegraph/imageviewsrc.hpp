#ifndef KLARTRAUM_COMPUTEGRAPH_FRAMEBUFFERSRC_HPP
#define KLARTRAUM_COMPUTEGRAPH_FRAMEBUFFERSRC_HPP

#include <map>
#include <vector>

#include <vulkan/vulkan.h>

#include "klartraum/computegraph/computegraphelement.hpp"

namespace klartraum {

class ImageViewSrcInterface : public virtual ComputeGraphElement {
public:
    virtual const char* getType() const {
        return "ImageViewSrcInterface";
    }

    virtual VkImageView& getImageView(uint32_t pathId) = 0;
    virtual VkImage& getImage(uint32_t pathId) = 0;
    virtual VkExtent2D& getImageExtent(uint32_t pathId) = 0;
};

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

    virtual void _record(VkCommandBuffer commandBuffer) {

    };

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
    
private:
    std::vector<VkImageView> imageViews;
    std::vector<VkImage> images;
    std::vector<VkExtent2D> imageExtents;
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

    virtual void _record(VkCommandBuffer commandBuffer) {

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

        barrierBack.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrierBack.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT;

        vkCmdPipelineBarrier(
            commandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrierBack);
    };

private:
    VkImageLayout oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkImageLayout newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

};

}

#endif // KLARTRAUM_COMPUTEGRAPH_FRAMEBUFFERSRC_HPP