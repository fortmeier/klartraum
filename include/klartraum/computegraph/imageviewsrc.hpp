#ifndef KLARTRAUM_COMPUTEGRAPH_FRAMEBUFFERSRC_HPP
#define KLARTRAUM_COMPUTEGRAPH_FRAMEBUFFERSRC_HPP

#include <map>
#include <vector>

#include <vulkan/vulkan.h>

#include "klartraum/computegraph/computegraphelement.hpp"

namespace klartraum {

class ImageViewSrc : public virtual ComputeGraphElement {
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

}

#endif // KLARTRAUM_COMPUTEGRAPH_FRAMEBUFFERSRC_HPP