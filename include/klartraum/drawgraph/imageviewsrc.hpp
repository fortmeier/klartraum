#ifndef KLARTRAUM_DRAWGRAPH_FRAMEBUFFERSRC_HPP
#define KLARTRAUM_DRAWGRAPH_FRAMEBUFFERSRC_HPP

#include <map>
#include <vector>

#include <vulkan/vulkan.h>

#include "klartraum/drawgraph/drawgraphelement.hpp"

namespace klartraum {

class ImageViewSrc : public virtual DrawGraphElement {
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
    
private:
    std::vector<VkImageView> imageViews;
    std::vector<VkImage> images;
};

class ImageSrc : public DrawGraphElement {
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

#endif // KLARTRAUM_DRAWGRAPH_FRAMEBUFFERSRC_HPP