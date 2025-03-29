#ifndef KLARTRAUM_DRAWGRAPH_FRAMEBUFFERSRC_HPP
#define KLARTRAUM_DRAWGRAPH_FRAMEBUFFERSRC_HPP

#include <map>
#include <vector>

#include <vulkan/vulkan.h>

#include "klartraum/drawgraph/drawgraphelement.hpp"

namespace klartraum {

class ImageViewSrc : public DrawGraphElement {
public:
    ImageViewSrc(VkImageView imageView) {
        imageViews.push_back(imageView);
    };

    ImageViewSrc(std::vector<VkImageView> imageViews) {
        this->imageViews = imageViews;
    };

    virtual const char* getName() const {
        return "ImageViewSrc";
    }

    virtual void _record(VkCommandBuffer commandBuffer) {

    };

    std::vector<VkImageView> imageViews;

};

}

#endif // KLARTRAUM_DRAWGRAPH_FRAMEBUFFERSRC_HPP