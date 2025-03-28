#ifndef KLARTRAUM_DRAWGRAPH_FRAMEBUFFERSRC_HPP
#define KLARTRAUM_DRAWGRAPH_FRAMEBUFFERSRC_HPP

#include <map>
#include <vector>

#include <vulkan/vulkan.h>

#include "klartraum/drawgraph/drawgraphelement.hpp"

namespace klartraum {

class FramebufferSrc : public DrawGraphElement {
public:
    FramebufferSrc(VkFramebuffer framebuffer) {
        framebuffers.push_back(framebuffer);
    };

    FramebufferSrc(std::vector<VkFramebuffer> framebuffers) {
        this->framebuffers = framebuffers;
    };

    virtual const char* getName() const {
        return "FramebufferSrc";
    }

    //uint32_t framebuffer_index;
    std::vector<VkFramebuffer> framebuffers;

    virtual void _record(VkCommandBuffer commandBuffer) {

    };

};

}

#endif // KLARTRAUM_DRAWGRAPH_FRAMEBUFFERSRC_HPP