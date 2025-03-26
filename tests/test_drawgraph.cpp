#include <gtest/gtest.h>

#include <map>
#include <vector>

#include "klartraum/glfw_frontend.hpp"

#include "klartraum/drawgraph/drawgraph.hpp"

using namespace klartraum;

class Drawable : public DrawGraphElement {
    virtual const char* getName() const {
        return "Drawable";
    }
};

class BlurOp : public DrawGraphElement {
    virtual const char* getName() const {
        return "BlurOp";
    }

};

class NoiseOp : public DrawGraphElement {
    virtual const char* getName() const {
        return "NoiseOp";
    }

};

class AddOp : public DrawGraphElement {
    virtual const char* getName() const {
        return "AddOp";
    }

};

class CopyOp : public DrawGraphElement {
    virtual const char* getName() const {
        return "CopyOp";
    }

};

class ImageViewSrc : public DrawGraphElement {
    virtual const char* getName() const {
        return "ImageViewSrc";
    }
};

class FramebufferSrc : public DrawGraphElement {
public:
    FramebufferSrc(int index) : framebuffer_index(index) {}

    virtual const char* getName() const {
        return "FramebufferSrc";
    }
    uint32_t framebuffer_index;
};


TEST(DrawGraph, create) {
    klartraum::GlfwFrontend frontend;

    auto& core = frontend.getKlartraumCore();
    auto& vulkanKernel = core.getVulkanKernel();
    auto& device = vulkanKernel.getDevice();

    /*
    STEP 1: create the drawgraph elements
    */

    auto framebuffer_src = std::make_shared<FramebufferSrc>(0);
    auto drawable = std::make_shared<Drawable>();
    drawable->set_input(framebuffer_src);

    auto blur = std::make_shared<BlurOp>();
    blur->set_input(drawable);
    auto noise = std::make_shared<NoiseOp>();
    noise->set_input(blur);

    auto add = std::make_shared<AddOp>();
    add->set_input(blur, 0);
    add->set_input(noise, 1);

    auto copy = std::make_shared<CopyOp>();
    copy->set_input(add);

    
    /*
    STEP 2: create the drawgraph backend
    */
    
    // this traverses the drawgraph and creates the vulkan objects
    auto& drawgraph = DrawGraph();
    drawgraph.compileFrom(copy);

    drawgraph.submitTo(vulkanKernel.getGraphicsQueue());

    return;
}